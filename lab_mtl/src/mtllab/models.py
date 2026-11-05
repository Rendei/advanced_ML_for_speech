"""Shared trunk (audio-text fusion from lab 1) with one small head per task.

Notation: B batch, Ta audio frames (12.5 Hz), Tt text tokens, D=768 encoder dim, d model dim, z_dim trunk output.

    hard parameter sharing:     x_a, x_t --> Trunk (shared) --> z --+--> Head_emotion   --> logits [B, 4]
                                                                    +--> Head_valence   --> value  [B, 1]
                                                                    +--> Head_arousal   --> value  [B, 1]
                                                                    +--> Head_dominance --> value  [B, 1]
                                                                    +--> Head_gender    --> logits [B, 2]

All tasks read the same vector ``z``, so the gradients of every task meet in the trunk: that is where tasks can help
each other (shared features) or fight (conflicting gradients). A "single-task" model is the same network with one head.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import Config, task_dim
from .tensors import Batch


class Encoder(nn.Module):
    """[B, T, D] -> [B, T, d]: projection, sinusoidal positions and one Transformer layer.

    h_t = Dropout(LN(W x_t + b)) + PE_t                    (D=768 -> d)
    PE[t, 2i] = sin(t / 10000^(2i/d)),  PE[t, 2i+1] = cos(t / 10000^(2i/d))
    H   = TransformerLayer(h)        # pre-norm: h <- h + MHA(LN h);  h <- h + FFN(LN h)
    """

    def __init__(self, d_in: int, cfg: Config) -> None:
        super().__init__()
        d = cfg.d_model
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.Dropout(cfg.dropout))
        self.layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, 2 * d, cfg.dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.d = d

    def _positions(self, t: int, device: torch.device) -> Tensor:
        pos = torch.arange(t, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d, 2, device=device) * (-math.log(10_000.0) / self.d))
        pe = torch.zeros(t, self.d, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        h = self.proj(x)  # [B, T, d]
        h = h + self._positions(h.shape[1], h.device)
        out: Tensor = self.layer(h, src_key_padding_mask=~mask)  # [B, T, d]
        return out


class AttnPool(nn.Module):
    """Masked attention pooling: [B, T, d] -> [B, d].

    s_t = w^T h_t + b,   alpha = softmax_t(s)  over the real (non-padding) steps,   z = sum_t alpha_t h_t
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, h: Tensor, mask: Tensor) -> Tensor:
        s = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))  # [B, T]
        pooled: Tensor = torch.einsum("bt,btd->bd", torch.softmax(s, dim=-1), h)
        return pooled


class CrossBlock(nn.Module):
    """Queries from one modality attend to keys/values of the other (pre-norm, residual, FFN).

    q' = q + Dropout( MHA( LN(q), LN(kv), LN(kv) ) )          # attend to the other modality
    q'' = q' + Dropout( FFN( LN(q') ) ),   FFN(x) = W2 GELU(W1 x)    (d -> 2d -> d)
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d = cfg.d_model
        self.norm_q, self.norm_kv, self.norm_ff = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, q: Tensor, kv: Tensor, kv_mask: Tensor) -> Tensor:
        """q [B, Tq, d], kv [B, Tk, d], kv_mask [B, Tk] -> q' [B, Tq, d]."""
        kv_n = self.norm_kv(kv)
        out, _ = self.attn(self.norm_q(q), kv_n, kv_n, key_padding_mask=~kv_mask, need_weights=False)
        q = q + self.drop(out)
        updated: Tensor = q + self.drop(self.ff(self.norm_ff(q)))
        return updated


class Trunk(nn.Module):
    """The shared part: audio-text fusion that turns a Batch into one vector ``z`` [B, z_dim].

        concat  : z = [Pool_a(H_a) ; Pool_t(H_t)]                                   z_dim = 2d
        cross   : Q_0 = H_t, Q_l = CrossBlock_l(Q_{l-1}; H_a), z = Pool_t(Q_L)       z_dim = d   (Q=text, K/V=audio)
        bicross : cross plus the mirrored stream (Q=audio, K/V=text), both pooled     z_dim = 2d
    followed by LayerNorm(z).
    """

    def __init__(self, kind: str, cfg: Config, d_in: int) -> None:
        super().__init__()
        if kind not in {"concat", "cross", "bicross"}:
            raise ValueError(f"unknown trunk {kind!r}")
        self.kind = kind
        d, n_layers = cfg.d_model, cfg.n_cross_layers
        self.enc_a = Encoder(d_in, cfg)
        self.enc_t = Encoder(d_in, cfg)
        self.t_from_a = nn.ModuleList(CrossBlock(cfg) for _ in range(n_layers)) if kind != "concat" else None
        self.a_from_t = nn.ModuleList(CrossBlock(cfg) for _ in range(n_layers)) if kind == "bicross" else None
        self.pool_t = AttnPool(d)
        self.pool_a = AttnPool(d) if kind != "cross" else None
        self.z_dim = d if kind == "cross" else 2 * d
        self.norm = nn.LayerNorm(self.z_dim)

    def forward(self, b: Batch) -> Tensor:
        ha = self.enc_a(b.audio, b.audio_mask)  # H_a [B, Ta, d]
        ht = self.enc_t(b.text, b.text_mask)  # H_t [B, Tt, d]
        if self.t_from_a is None:  # concat
            assert self.pool_a is not None
            z = torch.cat([self.pool_a(ha, b.audio_mask), self.pool_t(ht, b.text_mask)], dim=-1)  # [B, 2d]
        else:
            q = ht
            for block in self.t_from_a:  # Q = text, K/V = audio
                q = block(q, ha, b.audio_mask)
            z = self.pool_t(q, b.text_mask)  # [B, d]
            if self.a_from_t is not None:  # bicross
                assert self.pool_a is not None
                p = ha
                for block in self.a_from_t:  # Q = audio, K/V = text
                    p = block(p, ht, b.text_mask)
                z = torch.cat([z, self.pool_a(p, b.audio_mask)], dim=-1)  # [B, 2d]
        out: Tensor = self.norm(z)
        return out


class MTLNet(nn.Module):
    """Trunk + one head per task. Head of a task = Dropout -> Linear(z_dim, h) -> GELU -> Linear(h, task_dim)."""

    def __init__(self, tasks: tuple[str, ...], cfg: Config, d_in: int) -> None:
        super().__init__()
        self.tasks = tasks
        self.trunk = Trunk(cfg.trunk, cfg, d_in)
        self.heads = nn.ModuleDict(
            {
                t: nn.Sequential(
                    nn.Dropout(cfg.dropout),
                    nn.Linear(self.trunk.z_dim, cfg.head_hidden),
                    nn.GELU(),
                    nn.Linear(cfg.head_hidden, task_dim(t)),
                )
                for t in tasks
            }
        )

    def forward(self, b: Batch) -> dict[str, Tensor]:
        """Batch -> {task: output [B, task_dim]} (logits for classification, z-scored value for regression)."""
        z = self.trunk(b)  # [B, z_dim]
        outputs: dict[str, Tensor] = {}
        for t in self.tasks:
            out: Tensor = self.heads[t](z)
            outputs[t] = out
        return outputs

    def shared_parameters(self) -> list[nn.Parameter]:
        """Parameters whose gradient mixes all tasks (the trunk). Head parameters get the gradient of one task only."""
        return list(self.trunk.parameters())
