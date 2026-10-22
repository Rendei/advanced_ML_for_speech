"""Trainable heads on top of the frozen encoders.

Notation: B batch, Ta audio frames (12.5 Hz), Tt text tokens, D=768 encoder dim, d model dim, C classes.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import Config
from .tensors import Batch


class Encoder(nn.Module):
    """[B, T, D] -> [B, T, d]: projection, sinusoidal positions and one Transformer layer.

        h_t = Dropout(LN(W x_t + b)) + PE_t                    (D=768 -> d)
        PE[t, 2i] = sin(t / 10000^(2i/d)),  PE[t, 2i+1] = cos(t / 10000^(2i/d))
        H   = TransformerLayer(h)        # pre-norm: h <- h + MHA(LN h);  h <- h + FFN(LN h)

    Frozen WavLM/RoBERTa features are already contextual, so one light layer only adapts them to the task;
    PE tells the later cross-attention *where in time* a frame/token is.
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
        return torch.einsum("bt,btd->bd", torch.softmax(s, dim=-1), h)


class CrossBlock(nn.Module):
    """Queries from one modality attend to keys/values of the other (pre-norm, residual, FFN).

        q' = q + Dropout( MHA( LN(q), LN(kv), LN(kv) ) )          # attend to the other modality
        q'' = q' + Dropout( FFN( LN(q') ) ),   FFN(x) = W2 GELU(W1 x)    (d -> 2d -> d)

    Per head (d_head = d / n_heads):  A = softmax( Q K^T / sqrt(d_head) + M ),  out = A V,  M = -inf on padding
    frames.  Heads are concatenated and projected by W_o.  Returned ``A`` is the head-average, [B, Tq, Tk].
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d = cfg.d_model
        self.norm_q, self.norm_kv, self.norm_ff = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, q: Tensor, kv: Tensor, kv_mask: Tensor) -> tuple[Tensor, Tensor]:
        """q [B, Tq, d], kv [B, Tk, d], kv_mask [B, Tk] -> (q' [B, Tq, d], attention [B, Tq, Tk])."""
        kv_n = self.norm_kv(kv)
        out, w = self.attn(self.norm_q(q), kv_n, kv_n, key_padding_mask=~kv_mask, need_weights=True)
        q = q + self.drop(out)
        return q + self.drop(self.ff(self.norm_ff(q))), w



class Model(nn.Module):
    """One class for all five variants; ``kind`` in audio | text | concat | cross | bicross.

    Math (d = model dim, L = number of cross layers, ``Pool`` = AttnPool, ``Enc`` = Encoder):

        H_a = Enc_a(x_a) in R^{Ta x d}          H_t = Enc_t(x_t) in R^{Tt x d}      (frozen features -> d)

        audio   : z = Pool_a(H_a)                                                  z in R^d
        text    : z = Pool_t(H_t)                                                  z in R^d
        concat  : z = [Pool_a(H_a) ; Pool_t(H_t)]                                  z in R^{2d}
        cross   : Q_0 = H_t,  Q_l = CrossBlock_l(Q_{l-1}; H_a),  z = Pool_t(Q_L)   z in R^d
        bicross : cross as above, plus P_0 = H_a,  P_l = CrossBlock'_l(P_{l-1}; H_t),
                  z = [Pool_t(Q_L) ; Pool_a(P_L)]                                  z in R^{2d}

        logits = W LN(z) + b   (dropout before W),   p = softmax(logits)           p in R^C

    ``CrossBlock(Q; K)`` is: every query token i receives a mixture of the *other* modality's vectors,
    ``c_i = sum_j A_ij V_j`` with attention ``A = softmax(Q K^T / sqrt(d_head))`` (rows sum to 1, padding
    frames masked), and adds it to its own vector through a residual connection (see ``CrossBlock``).
    So "text queries, audio keys/values" means: each word asks "which moments of the speech matter to me?".
    """

    def __init__(self, kind: str, cfg: Config, d_in: int) -> None:
        super().__init__()
        if kind not in {"audio", "text", "concat", "cross", "bicross"}:
            raise ValueError(kind)
        self.kind = kind
        d = cfg.d_model
        n_layers = cfg.n_cross_layers
        # Unimodal encoders (a modality that a variant does not use is simply not created).
        self.enc_a = Encoder(d_in, cfg) if kind != "text" else None
        self.enc_t = Encoder(d_in, cfg) if kind != "audio" else None
        # Stacked cross-attention: t_from_a has Q = text, K/V = audio; a_from_t is the mirror image.
        has_cross = kind in {"cross", "bicross"}
        self.t_from_a = nn.ModuleList(CrossBlock(cfg) for _ in range(n_layers)) if has_cross else None
        self.a_from_t = nn.ModuleList(CrossBlock(cfg) for _ in range(n_layers)) if kind == "bicross" else None
        # Sequence -> vector pooling, one per stream that reaches the classifier.
        self.pool_a = AttnPool(d) if kind in {"audio", "concat", "bicross"} else None
        self.pool_t = AttnPool(d) if kind in {"text", "concat", "cross", "bicross"} else None
        # Classifier on z; z is d-dimensional, or 2d when two pooled vectors are concatenated.
        z_dim = 2 * d if kind in {"concat", "bicross"} else d
        self.head = nn.Sequential(nn.LayerNorm(z_dim), nn.Dropout(cfg.dropout), nn.Linear(z_dim, cfg.n_classes))
        # Head-averaged attention of the last "text <- audio" layer, [B, Tt, Ta]; only used for the plot.
        self.last_attention: Tensor | None = None

    def _cross_stream(
        self, blocks: nn.ModuleList, query: Tensor, context: Tensor, context_mask: Tensor, keep_attention: bool
    ) -> Tensor:
        """Q_l = CrossBlock_l(Q_{l-1}; context) for l = 1..L, shape [B, Tq, d] throughout."""
        for block in blocks:
            query, attention = block(query, context, context_mask)
            if keep_attention:
                self.last_attention = attention  # [B, Tq, Tk]
        return query

    def forward(self, b: Batch) -> Tensor:
        """Batch -> logits [B, C]."""
        ha = self.enc_a(b.audio, b.audio_mask) if self.enc_a is not None else None  # H_a [B, Ta, d]
        ht = self.enc_t(b.text, b.text_mask) if self.enc_t is not None else None  # H_t [B, Tt, d]

        if self.kind == "audio":
            assert ha is not None and self.pool_a is not None
            z = self.pool_a(ha, b.audio_mask)  # [B, d]
        elif self.kind == "text":
            assert ht is not None and self.pool_t is not None
            z = self.pool_t(ht, b.text_mask)  # [B, d]
        elif self.kind == "concat":
            assert ha is not None and ht is not None and self.pool_a is not None and self.pool_t is not None
            z = torch.cat([self.pool_a(ha, b.audio_mask), self.pool_t(ht, b.text_mask)], dim=-1)  # [B, 2d]
        else:  # cross | bicross
            assert ha is not None and ht is not None and self.t_from_a is not None and self.pool_t is not None
            q = self._cross_stream(self.t_from_a, ht, ha, b.audio_mask, keep_attention=True)  # Q_L [B, Tt, d]
            z = self.pool_t(q, b.text_mask)  # [B, d]
            if self.kind == "bicross":
                assert self.a_from_t is not None and self.pool_a is not None
                p = self._cross_stream(self.a_from_t, ha, ht, b.text_mask, keep_attention=False)  # P_L [B, Ta, d]
                z = torch.cat([z, self.pool_a(p, b.audio_mask)], dim=-1)  # [B, 2d]

        logits: Tensor = self.head(z)  # [B, C]
        return logits
