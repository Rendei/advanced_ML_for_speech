"""AlphaZero-lite: self-play with batched MCTS -> replay buffer -> policy/value regression, repeated."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from .config import AZConfig, NetConfig
from .game import BOARD_MASK, F32, I64, play
from .history import EvalFn, ReplayBuffer, TrainLog
from .mcts import BatchMCTS, policy_from_visits, sample_actions
from .net import NetEvaluator, PVNet, n_parameters, save_checkpoint, to_tensor


def self_play(
    model: PVNet, device: torch.device, cfg: AZConfig, n_games: int, rng: np.random.Generator
) -> tuple[I64, I64, F32, F32, F32]:
    """Play ``n_games`` games in parallel against itself.

    Returns ``(cur, mask, pi, z, mean_outcome_stats)`` for every position visited: ``pi`` is the normalised root
    visit distribution (the training target of the policy head), ``z`` the final result for the player to move
    there. The last array holds ``[mean game length, first-player win share, draw share]``.
    """
    mcts = BatchMCTS(NetEvaluator(model, device), cfg.sims, rng, cfg.c_puct, cfg.dirichlet_alpha, cfg.dirichlet_eps)
    cur = np.zeros(n_games, dtype=np.int64)
    mask = np.zeros(n_games, dtype=np.int64)
    alive = np.arange(n_games)
    rec_cur: list[I64] = []
    rec_mask: list[I64] = []
    rec_pi: list[F32] = []
    rec_game: list[npt.NDArray[np.int64]] = []
    rec_ply: list[npt.NDArray[np.int64]] = []
    result = np.zeros(n_games, dtype=np.int64)  # from the point of view of the first player
    length = np.zeros(n_games, dtype=np.int64)
    ply = 0
    while alive.size:
        res = mcts.search(cur[alive], mask[alive], noise=True)
        pi = (res.visits / np.maximum(res.visits.sum(1, keepdims=True), 1.0)).astype(np.float32)
        temperature = 1.0 if ply < cfg.temp_moves else 0.0
        a = sample_actions(policy_from_visits(res.visits, temperature, rng), rng)
        rec_cur.append(cur[alive].copy())
        rec_mask.append(mask[alive].copy())
        rec_pi.append(pi)
        rec_game.append(alive.copy())
        rec_ply.append(np.full(alive.size, ply, dtype=np.int64))
        ncur, nmask, won = play(cur[alive], mask[alive], a)
        cur[alive], mask[alive] = ncur, nmask
        result[alive[won]] = 1 if ply % 2 == 0 else -1
        full = ~won & (nmask == BOARD_MASK)
        length[alive[won | full]] = ply + 1
        alive = alive[~won & ~full]
        ply += 1

    game = np.concatenate(rec_game)
    pl = np.concatenate(rec_ply)
    z = np.where(pl % 2 == 0, result[game], -result[game]).astype(np.float32)
    stats = np.array(
        [length.mean(), (result == 1).mean(), (result == 0).mean()],
        dtype=np.float32,
    )
    return np.concatenate(rec_cur), np.concatenate(rec_mask), np.concatenate(rec_pi), z, stats


def train_az(
    name: str,
    cfg: AZConfig,
    net_cfg: NetConfig,
    device: torch.device,
    seed: int,
    ckpt_dir: Path,
    evaluate: EvalFn | None = None,
    curve_every: int = 2,
    log_prefix: str = "",
) -> tuple[PVNet, TrainLog]:
    """Run the full AlphaZero-lite loop. ``evaluate`` (optional) is called on the current net to record curves."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PVNet(net_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.iterations * cfg.train_steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=cfg.lr * 0.05)
    buf = ReplayBuffer(cfg.buffer)
    log = TrainLog(name, n_params=n_parameters(model))
    # (iteration, tag); with very few iterations several tags may share an iteration (e.g. "init" and "p25")
    snap_at = [(0, "init"), (cfg.iterations // 4, "p25"), (cfg.iterations // 2, "p50"), (cfg.iterations, "final")]

    def snapshot(it: int) -> None:
        for _, tag in (s for s in snap_at if s[0] == it):
            path = ckpt_dir / f"{name}_{tag}.pt"
            save_checkpoint(model, path)
            log.checkpoints[tag] = path

    def record_eval(it: int, games: int, t0: float) -> None:
        if evaluate is None:
            return
        model.eval()
        log.games.append(games)
        log.seconds.append(time.time() - t0 - eval_time[0])
        t_eval = time.time()
        log.metrics.append(evaluate(model))
        eval_time[0] += time.time() - t_eval

    eval_time = [0.0]  # evaluation time is excluded from the training clock
    t0 = time.time()
    snapshot(0)
    record_eval(0, 0, t0)
    games_done = 0
    for it in range(1, cfg.iterations + 1):
        t_sp = time.time()
        model.eval()
        cur, mask, pi, z, stats = self_play(model, device, cfg, cfg.games_per_iter, rng)
        buf.add(cur, mask, pi, z)
        games_done += cfg.games_per_iter
        log.total_positions += len(cur)
        log.selfplay_seconds += time.time() - t_sp

        model.train()
        pl = vl = 0.0
        for _ in range(cfg.train_steps):
            bc, bm, bpi, bz = buf.sample(cfg.batch, rng)
            x = to_tensor(bc, bm, device)
            logits, v = model(x)
            target = torch.from_numpy(bpi).to(device)
            loss_p = -(target * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()
            loss_v = F.mse_loss(v, torch.from_numpy(bz).to(device))
            opt.zero_grad(set_to_none=True)
            (loss_p + loss_v).backward()  # type: ignore[no-untyped-call]
            opt.step()
            sched.step()
            pl += float(loss_p.item())
            vl += float(loss_v.item())
        log.loss_x.append(games_done)
        log.losses.append(
            {
                "policy": pl / cfg.train_steps,
                "value": vl / cfg.train_steps,
                "game_len": float(stats[0]),
                "first_win": float(stats[1]),
                "draw": float(stats[2]),
            }
        )
        snapshot(it)
        if it % curve_every == 0 or it == cfg.iterations:
            record_eval(it, games_done, t0)
        print(
            f"{log_prefix}[{name}] it {it}/{cfg.iterations} games {games_done} "
            f"loss p={log.losses[-1]['policy']:.3f} v={log.losses[-1]['value']:.3f} "
            f"len={stats[0]:.1f} X/draw={stats[1]:.2f}/{stats[2]:.2f} "
            + (f"acc={log.metrics[-1].get('acc_expert', float('nan')):.3f} " if log.metrics else "")
            + f"t={time.time() - t0 - eval_time[0]:.0f}s",
            flush=True,
        )
    log.total_seconds = time.time() - t0 - eval_time[0]
    log.total_games = games_done
    model.eval()
    return model, log
