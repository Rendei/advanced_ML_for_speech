"""Negamax double DQN trained purely from self-play (the value-based baseline against AlphaZero-lite).

Both sides are played by the same network. A transition ``(s, a, r, s')`` is stored from the point of view of the
player who moved; ``s'`` is what the *opponent* sees next, so the bootstrap target is the negated value of the
opponent's best reply (negamax):  ``y = r - gamma * Q_target(s', argmax_a' Q_online(s', a'))``  over legal ``a'``,
and ``y = r`` when the move ends the game (``r = +1`` for a win, 0 for a draw).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import DQNConfig, NetConfig
from .game import BOARD_MASK, I64, WIDTH, legal_moves, play
from .history import EvalFn, TrainLog, mirror_batch
from .net import QNet, n_parameters, save_checkpoint, to_tensor


class _Transitions:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cur = np.zeros(capacity, dtype=np.int64)
        self.mask = np.zeros(capacity, dtype=np.int64)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.bool_)
        self.ncur = np.zeros(capacity, dtype=np.int64)
        self.nmask = np.zeros(capacity, dtype=np.int64)
        self.size = 0
        self._head = 0

    def add(self, cur: I64, mask: I64, a: I64, r: np.ndarray, done: np.ndarray, ncur: I64, nmask: I64) -> None:
        idx = (self._head + np.arange(len(cur))) % self.capacity
        self.cur[idx], self.mask[idx], self.a[idx], self.r[idx] = cur, mask, a, r
        self.done[idx], self.ncur[idx], self.nmask[idx] = done, ncur, nmask
        self._head = (self._head + len(cur)) % self.capacity
        self.size = min(self.size + len(cur), self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[I64, I64, I64, np.ndarray, np.ndarray, I64, I64]:
        i = rng.integers(0, self.size, size=n)
        cur, mask, a, ncur, nmask = mirror_batch(
            self.cur[i], self.mask[i], self.a[i], self.ncur[i], self.nmask[i], rng.random(n) < 0.5
        )
        return cur, mask, a, self.r[i], self.done[i], ncur, nmask


def train_dqn(
    name: str,
    cfg: DQNConfig,
    net_cfg: NetConfig,
    device: torch.device,
    seed: int,
    ckpt_dir: Path,
    evaluate: EvalFn | None = None,
    n_curve_points: int = 20,
    log_prefix: str = "",
) -> tuple[QNet, TrainLog]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    online = QNet(net_cfg).to(device)
    target = QNet(net_cfg).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    mem = _Transitions(cfg.buffer)
    log = TrainLog(name, n_params=n_parameters(online))
    eval_time = [0.0]
    t0 = time.time()

    def do_eval(games: int) -> None:
        if evaluate is None:
            return
        online.eval()
        log.games.append(games)
        log.seconds.append(time.time() - t0 - eval_time[0])
        t = time.time()
        log.metrics.append(evaluate(online))
        eval_time[0] += time.time() - t

    snaps = {0: "init", cfg.games // 4: "p25", cfg.games // 2: "p50"}
    pending = sorted(snaps)
    save_checkpoint(online, ckpt_dir / f"{name}_init.pt")
    log.checkpoints["init"] = ckpt_dir / f"{name}_init.pt"
    pending.pop(0)
    do_eval(0)

    e = cfg.envs
    cur = np.zeros(e, dtype=np.int64)
    mask = np.zeros(e, dtype=np.int64)
    games_done = 0
    grad_steps = 0
    owed = 0.0  # fractional gradient steps owed
    next_eval = cfg.games / n_curve_points
    loss_acc, loss_n = 0.0, 0
    while games_done < cfg.games:
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * games_done / (cfg.eps_frac * cfg.games))
        legal = legal_moves(mask)
        online.eval()
        with torch.inference_mode():
            q = online(to_tensor(cur, mask, device)).cpu().numpy()
        greedy = np.where(legal, q, -np.inf).argmax(axis=1)
        random_a = (rng.random((e, WIDTH)) * legal - (~legal)).argmax(axis=1)
        a = np.where(rng.random(e) < eps, random_a, greedy).astype(np.int64)
        ncur, nmask, won = play(cur, mask, a)
        full = ~won & (nmask == BOARD_MASK)
        done = won | full
        mem.add(cur, mask, a, won.astype(np.float32), done, ncur, nmask)
        cur, mask = ncur, nmask
        cur = np.where(done, 0, cur)
        mask = np.where(done, 0, mask)
        games_done += int(done.sum())

        if mem.size >= cfg.batch * 4:
            owed += cfg.updates_per_step
            online.train()
            while owed >= 1.0:
                owed -= 1.0
                bc, bm, ba, br, bd, bnc, bnm = mem.sample(cfg.batch, rng)
                s = to_tensor(bc, bm, device)
                s2 = to_tensor(bnc, bnm, device)
                legal2 = torch.from_numpy(legal_moves(bnm)).to(device)
                with torch.no_grad():
                    q2_online = online(s2).masked_fill(~legal2, -1e9)
                    best = q2_online.argmax(dim=1, keepdim=True)
                    q2 = target(s2).gather(1, best).squeeze(1)
                    r_t = torch.from_numpy(br).to(device)
                    y = r_t - cfg.gamma * q2 * (~torch.from_numpy(bd).to(device)).float()
                q_sa = online(s).gather(1, torch.from_numpy(ba).to(device).unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(q_sa, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                opt.step()
                grad_steps += 1
                loss_acc += float(loss.item())
                loss_n += 1
                if grad_steps % cfg.target_every == 0:
                    target.load_state_dict(online.state_dict())

        while pending and games_done >= pending[0]:
            tag = snaps[pending.pop(0)]
            save_checkpoint(online, ckpt_dir / f"{name}_{tag}.pt")
            log.checkpoints[tag] = ckpt_dir / f"{name}_{tag}.pt"
        if games_done >= next_eval:
            next_eval += cfg.games / n_curve_points
            if loss_n:
                log.loss_x.append(games_done)
                log.losses.append({"td": loss_acc / loss_n, "eps": eps})
                loss_acc, loss_n = 0.0, 0
            do_eval(games_done)
            print(
                f"{log_prefix}[{name}] games {games_done}/{cfg.games} eps={eps:.2f} "
                + (f"td={log.losses[-1]['td']:.4f} " if log.losses else "")
                + (f"acc={log.metrics[-1].get('acc_expert', float('nan')):.3f} " if log.metrics else "")
                + f"t={time.time() - t0 - eval_time[0]:.0f}s",
                flush=True,
            )

    save_checkpoint(online, ckpt_dir / f"{name}_final.pt")
    log.checkpoints["final"] = ckpt_dir / f"{name}_final.pt"
    if not log.games or log.games[-1] < games_done:
        do_eval(games_done)
    log.total_seconds = time.time() - t0 - eval_time[0]
    log.selfplay_seconds = log.total_seconds
    log.total_games = games_done
    log.total_positions = int(mem.size)
    online.eval()
    return online, log
