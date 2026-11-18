"""The whole experiment: train three agents, evaluate everyone against the perfect solver, run the tournament."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from .config import Config
from .dqn import train_dqn
from .eda import EdaResults, count_positions, first_moves, random_play
from .evaluation import (
    PositionSet,
    Tournament,
    bootstrap_ci,
    elo_with_ci,
    make_position_sets,
    run_tournament,
    score_player,
    summarise,
)
from .game import F32, I64, WIDTH, from_moves, legal_moves, mirror_bits
from .history import EvalFn, TrainLog
from .mcts import BatchMCTS, RolloutEvaluator
from .net import NetEvaluator, PVNet, QNet, load_pv, load_q, n_parameters, to_tensor
from .players import (
    GreedyPlayer,
    MCTSPlayer,
    MinimaxPlayer,
    PerfectPlayer,
    Player,
    PolicyPlayer,
    QPlayer,
    RandomPlayer,
)
from .solver import Solver
from .train_az import train_az

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class AgentEval:
    name: str
    family: str  # baseline | uct | dqn | az | az-lite | perfect
    sims: int  # search simulations per move (0 = no search)
    summary: dict[str, dict[str, float]]  # position set -> metric -> value
    ci: dict[str, dict[str, tuple[float, float]]]  # position set -> metric -> 95% bootstrap interval


@dataclass(frozen=True)
class SweepPoint:
    net: str  # "final" | "init" | "uct"
    sims: int
    summary: dict[str, dict[str, float]]
    seconds_per_position: float


@dataclass(frozen=True)
class ValueAnalysis:
    class_mean: dict[int, float]  # true result (-1/0/1) -> mean value head output on the expert set
    class_std: dict[int, float]
    sign_accuracy: float  # share of positions where the value head names the right result (thresholds +-1/3)
    corr: float
    ply_edges: list[int]
    mae_by_ply: list[float]
    traces: list[ValueTrace]


@dataclass(frozen=True)
class ValueTrace:
    title: str
    plies: list[int]
    net_value: list[float]  # for the first player (X)
    true_value: list[int]  # solver result for the first player (X)


@dataclass(frozen=True)
class PolicyExample:
    moves: str
    raw: list[float]
    visits: list[float]
    best: list[bool]
    raw_choice: int
    mcts_choice: int
    value: int


@dataclass
class Results:
    profile: str
    seed: int
    device: str
    sets: dict[str, PositionSet]
    eda: EdaResults
    logs: dict[str, TrainLog]
    agents: list[AgentEval]
    sweep: list[SweepPoint]
    tournament: Tournament
    elo: F64
    elo_lo: F64
    elo_hi: F64
    value: ValueAnalysis
    examples: list[PolicyExample]
    seconds: float
    timings: dict[str, float] = field(default_factory=dict)
    shapes: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
def make_eval_fn(sets: dict[str, PositionSet], device: torch.device, seed: int) -> EvalFn:
    """Metrics recorded along the learning curves: raw-policy (or greedy-Q) accuracy on both test sets."""
    rng = np.random.default_rng(seed)

    def fn(model: nn.Module) -> dict[str, float]:
        player: Player
        if isinstance(model, PVNet):
            player = PolicyPlayer(model, device, rng, "tmp")
        else:
            assert isinstance(model, QNet)
            player = QPlayer(model, device, rng, "tmp")
        out: dict[str, float] = {}
        for k, ps in sets.items():
            out.update({f"{m}_{k}": v for m, v in summarise(ps, score_player(player, ps)).items()})
        if isinstance(model, PVNet):
            out["value_corr"] = _value_corr(model, sets["expert"], device)
        return out

    return fn


def _value_corr(model: PVNet, ps: PositionSet, device: torch.device) -> float:
    v = _value_head(model, ps.cur, ps.mask, device)
    if np.std(v) < 1e-9:
        return 0.0
    return float(np.corrcoef(v, ps.value.astype(np.float64))[0, 1])


def _value_head(model: PVNet, cur: I64, mask: I64, device: torch.device) -> F32:
    model.eval()
    with torch.inference_mode():
        _, v = model(to_tensor(cur, mask, device))
    return np.asarray(v.cpu().numpy(), dtype=np.float32)


def _eval_agent(
    player: Player, family: str, sims: int, sets: dict[str, PositionSet], n_boot: int, rng: np.random.Generator
) -> AgentEval:
    summary: dict[str, dict[str, float]] = {}
    ci: dict[str, dict[str, tuple[float, float]]] = {}
    for k, ps in sets.items():
        sc = score_player(player, ps)
        summary[k] = summarise(ps, sc)
        nt = ps.nontrivial
        ci[k] = {
            "acc_nt": bootstrap_ci(sc.correct[nt], n_boot, rng),
            "blunder_nt": bootstrap_ci(sc.blunder[nt], n_boot, rng),
            "acc": bootstrap_ci(sc.correct, n_boot, rng),
        }
    return AgentEval(player.name, family, sims, summary, ci)


def _value_analysis(
    model: PVNet,
    device: torch.device,
    sets: dict[str, PositionSet],
    solver: Solver,
    games: list[tuple[str, list[int]]],
) -> ValueAnalysis:
    ps = sets["expert"]
    v = _value_head(model, ps.cur, ps.mask, device).astype(np.float64)
    truth = ps.value
    class_mean = {c: float(v[truth == c].mean()) for c in (-1, 0, 1) if (truth == c).any()}
    class_std = {c: float(v[truth == c].std()) for c in (-1, 0, 1) if (truth == c).any()}
    pred = np.where(v > 1 / 3, 1, np.where(v < -1 / 3, -1, 0))
    edges = [4, 10, 16, 22, 28, 35]
    mae = []
    for lo, hi in itertools.pairwise(edges):
        sel = (ps.ply >= lo) & (ps.ply < hi)
        mae.append(float(np.abs(v[sel] - truth[sel]).mean()) if sel.any() else float("nan"))
    traces: list[ValueTrace] = []
    for title, cols in games:
        plies: list[int] = []
        nv: list[float] = []
        tv: list[int] = []
        for k in range(len(cols)):
            prefix = "".join(str(c + 1) for c in cols[:k])
            cur, mask = from_moves(prefix)
            x_sign = 1.0 if k % 2 == 0 else -1.0  # the value head speaks for the player to move
            val = float(_value_head(model, np.array([cur]), np.array([mask]), device)[0])
            plies.append(k)
            nv.append(x_sign * val)
            tv.append(int(x_sign) * solver.value(prefix))
        traces.append(ValueTrace(title, plies, nv, tv))
    return ValueAnalysis(
        class_mean,
        class_std,
        float((pred == truth).mean()),
        float(np.corrcoef(v, truth)[0, 1]) if np.std(v) > 1e-9 else 0.0,
        edges,
        mae,
        traces,
    )


def _policy_examples(
    model: PVNet, device: torch.device, ps: PositionSet, sims: int, rng: np.random.Generator, n: int = 4
) -> list[PolicyExample]:
    """Hard positions (several moves, only some optimal) where the raw policy errs - and what search makes of them."""
    raw_player = PolicyPlayer(model, device, rng, "raw")
    sc = score_player(raw_player, ps)
    mcts = BatchMCTS(NetEvaluator(model, device), sims, rng)
    res = mcts.search(ps.cur, ps.mask)
    visits = res.visits / np.maximum(res.visits.sum(1, keepdims=True), 1)
    m_choice = visits.argmax(axis=1)
    rows = np.arange(len(ps))
    mcts_ok = ps.best[rows, m_choice]
    pool = np.nonzero(ps.nontrivial & ~sc.correct & mcts_ok & (ps.ply >= 8))[0]
    if len(pool) < n:
        pool = np.concatenate([pool, np.nonzero(ps.nontrivial & ~sc.correct & (ps.ply >= 8))[0]])
    chosen = list(dict.fromkeys(pool.tolist()))[:n]
    logits = _raw_probs(model, ps.cur[chosen], ps.mask[chosen], device)
    return [
        PolicyExample(
            ps.moves[i],
            logits[k].tolist(),
            visits[i].tolist(),
            ps.best[i].tolist(),
            int(sc.action[i]),
            int(m_choice[i]),
            int(ps.value[i]),
        )
        for k, i in enumerate(chosen)
    ]


def _raw_probs(model: PVNet, cur: I64, mask: I64, device: torch.device) -> F32:
    p, _ = NetEvaluator(model, device)(cur, mask)
    q = p * legal_moves(mask)
    return np.asarray(q / np.maximum(q.sum(1, keepdims=True), 1e-12), dtype=np.float32)


# ---------------------------------------------------------------------------------------------
def run_experiment(cfg: Config, device: torch.device) -> Results:
    t_start = time.time()
    ckpt_dir = cfg.out_dir / "checkpoints"
    rng = np.random.default_rng(cfg.seed)
    timings: dict[str, float] = {}
    with Solver(cfg.cache_dir) as solver:
        t = time.time()
        print("== test positions labelled by the perfect solver ==", flush=True)
        sets = make_position_sets(cfg.ev, solver, cfg.seed)
        for k, ps in sets.items():
            print(
                f"  {k}: {len(ps)} positions, {int(ps.nontrivial.sum())} non-trivial, "
                f"values W/D/L = {(ps.value == 1).sum()}/{(ps.value == 0).sum()}/{(ps.value == -1).sum()}",
                flush=True,
            )
        eda = EdaResults(count_positions(cfg.eda_depth), random_play(cfg.eda_random_games, rng), first_moves(solver))
        timings["labelling + EDA"] = time.time() - t

        evaluate = make_eval_fn(sets, device, cfg.seed)
        logs: dict[str, TrainLog] = {}
        print("== AlphaZero-lite ==", flush=True)
        az, logs["az"] = train_az("az", cfg.az, cfg.net, device, cfg.seed, ckpt_dir, evaluate, cfg.ev.curve_every)
        timings["train az"] = logs["az"].total_seconds
        print("== AlphaZero-lite, weaker search during self-play (ablation) ==", flush=True)
        az_lite, logs["az_lite"] = train_az(
            "az_lite", cfg.az_lite, cfg.net, device, cfg.seed + 1, ckpt_dir, evaluate, cfg.ev.curve_every
        )
        timings["train az_lite"] = logs["az_lite"].total_seconds
        print("== DQN ==", flush=True)
        dqn, logs["dqn"] = train_dqn("dqn", cfg.dqn, cfg.net, device, cfg.seed + 2, ckpt_dir, evaluate)
        timings["train dqn"] = logs["dqn"].total_seconds

        # ---- every agent against the solver ----
        t = time.time()
        print("== agents vs. perfect play ==", flush=True)
        init = load_pv(logs["az"].checkpoints["init"], cfg.net, device)
        p25 = load_pv(logs["az"].checkpoints["p25"], cfg.net, device)
        p50 = load_pv(logs["az"].checkpoints["p50"], cfg.net, device)
        entries: list[tuple[Player, str, int]] = [
            (RandomPlayer(rng), "baseline", 0),
            (GreedyPlayer(rng), "baseline", 0),
        ]
        entries += [(MinimaxPlayer(d, rng), "baseline", 0) for d in cfg.minimax_depths]
        entries.append(
            (
                MCTSPlayer(RolloutEvaluator(rng), cfg.mcts_rollout_sims, rng, f"uct-{cfg.mcts_rollout_sims}"),
                "uct",
                cfg.mcts_rollout_sims,
            )
        )
        entries.append((QPlayer(dqn, device, rng, "dqn"), "dqn", 0))
        entries += [
            (PolicyPlayer(init, device, rng, "az-init"), "az", 0),
            (PolicyPlayer(p25, device, rng, "az-25%"), "az", 0),
            (PolicyPlayer(p50, device, rng, "az-50%"), "az", 0),
            (PolicyPlayer(az, device, rng, "az-raw"), "az", 0),
        ]
        search_sims = (10, 25, 100)
        for s in search_sims:
            entries.append((MCTSPlayer(NetEvaluator(az, device), s, rng, f"az+mcts{s}"), "az", s))
        entries.append((PolicyPlayer(az_lite, device, rng, "azlite-raw"), "az-lite", 0))
        entries.append((MCTSPlayer(NetEvaluator(az_lite, device), 100, rng, "azlite+mcts100"), "az-lite", 100))
        entries.append((PerfectPlayer(solver, rng), "perfect", 0))
        agents: list[AgentEval] = []
        for player, family, sims in entries:
            agents.append(_eval_agent(player, family, sims, sets, cfg.ev.bootstrap, rng))
            a = agents[-1].summary["expert"]
            print(f"  {player.name:>16}: expert acc_nt={a['acc_nt']:.3f} blunder_nt={a['blunder_nt']:.3f}", flush=True)
        timings["agents vs solver"] = time.time() - t

        # ---- search budget sweep ----
        t = time.time()
        print("== search-budget sweep ==", flush=True)
        sweep: list[SweepPoint] = []
        n_pos = sum(len(ps) for ps in sets.values())
        for net_name, net in (("final", az), ("init", init)):
            for s in cfg.ev.sims_sweep:
                pl: Player = (
                    PolicyPlayer(net, device, rng, f"{net_name}-{s}")
                    if s == 0
                    else MCTSPlayer(NetEvaluator(net, device), s, rng, f"{net_name}-{s}")
                )
                t1 = time.time()
                summ = {k: summarise(ps, score_player(pl, ps)) for k, ps in sets.items()}
                sweep.append(SweepPoint(net_name, s, summ, (time.time() - t1) / n_pos))
        for s in cfg.ev.sims_sweep:
            if s == 0:
                continue
            pl = MCTSPlayer(RolloutEvaluator(rng), s, rng, f"uct-{s}")
            t1 = time.time()
            summ = {k: summarise(ps, score_player(pl, ps)) for k, ps in sets.items()}
            sweep.append(SweepPoint("uct", s, summ, (time.time() - t1) / n_pos))
        timings["sweep"] = time.time() - t

        # ---- tournament ----
        t = time.time()
        print("== tournament ==", flush=True)
        by_name = {p.name: p for p, _, _ in entries}
        pool_names = [
            "random",
            "greedy",
            *[f"minimax-{d}" for d in cfg.minimax_depths],
            f"uct-{cfg.mcts_rollout_sims}",
            "dqn",
            "az-init",
            "az-25%",
            "az-50%",
            "az-raw",
            "az+mcts25",
            "az+mcts100",
            "azlite-raw",
            "azlite+mcts100",
            "perfect",
        ]
        players = [by_name[n] for n in pool_names]
        tour = run_tournament(players, cfg.ev.games_per_pair, cfg.ev.open_plies, rng)
        elo, lo, hi = elo_with_ci(tour, "random", cfg.ev.bootstrap, rng)
        timings["tournament"] = time.time() - t

        # ---- value head and examples ----
        print("== value head / policy examples ==", flush=True)
        trace_games: list[tuple[str, list[int]]] = []
        i_az = tour.names.index("az+mcts100")
        for (i, j), games in tour.sample_games.items():
            if i_az in (i, j) and len(trace_games) < 3:
                other = tour.names[j if i == i_az else i]
                moves = games[0][0]
                trace_games.append((f"az+mcts100 против {other}", moves))
        value = _value_analysis(az, device, sets, solver, trace_games)
        examples = _policy_examples(az, device, sets["expert"], 100, rng)
        solver.flush()

    shapes = {
        "position": "(cur, mask): 2 x int64 bitboards (bits 7c..7c+5 = column c, row 0 at the bottom)",
        "network input": "(B, 3, 6, 7) float32: own stones, opponent stones, ones",
        "policy head": "(B, 7) logits  ->  softmax over the legal columns",
        "value head": "(B,) in [-1, 1], for the player to move",
        "Q head": "(B, 7) Q-values of the player to move",
        "MCTS tree": f"(G, sims+1, 7) arrays N, W, P and child index for G={cfg.az.games_per_iter} parallel games",
        "replay sample": "bitboards (B,), (B,)  +  pi (B, 7) float32  +  z (B,) float32",
    }
    n_params = {"az": n_parameters(az), "dqn": n_parameters(dqn)}
    for k, v in n_params.items():
        logs[k].n_params = v
    return Results(
        cfg.profile,
        cfg.seed,
        str(device),
        sets,
        eda,
        logs,
        agents,
        sweep,
        tour,
        elo,
        lo,
        hi,
        value,
        examples,
        time.time() - t_start,
        timings,
        shapes,
    )


__all__ = ["WIDTH", "AgentEval", "Results", "load_q", "mirror_bits", "run_experiment"]
