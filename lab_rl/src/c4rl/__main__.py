"""CLI.

python -m c4rl eda              # analysis of the game only (state space, random play, solver values) -> eda.html
python -m c4rl run              # train AlphaZero-lite and DQN, evaluate against the solver, tournament, report.html
python -m c4rl report           # rebuild report.html from the saved results of a previous run (no training)
python -m c4rl serve            # web interface: play against the trained agents in the browser
python -m c4rl play             # the same in the terminal
python -m c4rl export-weights   # copy the finished run's checkpoints into the versioned weights/ folder
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from .config import make_config
from .eda import EdaResults, count_positions, first_moves, random_play
from .evaluation import make_position_sets
from .experiment import Results, run_experiment
from .models import discover, export_weights
from .report import build_eda, build_report
from .serve import Engine, play_terminal, serve
from .solver import Solver


def _engine(args: argparse.Namespace) -> Engine:
    registry = discover(args.weights, args.out / "checkpoints")
    if not registry:
        print("no trained networks found (weights/ and outputs/checkpoints are empty): only classic bots are available")
    return Engine(registry, torch.device(args.device), args.cache)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="c4rl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, text in (
        ("eda", "analysis of the game (no training) -> eda.html"),
        ("run", "train, evaluate against the solver, build the report"),
        ("report", "rebuild the report from the saved results of a previous run (no training)"),
        ("serve", "web interface to play against the trained agents"),
        ("play", "play against an agent in the terminal"),
        ("export-weights", "copy the finished run's checkpoints into weights/"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--profile", choices=["smoke", "standard", "full"], default="standard")
        p.add_argument("--out", type=Path, default=Path("outputs"))
        p.add_argument("--cache", type=Path, default=Path("cache"))
        p.add_argument("--weights", type=Path, default=Path("weights"))
        if name in ("serve", "play"):
            p.add_argument("--device", default="cpu", help="cpu is plenty for one game at a time")
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
        if name == "play":
            p.add_argument(
                "--opponent", default="az_final", help="a model id (see the web list), minimax-4, perfect, ..."
            )
            p.add_argument("--sims", type=int, default=100, help="MCTS simulations per move for AlphaZero models")
            p.add_argument("--side", choices=["X", "O"], default="X", help="X moves first")
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        serve(_engine(args), args.host, args.port)
        return
    if args.cmd == "play":
        engine = _engine(args)
        try:
            play_terminal(engine, args.opponent, args.sims, args.side == "X")
        finally:
            engine.close()
        return
    if args.cmd == "export-weights":
        res = pickle.loads((args.out / "results.pkl").read_bytes())
        assert isinstance(res, Results)
        elo = {n: float(e) for n, e in zip(res.tournament.names, res.elo, strict=True)}
        for path in export_weights(args.out / "checkpoints", args.weights, elo):
            print(f"copied {path}")
        return

    cfg = make_config(args.profile, args.out, args.cache)
    if args.cmd == "eda":
        rng = np.random.default_rng(cfg.seed)
        with Solver(cfg.cache_dir) as solver:
            sets = make_position_sets(cfg.ev, solver, cfg.seed)
            eda = EdaResults(
                count_positions(cfg.eda_depth), random_play(cfg.eda_random_games, rng), first_moves(solver)
            )
        print(f"eda: {build_eda(cfg, eda, sets)}", flush=True)
        return

    results_path = cfg.out_dir / "results.pkl"
    if args.cmd == "run":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"profile={args.profile} device={device}", flush=True)
        res = run_experiment(cfg, device)
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        results_path.write_bytes(pickle.dumps(res))  # only our own output is ever loaded back
    else:
        res = pickle.loads(results_path.read_bytes())
        assert isinstance(res, Results)
    print(f"report: {build_report(cfg, res)}", flush=True)


if __name__ == "__main__":
    main()
