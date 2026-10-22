"""CLI: ``python -m fusionlab run --profile standard`` and ``python -m fusionlab report``."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from .config import make_config
from .data import download
from .experiment import Results, run_experiment
from .features import build_features
from .report import build_report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="fusionlab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, text in (
        ("run", "download data, extract features, train, build the report"),
        ("report", "rebuild the report from the saved results of a previous run (no training)"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--profile", choices=["smoke", "standard", "tuned", "full"], default="standard")
        p.add_argument("--out", type=Path, default=Path("outputs"))
        p.add_argument("--cache", type=Path, default=Path("cache"))
    args = ap.parse_args(argv)

    cfg = make_config(args.profile, args.out, args.cache)
    files = download(cfg)
    feats = build_features(cfg, files)
    results_path = cfg.out_dir / "results.pkl"
    if args.cmd == "run":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"profile={args.profile} device={device}", flush=True)
        res = run_experiment(cfg, feats, device)
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        results_path.write_bytes(pickle.dumps(res))  # only our own output is ever loaded back
    else:
        res = pickle.loads(results_path.read_bytes())
        assert isinstance(res, Results)
    path = build_report(cfg, feats, res)
    print(f"report: {path}", flush=True)


if __name__ == "__main__":
    main()
