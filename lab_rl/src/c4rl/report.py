"""Runtime reports: ``eda.html`` (the game only) and ``report.html`` (EDA + experiment) from this run."""

from __future__ import annotations

import base64
import io
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from . import viz
from .config import Config
from .eda import EdaResults
from .evaluation import PositionSet
from .experiment import AgentEval, Results

TEMPLATES = Path(__file__).parent


def _png(fig: Figure, name: str, out: Path | None) -> str:
    """Base64 PNG for embedding into the HTML; also saved to ``out/<name>.png`` when ``out`` is given."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.png").write_bytes(buf.getvalue())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _pct(x: float, d: int = 1) -> str:
    return "—" if np.isnan(x) else f"{100 * x:.{d}f}"


def _environment(cfg: Config) -> dict[str, Any]:
    return {
        "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cfg": cfg,
        "env": f"Python {platform.python_version()}, torch {torch.__version__}, "
        f"{'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
    }


def _render(template: str, ctx: dict[str, Any], out: Path) -> Path:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES), autoescape=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(env.get_template(template).render(**ctx), encoding="utf-8")
    return out


# ---- EDA ----------
def eda_context(eda: EdaResults, sets: dict[str, PositionSet], fig_dir: Path | None) -> dict[str, Any]:
    c = eda.counts
    figs = {
        "space": _png(viz.eda_state_space(c), "eda_state_space", fig_dir),
        "random": _png(viz.eda_random_play(eda.random), "eda_random_play", fig_dir),
        "first": _png(viz.eda_first_moves(eda.first, eda.random), "eda_first_moves", fig_dir),
        "sets": _png(viz.eda_sets(sets), "eda_test_sets", fig_dir),
    }
    count_rows = [
        {
            "ply": k,
            "total": f"{t:,}".replace(",", " "),
            "term": f"{w:,}".replace(",", " "),
            "mirror": f"{m:,}".replace(",", " "),
        }
        for k, (t, w, m) in enumerate(zip(c.total, c.terminal, c.up_to_mirror, strict=True))
    ]
    rp = eda.random
    set_rows = []
    for name, ps in sets.items():
        n = len(ps)
        set_rows.append(
            {
                "name": name,
                "n": n,
                "nontrivial": f"{int(ps.nontrivial.sum())} ({_pct(ps.nontrivial.mean(), 0)}%)",
                "wdl": " / ".join(f"{_pct(float((ps.value == v).mean()), 0)}" for v in (1, 0, -1)),
                "mean_best": f"{ps.n_best.mean():.2f}",
                "mean_ply": f"{ps.ply.mean():.1f}",
            }
        )
    return {
        "edafigs": figs,
        "count_rows": count_rows,
        "depth": len(c.total) - 1,
        "random_len": f"{rp.length.mean():.1f}",
        "random_x": _pct(float((rp.result == 1).mean())),
        "random_draw": _pct(float((rp.result == 0).mean())),
        "random_o": _pct(float((rp.result == -1).mean())),
        "n_random_games": len(rp.length),
        "first_win": ", ".join(str(k + 1) for k, o in enumerate(eda.first.outcome) if o == 1) or "—",
        "first_draw": ", ".join(str(k + 1) for k, o in enumerate(eda.first.outcome) if o == 0),
        "first_loss": ", ".join(str(k + 1) for k, o in enumerate(eda.first.outcome) if o == -1),
        "set_rows": set_rows,
        "branching": f"{c.total[-1] / c.total[-2]:.2f}",
    }


def build_eda(cfg: Config, eda: EdaResults, sets: dict[str, PositionSet]) -> Path:
    ctx = {**_environment(cfg), **eda_context(eda, sets, cfg.out_dir / "figures")}
    return _render("eda.html.j2", ctx, cfg.out_dir / "eda.html")


# ---- full report ----------
def _agent_row(a: AgentEval) -> dict[str, Any]:
    cells = {}
    for k in ("random", "expert"):
        s, ci = a.summary[k], a.ci[k]
        cells[k] = {
            "acc": _pct(s["acc"]),
            "acc_nt": _pct(s["acc_nt"]),
            "ci": f"[{_pct(ci['acc_nt'][0])}; {_pct(ci['acc_nt'][1])}]",
            "blunder_nt": _pct(s["blunder_nt"]),
            "bci": f"[{_pct(ci['blunder_nt'][0])}; {_pct(ci['blunder_nt'][1])}]",
        }
    return {
        "name": a.name,
        "family": viz.FAMILY_RU[a.family],
        "sims": a.sims,
        **{f"c_{k}": v for k, v in cells.items()},
    }


def _acc(res: Results, name: str, ev: str = "expert", metric: str = "acc_nt") -> float:
    return next(a for a in res.agents if a.name == name).summary[ev][metric]


def _ci(res: Results, name: str, ev: str = "expert", metric: str = "acc_nt") -> tuple[float, float]:
    return next(a for a in res.agents if a.name == name).ci[ev][metric]


def _elo(res: Results, name: str) -> tuple[float, float, float]:
    i = res.tournament.names.index(name)
    return float(res.elo[i]), float(res.elo_lo[i]), float(res.elo_hi[i])


def build_report(cfg: Config, res: Results) -> Path:
    fig_dir = cfg.out_dir / "figures"
    ctx: dict[str, Any] = {**_environment(cfg), **eda_context(res.eda, res.sets, fig_dir)}
    families = {a.name: a.family for a in res.agents}
    names = [a.name for a in res.agents]
    fams = [a.family for a in res.agents]
    summary = {a.name: a.summary for a in res.agents}
    ci = {a.name: a.ci for a in res.agents}

    figs = ctx["edafigs"]
    figs["pipeline"] = _png(viz.pipeline_diagram(), "pipeline", fig_dir)
    figs["curves"] = _png(viz.learning_curves(res.logs), "learning_curves", fig_dir)
    figs["diag"] = _png(viz.training_diagnostics(res.logs), "training_diagnostics", fig_dir)
    figs["acc"] = _png(
        viz.agents_bars(names, fams, summary, ci, "acc_nt", "точность на «нетривиальных» позициях"),
        "agents_accuracy",
        fig_dir,
    )
    figs["blunder"] = _png(
        viz.agents_bars(names, fams, summary, ci, "blunder_nt", "доля грубых ошибок (результат ухудшен)"),
        "agents_blunders",
        fig_dir,
    )
    sweep_pts = [(p.net, p.sims, p.summary["expert"]["acc_nt"], p.seconds_per_position) for p in res.sweep]
    figs["sweep"] = _png(viz.sims_sweep(sweep_pts, {}), "sims_sweep", fig_dir)

    t = res.tournament
    t_fams = [families.get(n, "baseline") for n in t.names]
    figs["elo"] = _png(viz.elo_forest(t.names, t_fams, res.elo, res.elo_lo, res.elo_hi), "elo", fig_dir)
    figs["matrix"] = _png(viz.tournament_matrix(t, res.elo), "tournament_matrix", fig_dir)
    figs["elo_progress"] = _png(
        viz.elo_progress(t.names, res.elo, res.elo_lo, res.elo_hi, res.logs["az"].total_games), "elo_progress", fig_dir
    )
    v = res.value
    figs["value"] = _png(viz.value_head(v.class_mean, v.class_std, v.ply_edges, v.mae_by_ply), "value_head", fig_dir)
    figs["traces"] = _png(
        viz.value_traces([(tr.title, tr.plies, tr.net_value, tr.true_value) for tr in v.traces]),
        "value_traces",
        fig_dir,
    )
    figs["examples"] = (
        _png(
            viz.policy_examples(
                [(e.moves, e.raw, e.visits, e.best, e.raw_choice, e.mcts_choice) for e in res.examples]
            ),
            "policy_examples",
            fig_dir,
        )
        if res.examples
        else ""
    )

    # ---- tables ----
    agent_rows = [_agent_row(a) for a in res.agents]
    order = np.argsort(-res.elo)
    games_n = t.points()[1]
    elo_rows = [
        {
            "name": t.names[i],
            "family": viz.FAMILY_RU[t_fams[i]],
            "elo": f"{res.elo[i]:.0f}",
            "ci": f"[{res.elo_lo[i]:.0f}; {res.elo_hi[i]:.0f}]",
            "games": int(games_n[i].sum()),
        }
        for i in order
    ]
    run_rows = [
        {
            "name": viz.RUN_LABEL[k],
            "params": f"{log.n_params / 1e3:.0f}k",
            "games": f"{log.total_games:,}".replace(",", " "),
            "positions": f"{log.total_positions:,}".replace(",", " "),
            "minutes": f"{log.total_seconds / 60:.1f}",
            "sims": {"az": cfg.az.sims, "az_lite": cfg.az_lite.sims, "dqn": 0}[k],
        }
        for k, log in res.logs.items()
    ]
    sweep_rows = [
        {
            "net": {"final": "обученная сеть", "init": "необученная сеть", "uct": "UCT"}[p.net],
            "sims": p.sims,
            "expert": _pct(p.summary["expert"]["acc_nt"]),
            "random": _pct(p.summary["random"]["acc_nt"]),
            "ms": f"{1000 * p.seconds_per_position:.2f}",
        }
        for p in res.sweep
    ]

    # ---- the conclusions are computed, not typed in ----
    az_raw, az_m = _acc(res, "az-raw"), _acc(res, "az+mcts100")
    dqn_acc = _acc(res, "dqn")
    lite_raw = _acc(res, "azlite-raw")
    best_minimax = max((n for n in names if n.startswith("minimax")), key=lambda n: _elo(res, n)[0])
    e_az_m, e_az_m_lo, e_az_m_hi = _elo(res, "az+mcts100")
    e_az_raw = _elo(res, "az-raw")
    e_dqn = _elo(res, "dqn")
    e_mm = _elo(res, best_minimax)
    e_lite = _elo(res, "azlite-raw")
    e_perfect = _elo(res, "perfect")
    ci_raw, ci_m = _ci(res, "az-raw"), _ci(res, "az+mcts100")
    facts = {
        "az_raw": _pct(az_raw),
        "az_mcts": _pct(az_m),
        "dqn_acc": _pct(dqn_acc),
        "lite_raw": _pct(lite_raw),
        "search_gain": f"{100 * (az_m - az_raw):+.1f}",
        "search_significant": ci_m[0] > ci_raw[1],
        "az_vs_dqn": f"{100 * (az_raw - dqn_acc):+.1f}",
        "az_beats_dqn_elo": e_az_raw[1] > e_dqn[2],
        "dqn_beats_az_elo": e_dqn[1] > e_az_raw[2],
        "e_az_m": f"{e_az_m:.0f}",
        "e_az_raw": f"{e_az_raw[0]:.0f}",
        "e_dqn": f"{e_dqn[0]:.0f}",
        "e_mm": f"{e_mm[0]:.0f}",
        "mm_name": best_minimax,
        "az_m_beats_mm": e_az_m_lo > e_mm[2],
        "mm_beats_az_m": e_mm[1] > e_az_m_hi,
        "e_lite": f"{e_lite[0]:.0f}",
        "lite_vs_az": f"{e_lite[0] - e_az_raw[0]:+.0f}",
        "lite_sig_worse": e_lite[2] < e_az_raw[1],
        "lite_sig_better": e_lite[1] > e_az_raw[2],
        "perfect_acc": _pct(_acc(res, "perfect")),
        "e_perfect": f"{e_perfect[0]:.0f}",
        "perfect_gap": f"{e_perfect[0] - e_az_m:.0f}",
        "perfect_overlap": e_az_m_hi >= e_perfect[1],
        "value_sign": _pct(res.value.sign_accuracy),
        "value_corr": f"{res.value.corr:.2f}",
        "init_acc": _pct(_acc(res, "az-init")),
        "az_games": f"{res.logs['az'].total_games:,}".replace(",", " "),
        "dqn_games": f"{res.logs['dqn'].total_games:,}".replace(",", " "),
    }
    init_s = [p for p in res.sweep if p.net == "init" and p.sims > 0]
    fin_s = [p for p in res.sweep if p.net == "final" and p.sims > 0]
    facts["init_search_best"] = _pct(max((p.summary["expert"]["acc_nt"] for p in init_s), default=float("nan")))
    facts["final_search_best"] = _pct(max((p.summary["expert"]["acc_nt"] for p in fin_s), default=float("nan")))

    # game replays for the HTML viewer
    replays: list[dict[str, Any]] = []
    i_az = t.names.index("az+mcts100")
    for (i, j), games in sorted(t.sample_games.items()):
        if i_az not in (i, j):
            continue
        for k, (moves, outcome) in enumerate(games[:6]):
            x_is_i = k < len(games) // 2
            x_name = t.names[i] if x_is_i else t.names[j]
            o_name = t.names[j] if x_is_i else t.names[i]
            x_res = outcome if x_is_i else -outcome  # outcome is for player i
            label = "X побеждает" if x_res == 1 else "O побеждает" if x_res == -1 else "ничья"
            replays.append({"title": f"X: {x_name} — O: {o_name}  ({label})", "moves": moves})
    replays = replays[:24]

    ctx.update(
        {
            "figs": figs,
            "agent_rows": agent_rows,
            "elo_rows": elo_rows,
            "run_rows": run_rows,
            "sweep_rows": sweep_rows,
            "f": facts,
            "replays": replays,
            "timings": {k: f"{v / 60:.1f}" for k, v in res.timings.items()},
            "minutes": res.seconds / 60,
            "shapes": res.shapes,
            "seed": res.seed,
            "az": cfg.az,
            "az_lite": cfg.az_lite,
            "dqn": cfg.dqn,
            "net": cfg.net,
            "ev": cfg.ev,
            "n_sets": {k: len(v) for k, v in res.sets.items()},
            "n_games_total": int(sum(s.size for s in t.pair_scores.values())),
            "n_players": len(t.names),
            "minimax_depths": ", ".join(map(str, cfg.minimax_depths)),
            "uct_sims": cfg.mcts_rollout_sims,
            "device": res.device,
        }
    )
    path = _render("report.html.j2", ctx, cfg.out_dir / "report.html")

    metrics: dict[str, Any] = {
        "profile": res.profile,
        "agents": {a.name: {"family": a.family, "sims": a.sims, "summary": a.summary, "ci": a.ci} for a in res.agents},
        "elo": {
            t.names[i]: [float(res.elo[i]), float(res.elo_lo[i]), float(res.elo_hi[i])] for i in range(len(t.names))
        },
        "training": {
            k: {"games": log.total_games, "seconds": log.total_seconds, "final": log.metrics[-1] if log.metrics else {}}
            for k, log in res.logs.items()
        },
        "facts": facts,
    }
    (cfg.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
