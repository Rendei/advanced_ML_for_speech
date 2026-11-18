"""The play interface: model registry, bot replies, the HTTP API and the solver verdicts."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import torch

from c4rl.config import NetConfig
from c4rl.models import discover, export_weights, infer_architecture
from c4rl.net import PVNet, QNet, save_checkpoint
from c4rl.serve import Engine, GameError, make_server, replay
from c4rl.solver import solver_available


@pytest.fixture
def ckpt_dir(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    d = tmp_path / "ckpt"
    save_checkpoint(PVNet(NetConfig(8, 2)), d / "az_final.pt")
    save_checkpoint(QNet(NetConfig(16, 1)), d / "dqn_final.pt")
    return d


@pytest.fixture
def engine(ckpt_dir: Path, tmp_path: Path) -> Engine:
    return Engine(discover(tmp_path / "no_weights", ckpt_dir), torch.device("cpu"), tmp_path / "cache")


def test_architecture_is_read_off_the_weights(ckpt_dir: Path) -> None:
    assert infer_architecture(ckpt_dir / "az_final.pt") == ("pv", NetConfig(8, 2))
    assert infer_architecture(ckpt_dir / "dqn_final.pt") == ("q", NetConfig(16, 1))


def test_export_writes_manifest_with_elo_and_discover_reads_it(ckpt_dir: Path, tmp_path: Path) -> None:
    w = tmp_path / "weights"
    export_weights(ckpt_dir, w, {"az-raw": 788.4, "az+mcts100": 943.7, "dqn": 748.0})
    reg = discover(w)
    assert set(reg) == {"az_final", "dqn_final"}
    assert reg["az_final"].elo == 788 and reg["az_final"].elo_search == 944
    assert reg["dqn_final"].kind == "q" and reg["dqn_final"].elo == 748


def test_replay_status_and_illegal_moves() -> None:
    assert replay([3, 3, 4])[2] == "playing"
    assert replay([0, 1, 0, 1, 0, 1, 0])[2] == "x"  # vertical four for X
    with pytest.raises(GameError):
        replay([0] * 7)  # column overflow
    with pytest.raises(GameError):
        replay([0, 1, 0, 1, 0, 1, 0, 2])  # moving after the game is over


@pytest.mark.parametrize("opponent", ["az_final", "dqn_final", "random", "greedy", "minimax-2"])
def test_bot_replies_with_a_legal_move(engine: Engine, opponent: str) -> None:
    out = engine.respond({"moves": [3, 3, 2], "opponent": opponent, "sims": 8, "hints": True})
    assert 0 <= out["move"] <= 6 and out["status"] in ("playing", "x", "o")
    replay([3, 3, 2, out["move"]])


def test_hints_contain_policy_visits_and_value(engine: Engine) -> None:
    out = engine.respond({"moves": [3], "opponent": "az_final", "sims": 16, "hints": True})
    h = out["hints"]
    assert len(h["policy"]) == 7 and len(h["visits"]) == 7 and -1 <= h["value"] <= 1
    assert abs(sum(h["visits"]) - 1) < 1e-4
    quiet = engine.respond({"moves": [3], "opponent": "az_final", "sims": 0, "hints": False})
    assert quiet["hints"] == {}


def test_bot_takes_an_immediate_win(engine: Engine) -> None:
    # X: 1,2,3 on the bottom row, O: 7,7,5 -> X to move wins with column 4 (index 3); greedy always sees it
    out = engine.respond({"moves": [0, 6, 1, 6, 2, 4], "opponent": "greedy", "sims": 0})
    assert out["move"] == 3 and out["status"] == "x"


def test_finished_game_gets_no_bot_reply(engine: Engine) -> None:
    out = engine.respond({"moves": [0, 1, 0, 1, 0, 1, 0], "opponent": "random"})
    assert out["move"] is None and out["status"] == "x"


def test_unknown_opponent_and_bad_requests(engine: Engine) -> None:
    with pytest.raises(GameError):
        engine.respond({"moves": [3], "opponent": "nobody"})
    with pytest.raises(GameError):
        engine.respond({"moves": [9], "opponent": "random"})


def test_http_api_end_to_end(engine: Engine) -> None:
    server = make_server(engine, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "Connect-4" in page and "/api/move" in page
        opp = json.loads(urllib.request.urlopen(base + "/api/opponents").read())
        ids = {o["id"] for o in opp}
        assert {"az_final", "dqn_final", "random", "perfect"} <= ids
        assert next(o for o in opp if o["id"] == "az_final")["searchable"] is True

        def post(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            req = urllib.request.Request(
                base + "/api/move", json.dumps(body).encode(), {"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        code, out = post({"moves": [], "opponent": "az_final", "sims": 4})
        assert code == 200 and 0 <= out["move"] <= 6
        code, err = post({"moves": [0] * 7, "opponent": "random"})
        assert code == 400 and "illegal" in err["error"]
    finally:
        server.shutdown()
        server.server_close()


needs_solver = pytest.mark.skipif(not solver_available(), reason="c4solver binary not installed")


@needs_solver
def test_judge_flags_a_blunder_and_accepts_the_best_move(tmp_path: Path) -> None:
    cache = Path("cache") if Path("cache").is_dir() else tmp_path
    eng = Engine({}, torch.device("cpu"), cache)
    try:
        good = eng.respond({"moves": [3], "opponent": "random", "judge": True})
        assert good["judge"]["optimal"] is True  # the centre column is the winning first move
        bad = eng.respond({"moves": [0], "opponent": "random", "judge": True})
        assert bad["judge"]["optimal"] is False and 3 in bad["judge"]["best_columns"]
        perfect = eng.respond({"moves": [3, 3], "opponent": "perfect"})
        assert 0 <= perfect["move"] <= 6
    finally:
        eng.close()
