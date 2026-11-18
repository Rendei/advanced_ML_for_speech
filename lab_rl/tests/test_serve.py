"""The play interface: model registry, bot replies, the HTTP API and the solver verdicts."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import torch

from c4rl.config import NetConfig
from c4rl.models import discover, export_weights, infer_architecture
from c4rl.net import PVNet, QNet, save_checkpoint
from c4rl.serve import Busy, Engine, GameError, Limits, make_server, replay
from c4rl.solver import Solver, solver_available


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


# ---------------------------------------------------------------------------------------------
# internet-facing mode: limits and no solver
# ---------------------------------------------------------------------------------------------
def _serve(engine: Engine) -> tuple[Any, str]:
    server = make_server(engine, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _post(base: str, body: Any, raw: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, Any]:
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(base + "/api/move", data, {"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _public_engine(ckpt_dir: Path, tmp_path: Path, **kw: Any) -> Engine:
    return Engine(
        discover(tmp_path / "none", ckpt_dir), torch.device("cpu"), tmp_path / "cache", Limits(public=True, **kw)
    )


def test_public_mode_hides_the_solver(ckpt_dir: Path, tmp_path: Path) -> None:
    eng = _public_engine(ckpt_dir, tmp_path)
    assert "perfect" not in {o["id"] for o in eng.opponents()}
    assert eng.info() == {"public": True, "max_sims": 400, "solver": False, "solver_slots": 0}
    with pytest.raises(GameError):
        eng.respond({"moves": [3], "opponent": "perfect"})
    with pytest.raises(GameError):
        eng.respond({"moves": [3], "opponent": "random", "judge": True})


@needs_solver
def test_public_solver_is_limited_to_the_configured_slots(ckpt_dir: Path, tmp_path: Path) -> None:
    cache = Path("cache") if Path("cache").is_dir() else tmp_path / "cache"
    eng = Engine(discover(tmp_path / "none", ckpt_dir), torch.device("cpu"), cache, Limits(public=True, solver_slots=2))
    try:
        assert eng.info()["solver"] is True and eng.info()["solver_slots"] == 2
        assert "perfect" in {o["id"] for o in eng.opponents()}
        with pytest.raises(Busy):  # not started yet: nobody waits for the book download
            eng.respond({"moves": [3, 3], "opponent": "perfect"})
        eng.warm_up()
        deadline = time.monotonic() + 900
        while eng._solver is None and time.monotonic() < deadline:
            time.sleep(0.2)
        solver = eng._solver
        assert solver is not None
        solver._cache.clear()  # cache hits do not need a slot; force real queries
        held = [solver._acquire(), solver._acquire()]  # both slots taken by "other players"
        try:
            out = eng.respond({"moves": [3, 3], "opponent": "random", "judge": True})
            assert "busy" in out["judge_error"] and out["move"] is not None  # the game itself keeps going
            with pytest.raises(Busy):
                eng.respond({"moves": [3, 3], "opponent": "perfect"})
        finally:
            for proc in held:
                solver._release(proc)
        assert eng.respond({"moves": [3], "opponent": "random", "judge": True})["judge"]["optimal"] is True
        assert eng.respond({"moves": [3, 3], "opponent": "perfect"})["move"] is not None
    finally:
        eng.close()


@needs_solver
def test_solver_pool_answers_agree_under_concurrency(tmp_path: Path) -> None:
    cache = Path("cache") if Path("cache").is_dir() else tmp_path
    histories = ["", "4", "44", "45", "443", "4453", "3", "34"]
    with Solver(cache, workers=1) as one:
        expected = {h: one.analyze(h) for h in histories}
    with Solver(cache, workers=3, wait=60.0) as pool:
        pool._cache.clear()
        got: dict[str, list[int]] = {}
        threads = [threading.Thread(target=lambda h=h: got.__setitem__(h, pool.analyze(h))) for h in histories]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert got == expected


def test_sims_are_capped(engine: Engine) -> None:
    for bad in (-1, 401, 10**9):
        with pytest.raises(GameError):
            engine.respond({"moves": [3], "opponent": "az_final", "sims": bad})
    assert engine.respond({"moves": [3], "opponent": "az_final", "sims": 2})["move"] is not None


def test_http_rejects_oversized_malformed_and_unknown(ckpt_dir: Path, tmp_path: Path) -> None:
    server, base = _serve(_public_engine(ckpt_dir, tmp_path))
    try:
        assert _post(base, None, raw=b"x" * 5000)[0] == 413
        assert _post(base, None, raw=b"not json")[0] == 400
        assert _post(base, None, raw=b'["a list"]')[0] == 400
        assert _post(base, {"moves": "3", "opponent": "random"})[0] == 400
        assert _post(base, {"moves": [3], "opponent": "random", "sims": 10**9})[0] == 400
        assert urllib.request.urlopen(base + "/healthz").read() == b"ok"
        headers = urllib.request.urlopen(base + "/").headers
        assert headers["X-Content-Type-Options"] == "nosniff" and "default-src" in headers["Content-Security-Policy"]
    finally:
        server.shutdown()
        server.server_close()


def test_rate_limit_and_busy(ckpt_dir: Path, tmp_path: Path) -> None:
    server, base = _serve(_public_engine(ckpt_dir, tmp_path, rate_per_min=3))
    try:
        codes = [_post(base, {"moves": [], "opponent": "random"})[0] for _ in range(5)]
        assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    finally:
        server.shutdown()
        server.server_close()
    busy = _public_engine(ckpt_dir, tmp_path, max_inflight=0)
    server, base = _serve(busy)
    try:
        code, body = _post(base, {"moves": [], "opponent": "random"})
        assert code == 503 and "busy" in body["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_rate_limit_uses_forwarded_address_only_behind_a_trusted_proxy(ckpt_dir: Path, tmp_path: Path) -> None:
    eng = _public_engine(ckpt_dir, tmp_path, rate_per_min=2, trust_proxy=True)
    server, base = _serve(eng)
    try:
        a = {"X-Forwarded-For": "203.0.113.1"}
        b = {"X-Forwarded-For": "203.0.113.2"}
        body = {"moves": [], "opponent": "random"}
        assert [_post(base, body, headers=a)[0] for _ in range(3)] == [200, 200, 429]
        assert _post(base, body, headers=b)[0] == 200  # another client is not affected
    finally:
        server.shutdown()
        server.server_close()
