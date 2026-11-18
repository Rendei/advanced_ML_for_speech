"""Play against the trained agents: a small stdlib web server (``python -m c4rl serve``) and a terminal mode.

The server is stateless: the browser sends the list of moves played so far, the server replays it, asks the chosen
opponent for a move and answers with that move plus optional "thoughts" (policy, MCTS visit shares, value) and an
optional solver verdict on the human's last move.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .game import WIDTH, legal_cols, legal_moves, play_int, to_moves
from .mcts import BatchMCTS
from .models import ModelInfo, load_model
from .net import NetEvaluator, PVNet, QNet, to_tensor
from .players import GreedyPlayer, MinimaxPlayer, PerfectPlayer, Player, RandomPlayer
from .solver import Solver, solver_available

CLASSIC = {
    "random": "случайный игрок",
    "greedy": "greedy: выигрывает/блокирует за один ход",
    "minimax-2": "minimax, глубина 2",
    "minimax-4": "minimax, глубина 4",
    "perfect": "идеальный игрок (солвер)",
}
OUTCOME_RU = {1: "выигрыш", 0: "ничья", -1: "проигрыш"}


class GameError(ValueError):
    """The request cannot be honoured (illegal move, finished game, unknown opponent, ...)."""


def replay(moves: list[int]) -> tuple[int, int, str]:
    """Position after ``moves`` and the status: ``playing`` / ``x`` / ``o`` / ``draw``."""
    cur = mask = 0
    status = "playing"
    for i, c in enumerate(moves):
        if status != "playing" or c not in legal_cols(mask):
            raise GameError(f"illegal move {c + 1} at ply {i + 1}")
        cur, mask, won = play_int(cur, mask, c)
        if won:
            status = "x" if i % 2 == 0 else "o"
        elif not legal_cols(mask):
            status = "draw"
    return cur, mask, status


class Engine:
    """Everything the web page and the terminal need: opponents, bot moves, solver verdicts."""

    def __init__(self, registry: dict[str, ModelInfo], device: torch.device, cache_dir: Path) -> None:
        self.registry = registry
        self.device = device
        self.cache_dir = cache_dir
        self.rng = np.random.default_rng()
        self._models: dict[str, PVNet | QNet] = {}
        self._solver: Solver | None = None
        self._lock = threading.Lock()  # one search at a time: models and the solver pipe are shared

    # ------------------------------------------------------------------ catalogue
    def opponents(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for info in self.registry.values():
            out.append(
                {
                    "id": info.id,
                    "label": info.label,
                    "group": info.group,
                    "searchable": info.kind == "pv",
                    "elo": info.elo,
                    "elo_search": info.elo_search,
                }
            )
        for pid, label in CLASSIC.items():
            if pid == "perfect" and not solver_available() and self._solver is None:
                label += " — нужен c4solver"
            out.append({"id": pid, "label": label, "group": "Классические боты", "searchable": False})
        return out

    # ------------------------------------------------------------------ bot
    def _model(self, model_id: str) -> PVNet | QNet:
        if model_id not in self._models:
            self._models[model_id] = load_model(self.registry[model_id], self.device)
        return self._models[model_id]

    def _get_solver(self) -> Solver:
        if self._solver is None:
            self._solver = Solver(self.cache_dir)
        return self._solver

    def bot_move(self, moves: list[int], opponent: str, sims: int, hints: bool) -> dict[str, Any]:
        cur, mask, status = replay(moves)
        if status != "playing":
            raise GameError("the game is already over")
        c = np.array([cur], dtype=np.int64)
        m = np.array([mask], dtype=np.int64)
        legal = legal_moves(m)
        info: dict[str, Any] = {}
        with self._lock:
            if opponent in self.registry:
                net = self._model(opponent)
                if isinstance(net, PVNet):
                    ev = NetEvaluator(net, self.device)
                    probs, value = ev(c, m)
                    raw = probs * legal
                    raw = raw / max(float(raw.sum()), 1e-12)
                    scores = raw[0]
                    info = {"policy": raw[0].tolist(), "value": float(value[0])}
                    if sims > 0:
                        res = BatchMCTS(ev, sims, self.rng).search(c, m)
                        vis = res.visits[0] / max(float(res.visits[0].sum()), 1.0)
                        scores = vis
                        info.update({"visits": vis.tolist(), "value": float(res.value[0])})
                else:
                    net.eval()
                    with torch.inference_mode():
                        q = net(to_tensor(c, m, self.device)).cpu().numpy()[0]
                    scores = np.where(legal[0], q, -np.inf)
                    info = {"q": [None if not legal[0][k] else float(q[k]) for k in range(WIDTH)]}
            elif opponent in CLASSIC:
                player = self._classic(opponent)
                scores = np.zeros(WIDTH)
                scores[int(player.act(c, m, [to_moves(moves)])[0])] = 1.0
            else:
                raise GameError(f"unknown opponent {opponent!r}")
        scores = np.where(legal[0], scores, -np.inf) + self.rng.random(WIDTH) * 1e-9
        move = int(scores.argmax())
        _, _, after = replay([*moves, move])
        return {"move": move, "status": after, "hints": info if hints else {}}

    def _classic(self, pid: str) -> Player:
        if pid == "random":
            return RandomPlayer(self.rng)
        if pid == "greedy":
            return GreedyPlayer(self.rng)
        if pid.startswith("minimax-"):
            return MinimaxPlayer(int(pid.split("-")[1]), self.rng)
        return PerfectPlayer(self._get_solver(), self.rng)

    # ------------------------------------------------------------------ judge
    def judge(self, before: list[int], chosen: int) -> dict[str, Any]:
        """Solver verdict on ``chosen`` played in the position after ``before``."""
        replay(before)
        with self._lock:
            outcomes = self._get_solver().outcomes(to_moves(before))
        best = int(outcomes.max())
        got = int(outcomes[chosen])
        return {
            "optimal": got == best,
            "best_columns": [int(k) for k in np.nonzero(outcomes == best)[0]],
            "position": OUTCOME_RU[best],
            "after_move": OUTCOME_RU.get(got, "?"),
        }

    def respond(self, req: dict[str, Any]) -> dict[str, Any]:
        moves = [int(x) for x in req.get("moves", [])]
        out: dict[str, Any] = {}
        _, _, status = replay(moves)
        if req.get("judge") and moves:
            try:
                out["judge"] = self.judge(moves[:-1], moves[-1])
            except (OSError, RuntimeError) as exc:  # solver binary missing / book download failed
                out["judge_error"] = str(exc)
        if status == "playing":
            out.update(
                self.bot_move(moves, str(req.get("opponent", "")), int(req.get("sims", 0)), bool(req.get("hints")))
            )
        else:
            out.update({"move": None, "status": status, "hints": {}})
        return out

    def close(self) -> None:
        if self._solver is not None:
            self._solver.close()


# ---------------------------------------------------------------------------------------------
def make_server(engine: Engine, host: str, port: int) -> ThreadingHTTPServer:
    page = resources.files("c4rl").joinpath("play.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/opponents":
                self._json(200, engine.opponents())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/api/move":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length) or b"{}")
                self._json(200, engine.respond(req))
            except (GameError, ValueError, KeyError, TypeError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def serve(engine: Engine, host: str, port: int) -> None:
    server = make_server(engine, host, port)
    print(f"Connect-4: open http://localhost:{port} in your browser (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()


def play_terminal(engine: Engine, opponent: str, sims: int, human_first: bool) -> None:
    from .game import render

    moves: list[int] = []
    print(f"You are {'X' if human_first else 'O'} against {opponent}. Enter a column 1-7 (q to quit).\n")
    while True:
        cur, mask, status = replay(moves)
        print(render(cur, mask), "\n")
        if status != "playing":
            print({"x": "X wins!", "o": "O wins!", "draw": "draw"}[status])
            return
        if (len(moves) % 2 == 0) == human_first:
            raw = input("your move: ").strip()
            if raw == "q":
                return
            if not raw.isdigit() or int(raw) - 1 not in legal_cols(mask):
                print("illegal move")
                continue
            moves.append(int(raw) - 1)
        else:
            reply = engine.bot_move(moves, opponent, sims, hints=False)
            print(f"bot plays {reply['move'] + 1}")
            moves.append(int(reply["move"]))
