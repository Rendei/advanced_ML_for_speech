"""Play against the trained agents: a small stdlib web server (``python -m c4rl serve``) and a terminal mode.

The server is stateless: the browser sends the list of moves played so far, the server replays it, asks the chosen
opponent for a move and answers with that move plus optional "thoughts" (policy, MCTS visit shares, value) and an
optional solver verdict on the human's last move.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
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
from .solver import Solver, SolverBusy, solver_available

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


class Busy(RuntimeError):
    """All computation slots are taken; the client should retry in a moment (HTTP 503)."""


@dataclass(frozen=True)
class Limits:
    """Guard rails for a server that is reachable from the internet (``--public``).

    The defaults are harmless for local play; ``public`` additionally switches off everything that needs the
    solver (the perfect opponent and the move verdicts): they download a 32 MB book and burn CPU. With
    ``solver_slots`` > 0 they come back, but at most that many solver queries run at the same time (each slot is a
    solver process of ~100 MB and one CPU core's worth of work); the others are told to retry.
    """

    public: bool = False
    solver_slots: int = 0  # public mode only: simultaneous solver queries (0 = solver switched off)
    solver_timeout: float = 30.0  # public mode: seconds before a solver query is killed
    max_sims: int = 400  # MCTS simulations per move a client may ask for
    max_body: int = 4096  # bytes of a request body
    max_inflight: int = 4  # bot computations running at the same time; the rest get 503
    rate_per_min: int = 120  # requests per client address and minute; the rest get 429
    trust_proxy: bool = False  # take the client address from X-Forwarded-For (only behind our own reverse proxy)
    io_timeout: float = 10.0  # seconds a client may stall while sending a request


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

    def __init__(
        self, registry: dict[str, ModelInfo], device: torch.device, cache_dir: Path, limits: Limits = Limits()
    ) -> None:
        self.registry = registry
        self.device = device
        self.cache_dir = cache_dir
        self.limits = limits
        self.rng = np.random.default_rng()
        self._models: dict[str, PVNet | QNet] = {}
        self._solver: Solver | None = None
        self._solver_error: str | None = None
        self._solver_lock = threading.Lock()
        self._lock = threading.Lock()  # one network search at a time: the models are shared
        self._slots = threading.BoundedSemaphore(max(limits.max_inflight, 1))
        self._no_slots = limits.max_inflight < 1

    @property
    def solver_allowed(self) -> bool:
        return not self.limits.public or self.limits.solver_slots > 0

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
            if pid == "perfect" and not self.solver_allowed:
                continue
            if pid == "perfect" and not solver_available() and self._solver is None:
                label += " — нужен c4solver"
            if pid == "perfect" and self.limits.public:
                label += f" — не более {self.limits.solver_slots} одновременно"
            out.append({"id": pid, "label": label, "group": "Классические боты", "searchable": False})
        return out

    # ------------------------------------------------------------------ bot
    def _model(self, model_id: str) -> PVNet | QNet:
        if model_id not in self._models:
            self._models[model_id] = load_model(self.registry[model_id], self.device)
        return self._models[model_id]

    def _make_solver(self) -> Solver:
        lim = self.limits
        if not lim.public:
            return Solver(self.cache_dir)
        return Solver(
            self.cache_dir,
            workers=lim.solver_slots,
            wait=1.0,  # a short queue smooths out bursts; beyond it the player is told the slots are taken
            query_timeout=lim.solver_timeout,
            flush_every=25,  # the container may be stopped at any moment
            max_cache=300_000,
        )

    def warm_up(self) -> None:
        """Public mode: start the solver (and download its book) in the background, before the first player."""
        if not (self.limits.public and self.solver_allowed):
            return

        def work() -> None:
            try:
                solver = self._make_solver()
                with self._solver_lock:
                    self._solver = solver
            except Exception as exc:  # no network / no binary: the solver stays off, the game itself works
                self._solver_error = f"{type(exc).__name__}: {exc}"
                print(f"solver warm-up failed: {self._solver_error}", flush=True)

        threading.Thread(target=work, name="solver-warmup", daemon=True).start()

    def _get_solver(self) -> Solver:
        if self.limits.public:  # never make a request wait for the book download
            if self._solver is None:
                raise SolverBusy(self._solver_error or "the solver is still starting, try again in a minute")
            return self._solver
        with self._solver_lock:
            if self._solver is None:
                self._solver = self._make_solver()
            return self._solver

    def bot_move(self, moves: list[int], opponent: str, sims: int, hints: bool) -> dict[str, Any]:
        cur, mask, status = replay(moves)
        if status != "playing":
            raise GameError("the game is already over")
        c = np.array([cur], dtype=np.int64)
        m = np.array([mask], dtype=np.int64)
        legal = legal_moves(m)
        info: dict[str, Any] = {}
        # the solver is thread-safe and bounded by its own slots: the perfect opponent does not queue behind the nets
        with nullcontext() if opponent == "perfect" else self._lock:
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
        if not self.solver_allowed:
            raise GameError("the solver is switched off on this server")
        return PerfectPlayer(self._get_solver(), np.random.default_rng())

    # ------------------------------------------------------------------ judge
    def judge(self, before: list[int], chosen: int) -> dict[str, Any]:
        """Solver verdict on ``chosen`` played in the position after ``before``."""
        replay(before)
        if not self.solver_allowed:
            raise GameError("the solver is switched off on this server")
        outcomes = self._get_solver().outcomes(to_moves(before))
        best = int(outcomes.max())
        got = int(outcomes[chosen])
        return {
            "optimal": got == best,
            "best_columns": [int(k) for k in np.nonzero(outcomes == best)[0]],
            "position": OUTCOME_RU[best],
            "after_move": OUTCOME_RU.get(got, "?"),
        }

    def info(self) -> dict[str, Any]:
        return {
            "public": self.limits.public,
            "max_sims": self.limits.max_sims,
            "solver": self.solver_allowed,
            "solver_slots": self.limits.solver_slots if self.limits.public else None,
        }

    def respond(self, req: dict[str, Any]) -> dict[str, Any]:
        """Handle one ``/api/move`` request (validated here: this is the only entry point reachable from a browser)."""
        if not isinstance(req, dict) or not isinstance(req.get("moves", []), list):
            raise GameError("malformed request")
        moves = [int(x) for x in req.get("moves", [])]
        sims = int(req.get("sims", 0))
        opponent = str(req.get("opponent", ""))
        if not 0 <= sims <= self.limits.max_sims:
            raise GameError(f"sims must be between 0 and {self.limits.max_sims}")
        if not self.solver_allowed and (req.get("judge") or opponent == "perfect"):
            raise GameError("the solver is switched off on this server")
        out: dict[str, Any] = {}
        _, _, status = replay(moves)
        if req.get("judge") and moves:
            try:
                out["judge"] = self.judge(moves[:-1], moves[-1])
            except (OSError, RuntimeError) as exc:  # solver busy / binary missing / book download failed
                out["judge_error"] = str(exc)
        if status == "playing":
            if self._no_slots or not self._slots.acquire(blocking=False):
                raise Busy("the server is busy, try again in a moment")
            try:
                out.update(self.bot_move(moves, opponent, sims, bool(req.get("hints"))))
            except SolverBusy as exc:
                raise Busy(str(exc)) from exc
            finally:
                self._slots.release()
        else:
            out.update({"move": None, "status": status, "hints": {}})
        return out

    def close(self) -> None:
        if self._solver is not None:
            self._solver.close()


# ---------------------------------------------------------------------------------------------
class _RateLimiter:
    """Sliding one-minute window per client address (thread-safe, bounded memory)."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > 10_000:  # forget idle clients
                self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < 60}
            q = self._hits.setdefault(client, deque())
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


def make_server(engine: Engine, host: str, port: int) -> ThreadingHTTPServer:
    page = resources.files("c4rl").joinpath("play.html").read_bytes()
    limits = engine.limits
    limiter = _RateLimiter(limits.rate_per_min)
    csp = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:"
    )

    class Handler(BaseHTTPRequestHandler):
        timeout = limits.io_timeout  # a client that stalls mid-request does not hold a thread forever

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if ctype.startswith("text/html"):
                self.send_header("Content-Security-Policy", csp)
                self.send_header("X-Frame-Options", "DENY")
            else:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _client(self) -> str:
            if limits.trust_proxy:
                forwarded = self.headers.get("X-Forwarded-For", "")
                if forwarded:
                    return forwarded.split(",")[-1].strip()  # the address our own proxy saw
            return str(self.client_address[0])

        def _admit(self) -> bool:
            if limiter.allow(self._client()):
                return True
            self._json(429, {"error": "too many requests, slow down"})
            return False

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send(200, b"ok", "text/plain; charset=utf-8")
            elif not self._admit():
                return
            elif self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/opponents":
                self._json(200, engine.opponents())
            elif self.path == "/api/info":
                self._json(200, engine.info())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._admit():
                return
            if self.path != "/api/move":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0:
                    raise ValueError("bad Content-Length")
                if length > limits.max_body:
                    self._json(413, {"error": "request too large"})
                    return
                req = json.loads(self.rfile.read(length) or b"{}")
                self._json(200, engine.respond(req))
            except Busy as exc:
                self._json(503, {"error": str(exc)})
            except (GameError, ValueError, KeyError, TypeError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def serve(engine: Engine, host: str, port: int) -> None:
    server = make_server(engine, host, port)
    engine.warm_up()
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
