"""Ground truth: Pascal Pons' perfect Connect-4 solver (C++, pinned commit) driven through a persistent subprocess.

The binary is built in the Docker image (``/opt/connect4``); if it is missing the code clones and builds it into
``cache/solver`` itself, so the project also runs outside Docker. The 32 MB opening book is downloaded on first use.
Every answer is cached on disk by position, so a second run never calls the solver again.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt

from .config import BOOK_URL, SOLVER_REF, SOLVER_REPO
from .game import WIDTH, from_moves

UNPLAYABLE = -1000
I64 = npt.NDArray[np.int64]


def _find_binary(cache_dir: Path) -> Path:
    env = os.environ.get("C4_SOLVER")
    if env and Path(env).is_file():
        return Path(env)
    found = shutil.which("c4solver")
    if found:
        return Path(found)
    root = cache_dir / "solver" / "connect4"
    binary = root / "c4solver"
    if not binary.is_file():
        print("solver: building Pascal Pons' c4solver from source ...", flush=True)
        root.parent.mkdir(parents=True, exist_ok=True)
        if not (root / ".git").is_dir():
            subprocess.run(["git", "clone", SOLVER_REPO, str(root)], check=True)
        subprocess.run(["git", "checkout", SOLVER_REF], cwd=root, check=True)
        subprocess.run(["make", "c4solver"], cwd=root, check=True)
    return binary


def _find_book(cache_dir: Path) -> Path:
    book = cache_dir / "solver" / "7x6.book"
    if not book.is_file():
        print(f"solver: downloading the opening book (32 MB) from {BOOK_URL}", flush=True)
        book.parent.mkdir(parents=True, exist_ok=True)
        tmp = book.with_suffix(".part")
        urllib.request.urlretrieve(BOOK_URL, tmp)
        tmp.replace(book)
    return book


def solver_available() -> bool:
    """True when the binary can be located without building (used by tests to skip)."""
    env = os.environ.get("C4_SOLVER")
    return bool((env and Path(env).is_file()) or shutil.which("c4solver"))


class SolverBusy(RuntimeError):
    """Every solver slot is taken and the caller did not want to wait any longer."""


class Solver:
    """``analyze(moves)`` -> outcome class of every column for the player to move: +1 win, 0 draw, -1 loss.

    The solver runs in *weak* mode (it only decides win/draw/loss, not the number of moves to the end), which is
    exactly the notion of "optimal move" used in the report: a move is optimal iff it does not worsen the game-
    theoretic result of the position. ``moves`` is the history as digits ``1..7`` (``"4453"``); the order is
    irrelevant for the answer but a legal sequence has to be supplied.

    Thread-safe: up to ``workers`` solver processes run queries at the same time (started lazily). A caller that
    finds all of them busy waits at most ``wait`` seconds (``None`` = as long as it takes) and then gets
    :class:`SolverBusy`; a query that runs longer than ``query_timeout`` seconds kills its process and fails.
    """

    def __init__(
        self,
        cache_dir: Path,
        workers: int = 1,
        wait: float | None = None,
        query_timeout: float | None = None,
        flush_every: int = 2000,
        max_cache: int | None = None,
    ) -> None:
        self._cache_path = cache_dir / "solver" / "answers.json"
        self._cache: dict[str, list[int]] = {}
        if self._cache_path.is_file():
            self._cache = json.loads(self._cache_path.read_text())
        self._cache_lock = threading.Lock()
        self._dirty = 0
        self._flush_every, self._max_cache = flush_every, max_cache
        self._binary = _find_binary(cache_dir)
        self._book = _find_book(cache_dir)
        self._max_workers, self._wait, self._timeout = max(workers, 1), wait, query_timeout
        self._pool = threading.Condition()
        self._idle: list[subprocess.Popen[str]] = []
        self._all: list[subprocess.Popen[str]] = []
        self._spawned = 0
        self._release(self._acquire())  # fail early (missing binary) and keep one process warm

    # ---------------------------------------------------------------- worker processes
    def _spawn(self) -> subprocess.Popen[str]:
        proc = subprocess.Popen(
            [str(self._binary), "-a", "-w", "-b", str(self._book)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        with self._pool:
            self._all.append(proc)
        return proc

    def _acquire(self) -> subprocess.Popen[str]:
        deadline = None if self._wait is None else time.monotonic() + self._wait
        with self._pool:
            while True:
                if self._idle:
                    return self._idle.pop()
                if self._spawned < self._max_workers:
                    self._spawned += 1
                    break
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    raise SolverBusy(f"all {self._max_workers} solver slots are busy, try again in a moment")
                self._pool.wait(left)
        try:
            return self._spawn()
        except BaseException:
            with self._pool:
                self._spawned -= 1
                self._pool.notify()
            raise

    def _release(self, proc: subprocess.Popen[str], healthy: bool = True) -> None:
        if not healthy:
            proc.kill()
            proc.wait()
        with self._pool:
            if healthy:
                self._idle.append(proc)
            else:
                self._spawned -= 1
                if proc in self._all:
                    self._all.remove(proc)
            self._pool.notify()

    def _ask(self, proc: subprocess.Popen[str], moves: str) -> list[int]:
        assert proc.stdin is not None and proc.stdout is not None
        watchdog = threading.Timer(self._timeout, proc.kill) if self._timeout else None
        if watchdog:
            watchdog.start()
        try:
            proc.stdin.write(moves + "\n")
            proc.stdin.flush()
            parts = proc.stdout.readline().split()
        finally:
            if watchdog:
                watchdog.cancel()
        if len(parts) < WIDTH:  # the echoed move string is absent for the empty position
            raise RuntimeError(f"solver gave {parts!r} for {moves!r}")
        return [int(x) for x in parts[-WIDTH:]]

    # ---------------------------------------------------------------- queries
    def analyze(self, moves: str) -> list[int]:
        """Raw per-column outcome scores (sign = win/draw/loss, ``UNPLAYABLE`` for full columns)."""
        cur, mask = from_moves(moves)  # validates; also gives the cache key
        key = f"{cur}:{mask}"
        with self._cache_lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        proc = self._acquire()
        try:
            scores = self._ask(proc, moves)
        except BaseException:
            self._release(proc, healthy=False)  # a dead, hung or confused process is replaced on demand
            raise
        self._release(proc)
        with self._cache_lock:
            if self._max_cache is None or len(self._cache) < self._max_cache:
                self._cache[key] = scores
                self._dirty += 1
            if self._dirty >= self._flush_every:
                self._flush_locked()
        return scores

    def outcomes(self, moves: str) -> npt.NDArray[np.int64]:
        """``(7,)`` in {-1, 0, 1} for playable columns and ``-2`` for full ones."""
        s = np.asarray(self.analyze(moves), dtype=np.int64)
        return np.where(s == UNPLAYABLE, -2, np.sign(s))

    def best_set(self, moves: str) -> npt.NDArray[np.bool_]:
        """Boolean ``(7,)``: the columns that keep the best achievable game-theoretic result."""
        o = self.outcomes(moves)
        return np.asarray(o == o.max(), dtype=np.bool_)

    def value(self, moves: str) -> int:
        """Game-theoretic result of the position for the player to move: +1 / 0 / -1."""
        return int(self.outcomes(moves).max())

    def _flush_locked(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache))
        tmp.replace(self._cache_path)
        self._dirty = 0

    def flush(self) -> None:
        with self._cache_lock:
            self._flush_locked()

    def close(self) -> None:
        self.flush()
        with self._pool:
            procs, self._all, self._idle = self._all, [], []
        for proc in procs:
            if proc.stdin:
                proc.stdin.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def __enter__(self) -> Solver:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
