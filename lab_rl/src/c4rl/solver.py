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


class Solver:
    """``analyze(moves)`` -> outcome class of every column for the player to move: +1 win, 0 draw, -1 loss.

    The solver runs in *weak* mode (it only decides win/draw/loss, not the number of moves to the end), which is
    exactly the notion of "optimal move" used in the report: a move is optimal iff it does not worsen the game-
    theoretic result of the position. ``moves`` is the history as digits ``1..7`` (``"4453"``); the order is
    irrelevant for the answer but a legal sequence has to be supplied.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_path = cache_dir / "solver" / "answers.json"
        self._cache: dict[str, list[int]] = {}
        if self._cache_path.is_file():
            self._cache = json.loads(self._cache_path.read_text())
        self._dirty = 0
        binary = _find_binary(cache_dir)
        book = _find_book(cache_dir)
        self._proc = subprocess.Popen(
            [str(binary), "-a", "-w", "-b", str(book)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def analyze(self, moves: str) -> list[int]:
        """Raw per-column outcome scores (sign = win/draw/loss, ``UNPLAYABLE`` for full columns)."""
        cur, mask = from_moves(moves)  # validates; also gives the cache key
        key = f"{cur}:{mask}"
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(moves + "\n")
        self._proc.stdin.flush()
        parts = self._proc.stdout.readline().split()
        if len(parts) < WIDTH:  # the echoed move string is absent for the empty position
            raise RuntimeError(f"solver gave {parts!r} for {moves!r}")
        scores = [int(x) for x in parts[-WIDTH:]]
        self._cache[key] = scores
        self._dirty += 1
        if self._dirty >= 2000:
            self.flush()
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

    def flush(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache))
        tmp.replace(self._cache_path)
        self._dirty = 0

    def close(self) -> None:
        self.flush()
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait(timeout=10)

    def __enter__(self) -> Solver:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
