"""Connect-4 on 64-bit bitboards, vectorised over a batch of positions with numpy.

Layout (the one used by Pascal Pons' solver): column ``c`` occupies bits ``7c .. 7c+5`` (row 0 is the bottom),
bit ``7c+6`` is an always-empty sentinel that keeps the shifted alignment tests from wrapping around columns.

A position is a pair ``(cur, mask)``:
  * ``mask`` - every stone on the board,
  * ``cur``  - the stones of the player **to move**.
The opponent's stones are therefore ``mask ^ cur``, and nothing else (no colour, no move counter) is stored.
The first player of the game is called X, the second O; whose turn it is follows from ``popcount(mask)``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

WIDTH = 7
HEIGHT = 6
CELLS = WIDTH * HEIGHT
H1 = HEIGHT + 1

I64 = npt.NDArray[np.int64]
F32 = npt.NDArray[np.float32]
BoolArr = npt.NDArray[np.bool_]

BOTTOM: int = sum(1 << (c * H1) for c in range(WIDTH))
BOARD_MASK: int = BOTTOM * ((1 << HEIGHT) - 1)  # the 42 playable cells
# bit of the top playable cell of every column; a column is full iff this bit is set
TOP_BIT: I64 = np.array([1 << (c * H1 + HEIGHT - 1) for c in range(WIDTH)], dtype=np.int64)
COL_BOTTOM: I64 = np.array([1 << (c * H1) for c in range(WIDTH)], dtype=np.int64)
# bit position of cell (row, col), shape (HEIGHT, WIDTH), row 0 at the bottom
CELL_SHIFT: I64 = np.array([[c * H1 + r for c in range(WIDTH)] for r in range(HEIGHT)], dtype=np.int64)

EMPTY_CUR = 0
EMPTY_MASK = 0


# ---------------------------------------------------------------------------------------------
# vectorised primitives
# ---------------------------------------------------------------------------------------------
def has_four(pos: I64) -> BoolArr:
    """True where ``pos`` (one player's stones) contains four in a row (any of the four directions)."""
    out = np.zeros(pos.shape, dtype=np.bool_)
    for shift in (H1, HEIGHT, HEIGHT + 2, 1):  # horizontal, diagonal, anti-diagonal, vertical
        m = pos & (pos >> shift)
        out |= (m & (m >> (2 * shift))) != 0
    return out


def legal_moves(mask: I64) -> BoolArr:
    """Boolean ``(..., 7)``: which columns still have room."""
    return np.asarray((mask[..., None] & TOP_BIT) == 0, dtype=np.bool_)


def play(cur: I64, mask: I64, action: npt.NDArray[np.integer]) -> tuple[I64, I64, BoolArr]:
    """Drop a stone of the player to move into column ``action`` (must be legal).

    Returns ``(new_cur, new_mask, won)``: the position seen by the *next* player to move, and whether the move
    just played completed four in a row.
    """
    new_mask = mask | (mask + COL_BOTTOM[action])
    mover = cur | (new_mask ^ mask)
    return mask ^ cur, new_mask, has_four(mover)


def n_stones(mask: I64) -> npt.NDArray[np.int64]:
    return np.bitwise_count(mask).astype(np.int64)


def planes(cur: I64, mask: I64) -> F32:
    """Network input ``(N, 3, 6, 7)``: stones of the player to move, stones of the opponent, constant ones plane."""
    own = (cur[:, None, None] >> CELL_SHIFT) & 1
    opp = ((mask ^ cur)[:, None, None] >> CELL_SHIFT) & 1
    out = np.empty((len(cur), 3, HEIGHT, WIDTH), dtype=np.float32)
    out[:, 0] = own
    out[:, 1] = opp
    out[:, 2] = 1.0
    return out


def mirror_action(a: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
    return (WIDTH - 1) - a


def mirror_bits(x: I64) -> I64:
    """Left-right mirror of a bitboard (column ``c`` -> ``6 - c``)."""
    out = np.zeros_like(x)
    for c in range(WIDTH):
        col = (x >> (c * H1)) & ((1 << HEIGHT) - 1)
        out |= col << ((WIDTH - 1 - c) * H1)
    return out


# ---------------------------------------------------------------------------------------------
# scalar helpers (Python ints): move strings, minimax, pretty printing
# ---------------------------------------------------------------------------------------------
def has_four_int(pos: int) -> bool:
    for shift in (H1, HEIGHT, HEIGHT + 2, 1):
        m = pos & (pos >> shift)
        if m & (m >> (2 * shift)):
            return True
    return False


def play_int(cur: int, mask: int, col: int) -> tuple[int, int, bool]:
    new_mask = mask | (mask + (1 << (col * H1)))
    mover = cur | (new_mask ^ mask)
    return mask ^ cur, new_mask, has_four_int(mover)


def legal_cols(mask: int) -> list[int]:
    return [c for c in range(WIDTH) if not mask & (1 << (c * H1 + HEIGHT - 1))]


def from_moves(moves: str) -> tuple[int, int]:
    """Position after a solver-style move string, columns numbered ``1..7`` (``"4453"`` = X:4, O:4, X:5, O:3)."""
    cur, mask = 0, 0
    for ch in moves:
        col = int(ch) - 1
        if not 0 <= col < WIDTH or mask & (1 << (col * H1 + HEIGHT - 1)):
            raise ValueError(f"illegal move {ch!r} in {moves!r}")
        cur, mask, won = play_int(cur, mask, col)
        if won:
            raise ValueError(f"game already won inside {moves!r}")
    return cur, mask


def to_moves(cols: list[int]) -> str:
    return "".join(str(c + 1) for c in cols)


def render(cur: int, mask: int) -> str:
    """ASCII board; X is the first player. Row 5 (top) is printed first."""
    x_to_move = bin(mask).count("1") % 2 == 0
    x_bits, o_bits = (cur, mask ^ cur) if x_to_move else (mask ^ cur, cur)
    rows = []
    for r in range(HEIGHT - 1, -1, -1):
        cells = []
        for c in range(WIDTH):
            bit = 1 << (c * H1 + r)
            cells.append("X" if x_bits & bit else "O" if o_bits & bit else ".")
        rows.append(" ".join(cells))
    rows.append("1 2 3 4 5 6 7")
    return "\n".join(rows)


def to_grid(cur: int, mask: int) -> list[list[int]]:
    """``grid[row][col]`` with 0 = empty, 1 = X (first player), 2 = O; row 0 is the bottom."""
    x_to_move = bin(mask).count("1") % 2 == 0
    x_bits, o_bits = (cur, mask ^ cur) if x_to_move else (mask ^ cur, cur)
    return [
        [1 if x_bits & (1 << (c * H1 + r)) else 2 if o_bits & (1 << (c * H1 + r)) else 0 for c in range(WIDTH)]
        for r in range(HEIGHT)
    ]
