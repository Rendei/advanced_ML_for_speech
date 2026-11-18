"""Unit tests: the bitboard engine against a naive board, MCTS tactics, replay buffer, Elo, self-play shapes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from c4rl.config import AZConfig, EvalConfig, NetConfig, make_config
from c4rl.eda import count_positions, random_play
from c4rl.evaluation import PositionSet, fit_elo, run_tournament, score_player
from c4rl.game import (
    BOARD_MASK,
    CELLS,
    HEIGHT,
    WIDTH,
    from_moves,
    legal_cols,
    legal_moves,
    mirror_bits,
    planes,
    play,
    play_int,
    render,
    to_grid,
)
from c4rl.history import ReplayBuffer
from c4rl.mcts import BatchMCTS, RolloutEvaluator, policy_from_visits
from c4rl.net import NetEvaluator, PVNet
from c4rl.players import GreedyPlayer, MCTSPlayer, MinimaxPlayer, RandomPlayer, play_games, random_openings
from c4rl.solver import Solver, solver_available
from c4rl.train_az import self_play


# ---------------------------------------------------------------------------------------------
# reference implementation on a plain 2D list
# ---------------------------------------------------------------------------------------------
def naive_wins(board: list[list[int]], player: int) -> bool:
    for r in range(HEIGHT):
        for c in range(WIDTH):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + k * dr, c + k * dc) for k in range(4)]
                if all(0 <= y < HEIGHT and 0 <= x < WIDTH and board[y][x] == player for y, x in cells):
                    return True
    return False


def test_bitboard_matches_naive_board_on_random_games() -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        board = [[0] * WIDTH for _ in range(HEIGHT)]
        heights = [0] * WIDTH
        cur = mask = 0
        player = 1
        for _ply in range(CELLS):
            legal = legal_cols(mask)
            assert legal == [c for c in range(WIDTH) if heights[c] < HEIGHT]
            if not legal:
                break
            c = int(rng.choice(legal))
            board[heights[c]][c] = player
            heights[c] += 1
            cur, mask, won = play_int(cur, mask, c)
            assert won == naive_wins(board, player)
            assert to_grid(cur, mask) == board
            if won:
                break
            player = 3 - player


def test_vectorised_play_equals_scalar() -> None:
    rng = np.random.default_rng(1)
    cur = np.zeros(64, dtype=np.int64)
    mask = np.zeros(64, dtype=np.int64)
    for _ in range(20):
        legal = legal_moves(mask)
        a = (rng.random(legal.shape) * legal - (~legal)).argmax(axis=1).astype(np.int64)
        ncur, nmask, won = play(cur, mask, a)
        for i in range(64):
            sc, sm, sw = play_int(int(cur[i]), int(mask[i]), int(a[i]))
            assert (sc, sm, sw) == (int(ncur[i]), int(nmask[i]), bool(won[i]))
        keep = ~won
        cur, mask = np.where(keep, ncur, 0), np.where(keep, nmask, 0)


def test_from_moves_and_render() -> None:
    cur, mask = from_moves("4453")
    assert bin(mask).count("1") == 4
    text = render(cur, mask)
    assert text.count("X") == 2 and text.count("O") == 2
    with pytest.raises(ValueError):
        from_moves("1111111")  # column overflow
    with pytest.raises(ValueError):
        from_moves("1212121")  # vertical four for X on the seventh move


def test_mirror_is_an_involution_and_commutes_with_play() -> None:
    cur, mask = from_moves("43521")
    c, m = np.array([cur]), np.array([mask])
    assert (mirror_bits(mirror_bits(m)) == m).all()
    mc, mm = mirror_bits(c), mirror_bits(m)
    a = np.array([2])
    n1c, n1m, _ = play(c, m, a)
    n2c, n2m, _ = play(mc, mm, np.array([WIDTH - 1 - 2]))
    assert (mirror_bits(n1c) == n2c).all() and (mirror_bits(n1m) == n2m).all()


def test_planes_shape_and_content() -> None:
    cur, mask = from_moves("44")
    x = planes(np.array([cur]), np.array([mask]))
    assert x.shape == (1, 3, HEIGHT, WIDTH)
    assert x[0, 0].sum() == 1 and x[0, 1].sum() == 1 and (x[0, 2] == 1).all()
    assert x[0, 0, 0, 3] == 1 and x[0, 1, 1, 3] == 1  # own stone at the bottom of column 4, the opponent on top of it


def test_position_counts_match_known_values() -> None:
    counts = count_positions(6)
    assert counts.total == [1, 7, 49, 238, 1120, 4263, 16422]
    assert sum(counts.terminal) == 0
    assert all(m <= t for m, t in zip(counts.up_to_mirror, counts.total, strict=True))


def test_random_play_statistics_are_sane() -> None:
    rp = random_play(2000, np.random.default_rng(0))
    assert rp.length.min() >= 7 and rp.length.max() <= CELLS
    assert 0.45 < (rp.result == 1).mean() < 0.65  # the first player has an edge in random play


# ---------------------------------------------------------------------------------------------
# search and players
# ---------------------------------------------------------------------------------------------
def _batch(moves: str) -> tuple[np.ndarray, np.ndarray]:
    cur, mask = from_moves(moves)
    return np.array([cur], dtype=np.int64), np.array([mask], dtype=np.int64)


def test_mcts_with_rollouts_takes_a_win_and_blocks() -> None:
    rng = np.random.default_rng(0)
    mcts = BatchMCTS(RolloutEvaluator(rng, 2), 200, rng)
    # X has 1,2,3 on the bottom row (O played 7,7): X to move wins with column 4... moves: X1 O7 X2 O7 X3 O5
    cur, mask = _batch("172735")
    v = mcts.search(cur, mask).visits[0]
    assert v.argmax() == 3
    # X threatens 1-2-3-4 on the bottom row, O (to move) must block column 4
    cur, mask = _batch("15253")
    assert mcts.search(cur, mask).visits[0].argmax() == 3


def test_mcts_visits_sum_to_sims_and_ignore_full_columns() -> None:
    rng = np.random.default_rng(1)
    mcts = BatchMCTS(RolloutEvaluator(rng), 50, rng)
    cur, mask = _batch("111111")  # column 1 is full
    res = mcts.search(cur, mask, noise=True)
    assert res.visits[0].sum() == pytest.approx(50)
    assert res.visits[0, 0] == 0
    assert -1.0 <= res.value[0] <= 1.0


def test_policy_from_visits() -> None:
    v = np.array([[0, 3, 1, 0, 0, 0, 0]], dtype=np.float32)
    assert policy_from_visits(v, 0.0, np.random.default_rng(0))[0].argmax() == 1
    p = policy_from_visits(v, 1.0)
    assert p[0].sum() == pytest.approx(1.0) and p[0, 1] == pytest.approx(0.75)


def test_greedy_and_minimax_see_one_move_tactics() -> None:
    rng = np.random.default_rng(0)
    cur, mask = _batch("172735")  # X (to move) wins at column 4
    assert GreedyPlayer(rng).act(cur, mask, [""])[0] == 3
    assert MinimaxPlayer(2, rng).act(cur, mask, [""])[0] == 3
    # X: 1,2,3 ... O to move must block column 4 (O has nothing better): moves "1 5 2 5 3" -> O to move
    cur, mask = _batch("15253")
    assert GreedyPlayer(rng).act(cur, mask, [""])[0] == 3
    assert MinimaxPlayer(2, rng).act(cur, mask, [""])[0] == 3


def test_play_games_outcomes_and_lengths() -> None:
    rng = np.random.default_rng(0)
    openings = random_openings(20, 2, rng)
    res = play_games(GreedyPlayer(rng), RandomPlayer(rng), openings)
    assert set(np.unique(res.outcome)) <= {-1, 0, 1}
    assert all(3 <= len(m) <= CELLS for m in res.moves)
    assert (res.outcome == 1).mean() > 0.6  # greedy beats random as the first player


def test_replay_buffer_ring_and_mirror() -> None:
    buf = ReplayBuffer(10)
    cur, mask = from_moves("4")
    for _ in range(3):
        n = 6
        buf.add(
            np.full(n, cur, dtype=np.int64),
            np.full(n, mask, dtype=np.int64),
            np.tile(np.arange(WIDTH, dtype=np.float32) / 21.0, (n, 1)),
            np.ones(n, dtype=np.float32),
        )
    assert buf.size == 10
    c, _m, pi, z = buf.sample(64, np.random.default_rng(0))
    assert c.shape == (64,) and pi.shape == (64, WIDTH) and (z == 1).all()
    assert np.allclose(pi.sum(1), 1.0)


def test_fit_elo_recovers_order_and_scale() -> None:
    rng = np.random.default_rng(0)
    true = np.array([0.0, 200.0, 400.0])
    pair: dict[tuple[int, int], np.ndarray] = {}
    for i in range(3):
        for j in range(i + 1, 3):
            p = 1 / (1 + 10 ** (-(true[i] - true[j]) / 400))
            pair[(i, j)] = (rng.random(400) < p).astype(np.float64)
    elo = fit_elo(["a", "b", "c"], pair, anchor="a")
    assert elo[0] == pytest.approx(0.0)
    assert elo[0] < elo[1] < elo[2]
    assert abs(elo[2] - 400) < 60


def test_random_net_self_play_shapes() -> None:
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = PVNet(NetConfig(8, 1)).eval()
    cfg = AZConfig(sims=4, temp_moves=4)
    cur, mask, pi, z, stats = self_play(net, torch.device("cpu"), cfg, 6, rng)
    assert len(cur) == len(mask) == len(pi) == len(z)
    assert np.allclose(pi.sum(1), 1.0, atol=1e-5)
    assert set(np.unique(z)) <= {-1.0, 0.0, 1.0}
    assert stats[0] >= 7
    assert (mask & ~BOARD_MASK == 0).all()


def test_net_evaluator_outputs() -> None:
    net = PVNet(NetConfig(8, 1))
    ev = NetEvaluator(net, torch.device("cpu"))
    cur, mask = _batch("4453")
    p, v = ev(cur, mask)
    assert p.shape == (1, WIDTH) and v.shape == (1,)
    assert p.sum() == pytest.approx(1.0, abs=1e-5) and -1 <= v[0] <= 1


def test_profiles_exist() -> None:
    for name in ("smoke", "standard", "full"):
        cfg = make_config(name, Path("o"), Path("c"))
        assert cfg.az.sims > cfg.az_lite.sims or name == "smoke"
        assert isinstance(cfg.ev, EvalConfig)
    with pytest.raises(ValueError):
        make_config("nope", Path("o"), Path("c"))


# ---------------------------------------------------------------------------------------------
# solver-dependent tests (run inside the Docker image, skipped elsewhere)
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def solver_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One cache directory for the whole session so that the 32 MB opening book is downloaded only once."""
    cache = Path("cache")  # the cache mounted by docker compose (book downloaded once and shared with real runs)
    return cache if cache.is_dir() else tmp_path_factory.mktemp("solver_cache")


needs_solver = pytest.mark.skipif(not solver_available(), reason="c4solver binary not installed")


@needs_solver
def test_solver_known_first_moves(solver_cache: Path) -> None:
    with Solver(solver_cache) as s:
        out = s.outcomes("")
        assert out[3] == 1  # the centre column wins for the first player
        assert out[2] == 0 and out[4] == 0  # columns 3 and 5 draw
        assert out[0] == -1 and out[6] == -1  # the edge columns lose
        assert s.value("") == 1


@needs_solver
def test_solver_agrees_with_tactics_and_is_cached(solver_cache: Path) -> None:
    with Solver(solver_cache) as s:
        best = s.best_set("172735")  # X wins immediately with column 4
        assert best[3] and best.sum() >= 1
        first = s.analyze("4453")
        assert s.analyze("4453") == first  # served from the cache
    assert (solver_cache / "solver" / "answers.json").is_file()


@needs_solver
def test_perfect_player_never_loses_a_won_game(solver_cache: Path) -> None:
    from c4rl.players import PerfectPlayer

    rng = np.random.default_rng(0)
    with Solver(solver_cache) as s:
        res = play_games(PerfectPlayer(s, rng), RandomPlayer(rng), [[3, 3]] * 4)
        assert (res.outcome != -1).all()  # perfect as X from a symmetric opening never loses to random


@needs_solver
def test_position_set_labels_are_consistent(solver_cache: Path) -> None:
    from c4rl.evaluation import make_position_sets

    cfg = EvalConfig(n_positions=12, min_ply=6, max_ply=20)
    with Solver(solver_cache) as s:
        sets = make_position_sets(cfg, s, seed=0)
    for ps in sets.values():
        assert isinstance(ps, PositionSet) and len(ps) == 12
        assert (ps.best.sum(1) >= 1).all()
        assert ((ps.outcome.max(1)) == ps.value).all()
        rng = np.random.default_rng(0)
        sc = score_player(RandomPlayer(rng), ps)
        assert sc.correct.shape == (12,)


def test_tournament_runs_without_solver() -> None:
    rng = np.random.default_rng(0)
    players = [RandomPlayer(rng), GreedyPlayer(rng), MCTSPlayer(RolloutEvaluator(rng), 20, rng, "uct-20")]
    t = run_tournament(players, 4, 2, rng)
    score, games = t.points()
    assert games[0, 1] == 4 and score.shape == (3, 3)
    elo = fit_elo(t.names, t.pair_scores, anchor="random")
    assert elo[0] == pytest.approx(0.0) and elo[1] > elo[0]
