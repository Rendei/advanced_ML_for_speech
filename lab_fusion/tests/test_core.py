"""Fast tests on tiny random tensors: no downloads, no pretrained models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fusionlab.config import CLASSES, MODEL_NAMES, Config, make_config
from fusionlab.models import Model
from fusionlab.tensors import Batch, session_splits
from fusionlab.train import confusion, scores, trace_shapes

D = 16


def _cfg() -> Config:
    return Config(d_model=32, n_heads=4, n_cross_layers=2, dropout=0.0)


def _batch(b: int = 5, ta: int = 9, tt: int = 6) -> Batch:
    g = torch.Generator().manual_seed(0)
    audio_mask = torch.ones(b, ta, dtype=torch.bool)
    audio_mask[0, 5:] = False
    text_mask = torch.ones(b, tt, dtype=torch.bool)
    text_mask[1, 3:] = False
    return Batch(
        torch.randn(b, ta, D, generator=g), audio_mask, torch.randn(b, tt, D, generator=g), text_mask,
        torch.randint(0, 4, (b,), generator=g),
    )


def test_scores_perfect_and_chance() -> None:
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    assert scores(y, y, 4) == {"wa": 1.0, "ua": 1.0, "f1": 1.0}
    s = scores(y, np.zeros_like(y), 4)
    assert s["wa"] == 0.25 and s["ua"] == 0.25


def test_ua_differs_from_wa_on_imbalance() -> None:
    y = np.array([0] * 8 + [1] * 2)
    s = scores(y, np.zeros_like(y), 4)
    assert s["wa"] == pytest.approx(0.8)
    assert s["ua"] == pytest.approx(0.5)  # recall 1.0 on class 0, 0.0 on class 1; absent classes are ignored


def test_confusion_counts() -> None:
    cm = confusion(np.array([0, 1, 1]), np.array([0, 1, 0]), 4)
    assert cm[0, 0] == 1 and cm[1, 1] == 1 and cm[1, 0] == 1 and cm.sum() == 3


def test_session_splits_are_speaker_disjoint() -> None:
    session = np.repeat(np.arange(1, 6), 10)
    splits = session_splits(session, 5)
    assert len(splits) == 5
    for sp in splits:
        parts = [set(sp.train), set(sp.val), set(sp.test)]
        assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        assert len(parts[0] | parts[1] | parts[2]) == 50
        assert len({int(session[i]) for i in sp.test}) == 1
    assert sorted(int(session[sp.test[0]]) for sp in splits) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("kind", MODEL_NAMES)
def test_model_output_shape_and_backward(kind: str) -> None:
    model = Model(kind, _cfg(), D)
    b = _batch()
    logits = model(b)
    assert logits.shape == (5, len(CLASSES))
    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_cross_attention_ignores_padded_audio() -> None:
    """Changing padded audio frames must not change the output (K/V mask is respected)."""
    model = Model("cross", _cfg(), D).eval()
    b = _batch()
    out1 = model(b)
    audio2 = b.audio.clone()
    audio2[0, 5:] = 100.0  # padded frames of sample 0
    out2 = model(Batch(audio2, b.audio_mask, b.text, b.text_mask, b.y))
    assert torch.allclose(out1, out2, atol=1e-5)
    assert model.last_attention is not None
    assert float(model.last_attention[0, :, 5:].detach().sum()) == pytest.approx(0.0, abs=1e-6)


def test_cross_model_uses_audio_but_text_model_does_not() -> None:
    b = _batch()
    other = Batch(torch.randn_like(b.audio), b.audio_mask, b.text, b.text_mask, b.y)
    torch.manual_seed(0)
    cross = Model("cross", _cfg(), D).eval()
    assert not torch.allclose(cross(b), cross(other), atol=1e-5)
    text = Model("text", _cfg(), D).eval()
    assert torch.allclose(text(b), text(other), atol=1e-6)


def test_trace_shapes_reports_cross_blocks() -> None:
    rows = dict(trace_shapes(Model("cross", _cfg(), D), _batch()))
    assert rows["input.audio"] == "(5, 9, 16)"
    assert rows["t_from_a.0"] == "(5, 6, 32)"
    assert rows["head"] == "(5, 4)"


def test_profiles() -> None:
    assert make_config("smoke").limit is not None
    assert make_config("standard").limit is None
    assert make_config("standard").tune_trials == 0 and make_config("tuned").tune_trials > 1
    with pytest.raises(ValueError):
        make_config("nope")


def test_row_similarity_identical_vs_diagonal() -> None:
    from fusionlab.report import _row_similarity

    same = np.tile(np.array([[0.1, 0.8, 0.1]]), (6, 1))
    diag = np.eye(6)
    assert _row_similarity(same) == pytest.approx(1.0)
    assert _row_similarity(diag) == pytest.approx(0.0)


def test_bootstrap_detects_a_clear_difference() -> None:
    from fusionlab.analysis import bootstrap_diff
    from fusionlab.experiment import Results

    rng = np.random.default_rng(0)
    n, c = 400, 4
    y = rng.integers(0, c, n)
    good = np.eye(c)[y] * 0.7 + 0.075  # always right
    bad = rng.dirichlet(np.ones(c), n)  # chance level
    res = Results(
        y=y, session=np.ones(n, dtype=np.int64), covered=np.ones(n, dtype=bool),
        probs={"a": good[None], "b": bad[None]}, probs_shuffled={}, fold_scores={}, histories={}, attention=None,
        shapes=[], n_params={}, seconds=0.0, seeds=(0,),
    )
    d = bootstrap_diff(res, "a", "b", Config(n_bootstrap=100))
    assert d.lo > 0.4 and d.p_not_better == 0.0


def test_sample_trials_default_first_distinct_and_kind_aware() -> None:
    from fusionlab.config import SEARCH_SPACE, sample_trials

    base = Config()
    trials = sample_trials(base, "cross", 8)
    assert trials[0] is base and len(trials) == 8
    key = [tuple(getattr(t, k) for k in SEARCH_SPACE) for t in trials]
    assert len(set(key)) == 8  # all distinct
    assert all(t.d_model % t.n_heads == 0 for t in trials)
    # models without cross-attention never vary the number of cross layers
    assert {t.n_cross_layers for t in sample_trials(base, "text", 8)} == {base.n_cross_layers}
    assert sample_trials(base, "audio", 5) == sample_trials(base, "audio", 5)  # deterministic


def test_sample_trials_stops_when_space_is_exhausted() -> None:
    from fusionlab.config import sample_trials

    assert len(sample_trials(Config(), "text", 10_000)) < 10_000
