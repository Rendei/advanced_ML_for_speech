"""Fast tests on tiny random tensors: no downloads, no pretrained models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from mtllab.analysis import bootstrap_diff, conflict_share, mean_cosines, metric
from mtllab.balancing import (
    UncertaintyWeighting,
    cosine_matrix,
    fixed_weights,
    pcgrad_merge,
    shared_task_grads,
)
from mtllab.config import (
    REGIME_BY_NAME,
    REGIMES,
    TASKS,
    Config,
    is_regression,
    make_config,
    primary_metric,
    task_dim,
)
from mtllab.data import _UTT_RE, Meta
from mtllab.experiment import run_experiment
from mtllab.features import Features
from mtllab.models import MTLNet, Trunk
from mtllab.report import build_eda, build_report
from mtllab.tasks import Standardizer, ccc, class_scores, decode_outputs, task_losses, task_scores
from mtllab.tensors import Batch, Dataset, session_splits
from mtllab.train import predict, train_regime

D = 16  # toy feature dimension


def toy_features(n: int = 120, seed: int = 0) -> Features:
    """Random features and labels with the same structure as the real corpus (5 sessions, dialogues, 3 VAD scales)."""
    rng = np.random.default_rng(seed)
    a_len, t_len = rng.integers(5, 12, n), rng.integers(3, 8, n)
    emotion = rng.integers(0, 4, n)
    meta = Meta(
        row=np.arange(n, dtype=np.int64),
        utt_id=[f"Ses0{1 + i % 5}M_impro01_F{i:03d}" for i in range(n)],
        session=(np.arange(n) % 5 + 1).astype(np.int64),
        dialogue=(np.arange(n) // 4).astype(np.int64),
        gender=rng.integers(0, 2, n).astype(np.int64),
        emotion=emotion.astype(np.int64),
        vad=np.clip(rng.normal(3, 1, (n, 3)) + emotion[:, None] * 0.3, 1, 5),
        text=["hello there"] * n,
        seconds=rng.uniform(1, 5, n),
    )
    return Features(
        meta,
        rng.normal(size=(int(a_len.sum()), D)).astype(np.float16),
        a_len.astype(np.int64),
        rng.normal(size=(int(t_len.sum()), D)).astype(np.float16),
        t_len.astype(np.int64),
    )


def toy_cfg(**kw: object) -> Config:
    base = Config(
        d_model=32,
        n_heads=4,
        n_cross_layers=1,
        head_hidden=16,
        epochs=2,
        batch_size=32,
        n_folds=2,
        seeds=(0,),
        n_bootstrap=20,
        probe_size=16,
        patience=2,
        trunk="concat",
    )
    return replace(base, **kw)  # type: ignore[arg-type]


# ---- configuration ----------
def test_regimes_are_consistent() -> None:
    names = [r.name for r in REGIMES]
    assert len(names) == len(set(names))
    assert all(set(r.tasks) <= set(TASKS) for r in REGIMES)
    assert REGIME_BY_NAME["st_gender"].selection_task == "gender"
    assert REGIME_BY_NAME["mtl_vad"].selection_task == "emotion"
    assert not REGIME_BY_NAME["st_emotion"].is_multitask and REGIME_BY_NAME["mtl_equal"].is_multitask
    assert {r.balancing for r in REGIMES} == {"equal", "uncertainty", "pcgrad"}


def test_task_metadata() -> None:
    assert [task_dim(t) for t in TASKS] == [4, 1, 1, 1, 2]
    assert [is_regression(t) for t in TASKS] == [False, True, True, True, False]
    assert primary_metric("arousal") == "ccc" and primary_metric("emotion") == "ua"
    assert make_config("smoke").limit is not None and make_config("standard").limit is None
    with pytest.raises(ValueError):
        make_config("nope")


# ---- data ----------
def test_gender_comes_from_the_suffix_not_the_prefix() -> None:
    m = _UTT_RE.match("Ses02M_impro02_F003")
    assert m is not None
    assert (m["session"], m["gender"], m["dialogue"]) == ("02", "F", "Ses02M_impro02")
    m2 = _UTT_RE.match("Ses05F_script03_2_M011")
    assert m2 is not None and (m2["session"], m2["gender"], m2["dialogue"]) == ("05", "M", "Ses05F_script03_2")
    assert _UTT_RE.match("garbage") is None


def test_session_splits_are_speaker_disjoint() -> None:
    session = np.repeat(np.arange(1, 6), 10)
    splits = session_splits(session, 5)
    assert sorted(int(session[sp.test[0]]) for sp in splits) == [1, 2, 3, 4, 5]
    for sp in splits:
        parts = [set(sp.train), set(sp.val), set(sp.test)]
        assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        assert len(parts[0] | parts[1] | parts[2]) == 50


# ---- targets, losses, metrics ----------
def test_standardizer_roundtrip() -> None:
    vad = torch.rand(50, 3) * 4 + 1
    std = Standardizer.fit(vad)
    z = std.encode(vad)
    assert torch.allclose(z.mean(0), torch.zeros(3), atol=1e-5) and torch.allclose(z.std(0), torch.ones(3), atol=1e-5)
    assert torch.allclose(std.decode(z), vad, atol=1e-5)


def test_ccc_properties() -> None:
    y = np.linspace(1, 5, 50)
    assert ccc(y, y) == pytest.approx(1.0)
    assert ccc(y, y + 1.0) < 0.9  # a constant shift is penalised although the correlation is perfect
    assert ccc(y, np.full_like(y, y.mean())) == pytest.approx(0.0, abs=1e-6)
    assert task_scores("valence", y, y)["pearson"] == pytest.approx(1.0)


def test_class_scores_ignore_absent_classes() -> None:
    y = np.array([0] * 8 + [1] * 2)
    s = class_scores(y, np.zeros_like(y), 4)
    assert s["wa"] == pytest.approx(0.8) and s["ua"] == pytest.approx(0.5)


def _batch(b: int = 6, ta: int = 9, tt: int = 5) -> Batch:
    g = torch.Generator().manual_seed(0)
    am = torch.ones(b, ta, dtype=torch.bool)
    am[0, 5:] = False
    tm = torch.ones(b, tt, dtype=torch.bool)
    tm[1, 3:] = False
    return Batch(
        torch.randn(b, ta, D, generator=g),
        am,
        torch.randn(b, tt, D, generator=g),
        tm,
        torch.randint(0, 4, (b,), generator=g),
        torch.rand(b, 3, generator=g) * 4 + 1,
        torch.randint(0, 2, (b,), generator=g),
    )


def test_task_losses_cover_every_task_and_are_finite() -> None:
    net = MTLNet(TASKS, toy_cfg(), D)
    b = _batch()
    std = Standardizer.fit(b.vad)
    losses = task_losses(net(b), b, std, emotion_weight=torch.ones(4))
    assert set(losses) == set(TASKS) and all(torch.isfinite(v) for v in losses.values())
    decoded = decode_outputs(net(b), std)
    assert decoded["emotion"].shape == (6, 4) and torch.allclose(decoded["emotion"].sum(1), torch.ones(6))
    assert decoded["valence"].shape == (6,)


# ---- model ----------
@pytest.mark.parametrize("trunk", ["concat", "cross", "bicross"])
def test_trunk_shapes(trunk: str) -> None:
    t = Trunk(trunk, toy_cfg(trunk=trunk), D)
    z = t(_batch())
    assert z.shape == (6, t.z_dim) and t.z_dim == (32 if trunk == "cross" else 64)


def test_mtlnet_heads_and_shared_parameters() -> None:
    net = MTLNet(("emotion", "gender"), toy_cfg(), D)
    out = net(_batch())
    assert out["emotion"].shape == (6, 4) and out["gender"].shape == (6, 2)
    shared = {id(p) for p in net.shared_parameters()}
    head = {id(p) for h in net.heads.values() for p in h.parameters()}
    assert shared.isdisjoint(head) and len(shared) + len(head) == len(list(net.parameters()))


def test_trunk_ignores_padded_audio() -> None:
    t = Trunk("bicross", toy_cfg(trunk="bicross"), D).eval()
    b = _batch()
    audio = b.audio.clone()
    audio[0, 5:] = 100.0  # padded frames of sample 0
    out2 = t(Batch(audio, b.audio_mask, b.text, b.text_mask, b.emotion, b.vad, b.gender))
    assert torch.allclose(t(b), out2, atol=1e-5)


# ---- loss balancing ----------
def test_fixed_weights() -> None:
    assert fixed_weights(("emotion", "gender", "valence"), 0.3) == {"emotion": 1.0, "gender": 0.3, "valence": 0.3}
    assert fixed_weights(("gender",), 0.3) == {"gender": 1.0}


def test_uncertainty_weighting_starts_at_half_and_learns() -> None:
    uw = UncertaintyWeighting(("emotion", "gender"))
    assert uw.weights() == {"emotion": 0.5, "gender": 0.5}
    loss = uw({"emotion": torch.tensor(2.0), "gender": torch.tensor(1.0)})
    assert float(loss.detach()) == pytest.approx(1.5)  # 0.5*2 + 0.5*1
    loss.backward()
    # d/ds = -0.5 * exp(-s) * L + 0.5: -0.5 for L=2, and exactly 0 for L=1 (the equilibrium e^{-s} L = 1)
    assert uw.log_var.grad is not None
    assert torch.allclose(uw.log_var.grad, torch.tensor([-0.5, 0.0]))


def test_pcgrad_removes_the_conflicting_component() -> None:
    rng = np.random.default_rng(0)
    g = torch.tensor([[1.0, 0.0], [-1.0, 1.0]])
    merged = pcgrad_merge(g, rng)
    assert torch.allclose(merged, torch.tensor([0.5, 1.5]))  # g1' = [.5, .5], g2' = [0, 1]
    agree = torch.tensor([[1.0, 0.0], [1.0, 1.0]])  # no conflict -> plain sum
    assert torch.allclose(pcgrad_merge(agree, rng), torch.tensor([2.0, 1.0]))


def test_cosine_matrix_of_task_gradients() -> None:
    net = MTLNet(TASKS, toy_cfg(), D)
    b = _batch()
    losses = task_losses(net(b), b, Standardizer.fit(b.vad), None)
    grads = shared_task_grads(net.shared_parameters(), losses, TASKS)
    assert grads.shape[0] == len(TASKS)
    cos = cosine_matrix(grads)
    assert np.allclose(np.diag(cos), 1.0, atol=1e-5) and np.allclose(cos, cos.T, atol=1e-5)


# ---- training and the whole pipeline ----------
@pytest.mark.parametrize("name", ["st_emotion", "st_valence", "mtl_equal", "mtl_uncertainty", "mtl_pcgrad"])
def test_train_regime_runs_and_predicts(name: str) -> None:
    feats = toy_features()
    cfg = toy_cfg()
    ds = Dataset(feats)
    sp = session_splits(feats.meta.session, 2)[0]
    std = Standardizer.fit(ds.vad[torch.as_tensor(sp.train)])
    regime = REGIME_BY_NAME[name]
    model, hist = train_regime(regime, cfg, ds, sp, std, 0, torch.device("cpu"), D)
    assert len(hist.val_score) == len(hist.train_loss) >= 1 and all(np.isfinite(hist.train_loss))
    out = predict(model, ds, sp.test, torch.device("cpu"), std)
    assert set(out) == set(regime.tasks)
    for t in regime.tasks:
        assert out[t].shape == ((len(sp.test),) if is_regression(t) else (len(sp.test), task_dim(t)))
        assert np.isfinite(out[t]).all()
    assert bool(hist.cosines) == regime.is_multitask
    assert bool(hist.uncertainty_weights) == (regime.balancing == "uncertainty")
    assert bool(hist.pcgrad_conflict) == (regime.balancing == "pcgrad")


def test_experiment_report_and_eda_end_to_end(tmp_path: Path) -> None:
    feats = toy_features(n=100)
    cfg = toy_cfg(out_dir=tmp_path)
    res = run_experiment(cfg, feats, torch.device("cpu"))
    assert set(res.regimes) == {r.name for r in REGIMES}
    assert res.preds[("mtl_equal", "emotion")].shape == (1, 100, 4)
    assert res.preds[("mtl_equal", "valence")].shape == (1, 100)
    assert 0.0 <= metric(res, "mtl_equal", "emotion").mean <= 1.0
    d = bootstrap_diff(res, "mtl_equal", "st_emotion", "emotion", cfg)
    assert d.lo <= d.diff <= d.hi or abs(d.diff - d.lo) < 0.2  # interval is around the estimate (tiny sample)
    assert mean_cosines(res, "mtl_equal") is not None and conflict_share(res, "st_emotion") is None
    html = build_report(cfg, feats.meta, res).read_text(encoding="utf-8")
    assert "Многозадачное обучение" in html and "PCGrad" in html and (tmp_path / "metrics.json").exists()
    assert build_eda(cfg, feats.meta).exists()


def test_bootstrap_detects_a_clear_difference() -> None:
    from mtllab.experiment import Results

    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 4, n)
    good = np.eye(4)[y] * 0.7 + 0.075
    bad = rng.dirichlet(np.ones(4), n)
    res = Results(
        emotion=y.astype(np.int64),
        gender=np.zeros(n, dtype=np.int64),
        vad=np.zeros((n, 3)),
        session=np.ones(n, dtype=np.int64),
        dialogue=(np.arange(n) // 5).astype(np.int64),
        covered=np.ones(n, dtype=bool),
        regimes=("a", "b"),
        seeds=(0,),
        preds={("a", "emotion"): good[None], ("b", "emotion"): bad[None]},
        fold_scores={},
        histories={},
        shapes=[],
        n_params={},
        seconds=0.0,
    )
    d = bootstrap_diff(res, "a", "b", "emotion", Config(n_bootstrap=100))
    assert d.lo > 0.4 and d.p_not_better == 0.0
