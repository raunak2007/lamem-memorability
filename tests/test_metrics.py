"""Tests for the eval + calibration harness (the part a reviewer will read)."""
from __future__ import annotations

import numpy as np

from memorability import data, metrics
from run import run_eval


def test_spearman_is_rank_based():
    # a monotonic but nonlinear transform keeps rank perfect
    actual = np.linspace(0, 1, 50)
    pred = actual ** 2
    m = metrics.regression_metrics(pred, actual)
    assert m["spearman"] > 0.999


def test_reliability_curve_bins_cover_all_points():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 1, 200)
    actual = pred + rng.normal(0, 0.05, 200)
    rel = metrics.reliability_curve(pred, actual, n_bins=10)
    assert rel["n"].sum() == 200
    # a near-calibrated model has small gaps
    assert rel["gap"].abs().mean() < 0.1


def test_interval_coverage_matches_nominal_for_true_gaussian():
    rng = np.random.default_rng(1)
    n = 20000
    truth = rng.normal(0.5, 0.1, n)
    pred = np.full(n, 0.5)
    std = np.full(n, 0.1)
    cov = metrics.interval_coverage(pred, std, truth, nominal=(0.9,))
    emp = float(cov.loc[0, "empirical_coverage"])
    assert abs(emp - 0.9) < 0.02  # well-specified intervals cover at the nominal rate


def test_failure_cases_rank_by_abs_error():
    paths = [f"img_{i}.png" for i in range(5)]
    pred = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    actual = np.array([0.1, 0.5, 0.1, 0.2, 0.8])  # index 2 is the big miss
    fails = metrics.failure_cases(paths, pred, actual, k=1)
    assert fails["worst"].iloc[0]["image"] == "img_2.png"
    assert fails["best"].iloc[0]["abs_err"] == 0.0


def test_auroc_perfect_reversed_and_chance():
    score = np.array([0.1, 0.2, 0.3, 0.9, 1.0])
    labels = np.array([0, 0, 0, 1, 1])
    assert abs(metrics.auroc(score, labels) - 1.0) < 1e-9       # score ranks positives top
    assert abs(metrics.auroc(-score, labels) - 0.0) < 1e-9      # reversed
    assert abs(metrics.auroc(np.ones(5), labels) - 0.5) < 1e-9  # all ties -> chance
    assert np.isnan(metrics.auroc(score, np.zeros(5)))          # one class -> undefined


def test_selective_curve_confident_slice_has_lower_error():
    score = np.linspace(0, 1, 100)   # gate
    pred = score.copy()
    actual = np.zeros(100)           # abs error == score, monotone in the gate
    curve = metrics.selective_curve(score, pred, actual, fractions=(1.0, 0.5, 0.3))
    maes = curve.set_index("coverage")["mae"]
    assert maes.loc[0.3] < maes.loc[0.5] < maes.loc[1.0]


def test_selective_curve_tracks_shifted_fraction():
    score = np.concatenate([np.full(80, 0.1), np.full(20, 0.9)])  # shifted scored higher
    shifted = np.concatenate([np.zeros(80), np.ones(20)])
    pred = np.zeros(100)
    actual = np.zeros(100)
    curve = metrics.selective_curve(score, pred, actual, fractions=(1.0, 0.8), shifted=shifted)
    row_all = curve[curve.coverage == 1.0].iloc[0]
    row_80 = curve[curve.coverage == 0.8].iloc[0]
    assert abs(row_all["shifted_frac"] - 0.20) < 1e-9  # whole pool is 20% shifted
    assert row_80["shifted_frac"] < 1e-9               # dropping the top 20% purges shifted


def test_residuals_by_group_orders_worst_first():
    groups = ["a", "a", "b", "b"]
    pred = np.array([0.5, 0.5, 0.9, 0.9])
    actual = np.array([0.5, 0.5, 0.1, 0.1])  # group b is badly predicted
    out = metrics.residuals_by_group(groups, pred, actual)
    assert out.index[0] == "b"


def test_pairwise_decision_scores_calls_and_drops_true_ties():
    pred = np.array([0.9, 0.1, 0.2, 0.8, 0.5, 0.5])
    actual = np.array([0.8, 0.2, 0.7, 0.3, 0.5, 0.5])
    i = np.array([0, 2, 4])  # pairs (0,1) right pick, (2,3) wrong pick, (4,5) true tie
    j = np.array([1, 3, 5])
    correct, margin, keep = metrics.pairwise_decision(pred, actual, i, j)
    assert keep.tolist() == [True, True, False]  # the true-tie pair is dropped
    assert correct.tolist() == [True, False]     # higher pred correct, then wrong
    assert abs(margin[0] - 0.8) < 1e-9           # |0.9 - 0.1|


def test_accuracy_by_confidence_rises_with_confidence():
    rng = np.random.default_rng(0)
    n = 4000
    confidence = rng.uniform(0, 1, n)
    correct = rng.uniform(0, 1, n) < confidence  # P(correct) grows with confidence
    bc = metrics.accuracy_by_confidence(correct, confidence, n_bins=10)
    assert bc["n"].sum() == n
    assert bc["accuracy"].iloc[-1] > bc["accuracy"].iloc[0]


def test_decision_coverage_curve_confident_slice_is_more_accurate():
    confidence = np.linspace(0, 1, 100)
    correct = confidence > 0.5  # correctness perfectly ordered by confidence
    curve = metrics.decision_coverage_curve(correct, confidence, fractions=(1.0, 0.5))
    acc = curve.set_index("auto_fraction")["accuracy"]
    assert abs(acc.loc[0.5] - 1.0) < 1e-9  # the most-confident half is all correct
    assert abs(acc.loc[1.0] - 0.5) < 1e-9  # the whole pool is half correct
    panel = curve.set_index("auto_fraction")["panel_fraction"]
    assert abs(panel.loc[0.5] - 0.5) < 1e-9  # panel fraction complements auto fraction


def test_batch_selection_beats_random_pick():
    rng = np.random.default_rng(0)
    actual = rng.uniform(0, 1, 1000)
    pred = actual + rng.normal(0, 0.05, 1000)  # strong but imperfect selector
    out = metrics.batch_selection(pred, actual, batch_size=5, n_batches=5000, seed=0)
    assert out["random_p_best"] == 0.2
    assert out["p_pick_is_best"] > 0.5  # a good model lands the true best far above chance


def test_fixture_pipeline_runs_and_brightness_beats_mean(tmp_path):
    # the brightness baseline should order the fixture better than the mean baseline
    bright = run_eval(data.FIXTURE_ROOT, "train_tiny", "test_tiny", "brightness", str(tmp_path / "b"))
    mean = run_eval(data.FIXTURE_ROOT, "train_tiny", "test_tiny", "mean", str(tmp_path / "m"))
    bright_rho = bright["metrics"]["spearman"]
    mean_rho = mean["metrics"]["spearman"]
    # brightness carries real rank signal; the constant mean baseline has none (nan)
    assert bright_rho > 0.3
    assert np.isnan(mean_rho) or bright_rho > mean_rho
    assert (tmp_path / "b" / "findings.md").exists()
