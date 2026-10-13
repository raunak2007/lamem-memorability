"""Evaluation and calibration metrics for memorability prediction.

Rank accuracy answers "does the model order creative correctly" (Spearman, the
LaMem paper's headline metric). The rest is the trust layer that makes a
prediction actually usable: reliability (are the predicted scores calibrated),
interval coverage (if the model reports uncertainty, can you believe its
confidence), and failure analysis (where does it break, and on what kind of
creative). A response predictor is only useful if you know when to trust it.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


def regression_metrics(pred, actual) -> dict:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    # a constant predictor has no rank information; spearman is undefined there
    if np.std(pred) == 0 or np.std(actual) == 0:
        rho, pval = float("nan"), float("nan")
    else:
        rho, pval = spearmanr(pred, actual)
    err = pred - actual
    return {
        "n": int(pred.size),
        "spearman": float(rho),
        "spearman_p": float(pval),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def reliability_curve(pred, actual, n_bins: int = 10) -> pd.DataFrame:
    """Bin by predicted score; a calibrated model has mean_actual ~ mean_pred per bin."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    bin_idx = np.clip(np.digitize(pred, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "n": int(mask.sum()),
                "mean_pred": float(pred[mask].mean()),
                "mean_actual": float(actual[mask].mean()),
                "gap": float(pred[mask].mean() - actual[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def interval_coverage(pred, std, actual, nominal: Sequence[float] = (0.5, 0.8, 0.9, 0.95)) -> pd.DataFrame:
    """For models that report a per-item std: does the nominal% interval actually
    contain the truth that often? Under-coverage means the model is overconfident."""
    pred = np.asarray(pred, dtype=float)
    std = np.asarray(std, dtype=float)
    actual = np.asarray(actual, dtype=float)
    rows = []
    for q in nominal:
        z = norm.ppf(0.5 + q / 2.0)
        lo, hi = pred - z * std, pred + z * std
        rows.append(
            {
                "nominal": float(q),
                "empirical_coverage": float(np.mean((actual >= lo) & (actual <= hi))),
            }
        )
    return pd.DataFrame(rows)


def conformal_coverage(pred, std, actual, quantiles) -> pd.DataFrame:
    """Empirical coverage of split-conformal intervals pred +/- q*std, where q is
    the conformal quantile per target level. Should sit at or above nominal."""
    pred = np.asarray(pred, dtype=float)
    std = np.asarray(std, dtype=float)
    actual = np.asarray(actual, dtype=float)
    rows = []
    for cov, q in quantiles.items():
        lo, hi = pred - q * std, pred + q * std
        rows.append(
            {
                "nominal": float(cov),
                "conformal_coverage": float(np.mean((actual >= lo) & (actual <= hi))),
            }
        )
    return pd.DataFrame(rows)


def risk_coverage(pred, std, actual, fractions=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)) -> pd.DataFrame:
    """Selective prediction: accept only the most-confident `fraction` of items
    (smallest std) and report error on what's accepted. This is the deployment
    economics for a creative-testing product: auto-score the confident slice and
    route the rest to expensive human or neuro testing."""
    pred = np.asarray(pred, dtype=float)
    std = np.asarray(std, dtype=float)
    actual = np.asarray(actual, dtype=float)
    err = np.abs(pred - actual)
    order = np.argsort(std)  # most confident (smallest std) first
    rows = []
    for f in fractions:
        k = max(1, int(round(f * len(std))))
        idx = order[:k]
        rows.append(
            {
                "coverage": float(f),
                "n": int(k),
                "mae": float(err[idx].mean()),
                "rmse": float(np.sqrt(np.mean(err[idx] ** 2))),
            }
        )
    return pd.DataFrame(rows)


def auroc(score, labels) -> float:
    """Area under the ROC curve for `score` ranking the positive class
    (labels == 1) above the negative. Rank-based (Mann-Whitney U) with tie
    handling, so no sklearn dependency. Used to ask a sharp question: does the
    OOD distance separate shifted creative from in-distribution creative, and
    does the model's own confidence (std) fail to, which would show internal
    confidence cannot see domain shift while the OOD layer can."""
    score = np.asarray(score, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(score)  # average ranks for ties
    return float((r[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def selective_curve(score, pred, actual, fractions=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3), shifted=None) -> pd.DataFrame:
    """Triage by an arbitrary confidence `score` (lower = more confident): accept
    the most-confident `fraction`, report MAE on what's accepted. Unlike
    risk_coverage (which is hard-wired to std), the gate is whatever you pass, so
    you can compare gating by OOD distance vs by std vs random on one pool. If
    `shifted` (0/1 per item) is given, also report the share of the accepted slice
    that is from the shifted domain, so you can watch a good gate purge shifted
    creative from the auto-scored set."""
    score = np.asarray(score, dtype=float)
    err = np.abs(np.asarray(pred, dtype=float) - np.asarray(actual, dtype=float))
    order = np.argsort(score, kind="stable")  # most confident (lowest score) first
    rows = []
    for f in fractions:
        k = max(1, int(round(f * len(score))))
        idx = order[:k]
        row = {"coverage": float(f), "n": int(k), "mae": float(err[idx].mean())}
        if shifted is not None:
            row["shifted_frac"] = float(np.asarray(shifted, dtype=float)[idx].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_decision(pred, actual, i, j):
    """Creative A/B calls: for each pair (i, j) the model picks the higher-predicted
    item. Returns (correct, margin, keep): `correct[k]` is True when the pick is the
    truly more memorable of the pair, `margin[k]` is the predicted score gap the call
    rested on (the model's confidence in that call), and `keep` is the boolean mask of
    input pairs that were scored. Pairs tied in the true score are dropped, since
    "more memorable" is undefined there, so index per-item arrays with `i[keep]` /
    `j[keep]` to stay aligned with `correct` and `margin`."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    i = np.asarray(i)
    j = np.asarray(j)
    keep = actual[i] != actual[j]
    ii, jj = i[keep], j[keep]
    pick_i = pred[ii] >= pred[jj]
    true_i = actual[ii] > actual[jj]
    correct = pick_i == true_i
    margin = np.abs(pred[ii] - pred[jj])
    return correct, margin, keep


def accuracy_by_confidence(correct, confidence, n_bins: int = 10) -> pd.DataFrame:
    """Bin decisions into equal-count bins by `confidence` (e.g. the predicted
    margin) and report accuracy in each. A usable confidence signal is monotonic:
    the higher the confidence, the higher the decision accuracy."""
    correct = np.asarray(correct, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    edges = np.quantile(confidence, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    b = np.clip(np.digitize(confidence, edges) - 1, 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        if not m.any():
            continue
        rows.append(
            {
                "conf_bin": k,
                "n": int(m.sum()),
                "mean_confidence": float(confidence[m].mean()),
                "accuracy": float(correct[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def decision_coverage_curve(
    correct, confidence, fractions=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
) -> pd.DataFrame:
    """Selective A/B decisions: auto-handle the most-confident `fraction` of calls
    (largest `confidence`) and route the rest to a panel. Report accuracy on the
    auto-handled slice. This is the test-budget tradeoff for a creative-testing
    product: `auto_fraction` is the share of calls the model makes for free,
    `panel_fraction` is the expensive remainder, and `accuracy` is the quality of the
    automated calls. Pass the predicted margin to gate on decision confidence, or the
    negated pooled std to gate on per-item uncertainty, and compare the two."""
    correct = np.asarray(correct, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    order = np.argsort(-confidence, kind="stable")  # most confident first
    n = len(correct)
    rows = []
    for f in fractions:
        k = max(1, int(round(f * n)))
        idx = order[:k]
        rows.append(
            {
                "auto_fraction": float(f),
                "panel_fraction": float(1.0 - f),
                "n": int(k),
                "accuracy": float(correct[idx].mean()),
            }
        )
    return pd.DataFrame(rows)


def batch_selection(
    pred, actual, batch_size: int, n_batches: int, top_frac: float = 0.2, seed: int = 0
) -> dict:
    """Pick-the-winner simulation: draw a random batch of `batch_size` creatives, let
    the model pick the highest-predicted one, and measure how often that pick is the
    truly most-memorable in the batch and how often it lands in the true top
    `top_frac`, against the random-pick baseline. This is the 'choose the winning
    creative from a set' decision stated directly."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(pred)
    k_top = max(1, int(round(top_frac * batch_size)))
    hit_best = 0
    hit_top = 0
    for _ in range(n_batches):
        idx = rng.choice(n, size=batch_size, replace=False)
        pick = int(idx[np.argmax(pred[idx])])
        order = idx[np.argsort(-actual[idx])]
        if pick == int(order[0]):
            hit_best += 1
        if np.any(order[:k_top] == pick):
            hit_top += 1
    return {
        "batch_size": int(batch_size),
        "n_batches": int(n_batches),
        "p_pick_is_best": hit_best / n_batches,
        "p_pick_in_top": hit_top / n_batches,
        "random_p_best": 1.0 / batch_size,
        "random_p_in_top": k_top / batch_size,
        "top_frac": float(top_frac),
    }


def grouped_coverage(pred, std, actual, q, n_groups: int = 4) -> pd.DataFrame:
    """Group-conditional check: split by predicted-score quantile and ask whether
    the conformal interval (pred +/- q*std) still covers ~90% inside each group.
    Marginal coverage can hold on average yet fail for one segment of creative."""
    pred = np.asarray(pred, dtype=float)
    std = np.asarray(std, dtype=float)
    actual = np.asarray(actual, dtype=float)
    err = np.abs(pred - actual)
    covered = (actual >= pred - q * std) & (actual <= pred + q * std)
    edges = np.quantile(pred, np.linspace(0, 1, n_groups + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    gidx = np.clip(np.digitize(pred, edges) - 1, 0, n_groups - 1)
    rows = []
    for g in range(n_groups):
        m = gidx == g
        if not m.any():
            continue
        rows.append(
            {
                "pred_group": g,
                "n": int(m.sum()),
                "mean_pred": float(pred[m].mean()),
                "cov_at_0.90": float(covered[m].mean()),
                "mae": float(err[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def uncertainty_quality(pred, std, actual) -> dict:
    """Does the reported std actually track error? This is the trust-layer payoff:
    a usable confidence signal ranks high-error items as high-uncertainty, and
    dropping the least-confident predictions lowers the error on what remains."""
    pred = np.asarray(pred, dtype=float)
    std = np.asarray(std, dtype=float)
    actual = np.asarray(actual, dtype=float)
    err = np.abs(pred - actual)
    if np.std(std) == 0 or np.std(err) == 0:
        rho = float("nan")
    else:
        rho = float(spearmanr(std, err).correlation)
    order = np.argsort(std)  # most confident (smallest std) first
    half = max(1, len(std) // 2)
    return {
        "std_err_spearman": rho,
        "mae_all": float(err.mean()),
        "mae_confident_half": float(err[order[:half]].mean()),
        "mae_unconfident_half": float(err[order[-half:]].mean()),
    }


def failure_cases(image_paths: Sequence[str], pred, actual, k: int = 10) -> dict:
    """Largest and smallest absolute errors, to seed a failure taxonomy."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    resid = np.abs(pred - actual)
    order = np.argsort(-resid)

    def pack(indices) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "image": [os_basename(image_paths[i]) for i in indices],
                "pred": pred[indices],
                "actual": actual[indices],
                "abs_err": resid[indices],
            }
        )

    return {"worst": pack(order[:k]), "best": pack(order[::-1][:k])}


def residuals_by_group(groups: Sequence, pred, actual) -> pd.DataFrame:
    """MAE per group label. Add a category column to the test set and pass it here
    to see which kinds of creative the model cannot be trusted on."""
    df = pd.DataFrame(
        {"group": list(groups), "pred": np.asarray(pred, float), "actual": np.asarray(actual, float)}
    )
    df["abs_err"] = (df["pred"] - df["actual"]).abs()
    out = df.groupby("group")["abs_err"].agg(["mean", "count"]).rename(
        columns={"mean": "mae", "count": "n"}
    )
    return out.sort_values("mae", ascending=False)


def ood_distance(features, mu, sd) -> np.ndarray:
    """Per-item distance to the training feature distribution (diagonal
    Mahalanobis: RMS standardized deviation across feature dimensions). Higher
    means more out-of-distribution. In-distribution items sit near 1.0; creative
    unlike the training set scores higher. Used to show the model can flag
    creative it was never trained on."""
    features = np.asarray(features, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    z = np.clip((features - mu) / sd, -8.0, 8.0)  # cap near-constant dims so they don't blow up
    return np.sqrt(np.mean(z ** 2, axis=1))


def os_basename(path: str) -> str:
    import os

    return os.path.basename(path)
