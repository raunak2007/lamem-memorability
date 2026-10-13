"""Write findings.md and the calibration plot from a set of predictions."""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from memorability import metrics


def write_report(
    image_paths: Sequence[str],
    pred,
    actual,
    model_name: str,
    out_dir: str,
    std: Optional[Sequence[float]] = None,
    n_bins: int = 10,
    conformal: Optional[dict] = None,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)

    reg = metrics.regression_metrics(pred, actual)
    bins = max(2, min(n_bins, len(pred) // 3))
    rel = metrics.reliability_curve(pred, actual, n_bins=bins)
    fails = metrics.failure_cases(image_paths, pred, actual, k=min(10, len(pred)))
    cov = metrics.interval_coverage(pred, std, actual) if std is not None else None
    unc = metrics.uncertainty_quality(pred, std, actual) if std is not None else None
    conf = (
        metrics.conformal_coverage(pred, std, actual, conformal)
        if (std is not None and conformal)
        else None
    )
    risk = metrics.risk_coverage(pred, std, actual) if std is not None else None
    grp = (
        metrics.grouped_coverage(pred, std, actual, conformal[0.9])
        if (std is not None and conformal and 0.9 in conformal)
        else None
    )

    png = os.path.join(out_dir, "calibration.png")
    _plot(rel, png)
    risk_png = None
    if risk is not None:
        risk_png = os.path.join(out_dir, "risk_coverage.png")
        _plot_risk(risk, risk_png)
    md_path = os.path.join(out_dir, "findings.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(_markdown(model_name, reg, rel, cov, unc, conf, risk, grp, fails))

    return {
        "metrics": reg,
        "reliability": rel,
        "coverage": cov,
        "uncertainty": unc,
        "conformal": conf,
        "risk": risk,
        "grouped": grp,
        "failures": fails,
        "findings_md": md_path,
        "plot": png,
        "risk_plot": risk_png,
    }


def _plot(rel, png) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo = float(min(rel["mean_pred"].min(), rel["mean_actual"].min()))
    hi = float(max(rel["mean_pred"].max(), rel["mean_actual"].max()))
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=1, ls="--", label="perfectly calibrated")
    ax.plot(rel["mean_pred"], rel["mean_actual"], "o-", color="#4C78A8", label="model")
    ax.set_xlabel("mean predicted memorability (per bin)")
    ax.set_ylabel("mean actual memorability (per bin)")
    ax.set_title("Calibration: predicted vs actual")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def _plot_risk(risk, png) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(risk["coverage"], risk["mae"], "o-", color="#E45756")
    ax.set_xlabel("coverage (fraction auto-accepted, most confident first)")
    ax.set_ylabel("MAE on accepted")
    ax.set_title("Selective prediction: error vs coverage")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def _markdown(model_name, reg, rel, cov, unc, conf, risk, grp, fails) -> str:
    out = []
    out.append("# Memorability model: eval and trust report\n")
    out.append(f"Model: `{model_name}` · test items: {reg['n']}\n")

    out.append("## Rank accuracy (does it order creative correctly)")
    out.append(
        f"- Spearman rho: **{reg['spearman']:.3f}** (p={reg['spearman_p']:.2g})\n"
        f"- MAE: {reg['mae']:.3f} · RMSE: {reg['rmse']:.3f}\n"
    )
    out.append(
        "Spearman is the metric that matters for creative testing: can the model say "
        "which asset is more memorable, not just guess an absolute number.\n"
    )

    out.append("## Calibration (can you trust the predicted scores)")
    out.append(rel.round(3).to_markdown(index=False))
    out.append(
        "\nA calibrated model sits on the diagonal (mean actual = mean predicted per "
        "bin). A persistent `gap` means the model is systematically over- or "
        "under-confident in that score range. See `calibration.png`.\n"
    )

    out.append("## Uncertainty coverage (if the model reports it)")
    if cov is not None:
        out.append(cov.round(3).to_markdown(index=False))
        out.append(
            "\nEmpirical coverage below nominal means the model's confidence intervals "
            "are too tight, i.e. it is overconfident.\n"
        )
        if unc is not None:
            out.append("**Does the uncertainty track error (the trust-layer test)**\n")
            out.append(
                f"- Spearman(std, |error|): **{unc['std_err_spearman']:.3f}** — "
                "higher means the model flags the items it actually gets wrong.\n"
                f"- MAE on its confident half: **{unc['mae_confident_half']:.3f}**, "
                f"overall {unc['mae_all']:.3f}, unconfident half "
                f"{unc['mae_unconfident_half']:.3f}.\n"
            )
            out.append(
                "If error on the confident half is well below overall, the model knows "
                "when to be trusted: you act on its confident calls and route the rest "
                "to a human. That selective-trust behaviour is the product, not the "
                "single accuracy number.\n"
            )
        if conf is not None:
            out.append("**After split-conformal calibration** (held-out, distribution-free)\n")
            out.append(conf.round(3).to_markdown(index=False))
            out.append(
                "\nConformal scales the interval widths using held-out normalized "
                "residuals, so empirical coverage meets the nominal rate with a finite-"
                "sample guarantee. This is the fix for the raw intervals being too tight "
                "above: same per-item std ordering, honest widths.\n"
            )
        if risk is not None:
            out.append("**Selective prediction** (auto-score the confident slice, route the rest)\n")
            out.append(risk.round(3).to_markdown(index=False))
            out.append(
                "\n`coverage` is the fraction you auto-accept, taking the model's "
                "most-confident items first; `mae` is the error on just that slice. Error "
                "falling as coverage drops means the uncertainty is usable for triage: you "
                "auto-score the confident creative and send only the uncertain rest to "
                "expensive human or neuro testing. That is the product economics, not the "
                "single accuracy number. See `risk_coverage.png`.\n"
            )
        if grp is not None:
            out.append("**Coverage by predicted-score group** (does the guarantee hold per segment)\n")
            out.append(grp.round(3).to_markdown(index=False))
            out.append(
                "\nMarginal conformal guarantees about 90% coverage on average. If "
                "`cov_at_0.90` is uneven across groups, the honest next step is "
                "group-conditional (Mondrian) conformal, which guarantees coverage within "
                "each creative segment rather than only overall.\n"
            )
    else:
        out.append(
            "_This model reports no per-item uncertainty. Implement "
            "`predict_with_uncertainty` to return a std and this section fills in._\n"
        )

    out.append("## Where it breaks (largest errors)")
    out.append(fails["worst"].round(3).to_markdown(index=False))
    out.append(
        "\nThese are the items to inspect for a failure taxonomy. Tag the test set "
        "with a category and call `metrics.residuals_by_group` to see which kinds of "
        "creative the model cannot be trusted on. For a brain-response product, that "
        "map of where-not-to-trust-it is the deliverable, not the single accuracy "
        "number.\n"
    )

    out.append("## Scope")
    out.append(
        "Public data, method illustration. Run against real LaMem for real numbers; "
        "the fixture scores are synthetic (brightness-driven) so the harness runs "
        "offline.\n"
    )
    return "\n".join(out)
