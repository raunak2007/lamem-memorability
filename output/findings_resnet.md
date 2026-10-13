# Memorability model: eval and trust report

Model: `resnet_ridge` · test items: 10000

## Rank accuracy (does it order creative correctly)
- Spearman rho: **0.624** (p=0)
- MAE: 0.076 · RMSE: 0.096

Spearman is the metric that matters for creative testing: can the model say which asset is more memorable, not just guess an absolute number.

## Calibration (can you trust the predicted scores)
|   bin |    n |   mean_pred |   mean_actual |    gap |
|------:|-----:|------------:|--------------:|-------:|
|     0 | 1000 |       0.613 |         0.6   |  0.013 |
|     1 | 1000 |       0.667 |         0.667 |  0     |
|     2 | 1000 |       0.699 |         0.698 |  0.001 |
|     3 | 1000 |       0.728 |         0.733 | -0.005 |
|     4 | 1000 |       0.752 |         0.749 |  0.003 |
|     5 | 1000 |       0.774 |         0.78  | -0.006 |
|     6 | 1000 |       0.794 |         0.797 | -0.003 |
|     7 | 1000 |       0.815 |         0.817 | -0.002 |
|     8 | 1000 |       0.84  |         0.836 |  0.004 |
|     9 | 1000 |       0.883 |         0.866 |  0.017 |

A calibrated model sits on the diagonal (mean actual = mean predicted per bin). A persistent `gap` means the model is systematically over- or under-confident in that score range. See `calibration.png`.

## Uncertainty coverage (if the model reports it)
|   nominal |   empirical_coverage |
|----------:|---------------------:|
|      0.5  |                0.316 |
|      0.8  |                0.552 |
|      0.9  |                0.673 |
|      0.95 |                0.762 |

Empirical coverage below nominal means the model's confidence intervals are too tight, i.e. it is overconfident.

**Does the uncertainty track error (the trust-layer test)**

- Spearman(std, |error|): **0.129** — higher means the model flags the items it actually gets wrong.
- MAE on its confident half: **0.068**, overall 0.076, unconfident half 0.083.

If error on the confident half is well below overall, the model knows when to be trusted: you act on its confident calls and route the rest to a human. That selective-trust behaviour is the product, not the single accuracy number.

**After split-conformal calibration** (held-out, distribution-free)

|   nominal |   conformal_coverage |
|----------:|---------------------:|
|      0.5  |                0.504 |
|      0.8  |                0.802 |
|      0.9  |                0.901 |
|      0.95 |                0.952 |

Conformal scales the interval widths using held-out normalized residuals, so empirical coverage meets the nominal rate with a finite-sample guarantee. This is the fix for the raw intervals being too tight above: same per-item std ordering, honest widths.

**Selective prediction** (auto-score the confident slice, route the rest)

|   coverage |     n |   mae |   rmse |
|-----------:|------:|------:|-------:|
|        1   | 10000 | 0.076 |  0.096 |
|        0.9 |  9000 | 0.075 |  0.094 |
|        0.8 |  8000 | 0.073 |  0.092 |
|        0.7 |  7000 | 0.072 |  0.09  |
|        0.6 |  6000 | 0.07  |  0.088 |
|        0.5 |  5000 | 0.068 |  0.086 |
|        0.4 |  4000 | 0.067 |  0.084 |
|        0.3 |  3000 | 0.066 |  0.083 |

`coverage` is the fraction you auto-accept, taking the model's most-confident items first; `mae` is the error on just that slice. Error falling as coverage drops means the uncertainty is usable for triage: you auto-score the confident creative and send only the uncertain rest to expensive human or neuro testing. That is the product economics, not the single accuracy number. See `risk_coverage.png`.

**Coverage by predicted-score group** (does the guarantee hold per segment)

|   pred_group |    n |   mean_pred |   cov_at_0.90 |   mae |
|-------------:|-----:|------------:|--------------:|------:|
|            0 | 2500 |       0.65  |         0.894 | 0.087 |
|            1 | 2500 |       0.734 |         0.896 | 0.081 |
|            2 | 2500 |       0.789 |         0.906 | 0.072 |
|            3 | 2500 |       0.853 |         0.906 | 0.063 |

Marginal conformal guarantees about 90% coverage on average. If `cov_at_0.90` is uneven across groups, the honest next step is group-conditional (Mondrian) conformal, which guarantees coverage within each creative segment rather than only overall.

## Where it breaks (largest errors)
| image        |   pred |   actual |   abs_err |
|:-------------|-------:|---------:|----------:|
| 00036776.jpg |  0.657 |    0.229 |     0.428 |
| 00036267.jpg |  0.695 |    0.286 |     0.409 |
| 00040054.jpg |  0.75  |    0.351 |     0.399 |
| 00013281.jpg |  0.704 |    0.306 |     0.398 |
| 00055878.jpg |  0.698 |    0.308 |     0.39  |
| 00055839.jpg |  0.732 |    0.349 |     0.383 |
| 00020288.jpg |  0.759 |    0.378 |     0.38  |
| 00026939.jpg |  0.673 |    0.294 |     0.379 |
| 00001177.jpg |  0.628 |    0.25  |     0.378 |
| 00046823.jpg |  0.788 |    0.417 |     0.371 |

These are the items to inspect for a failure taxonomy. Tag the test set with a category and call `metrics.residuals_by_group` to see which kinds of creative the model cannot be trusted on. For a brain-response product, that map of where-not-to-trust-it is the deliverable, not the single accuracy number.

## Scope
Public data, method illustration. Run against real LaMem for real numbers; the fixture scores are synthetic (brightness-driven) so the harness runs offline.
