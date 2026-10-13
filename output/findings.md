# Memorability model: eval and trust report

Model: `clip_ridge` · test items: 10000

## Rank accuracy (does it order creative correctly)
- Spearman rho: **0.699** (p=0)
- MAE: 0.069 · RMSE: 0.088

Spearman is the metric that matters for creative testing: can the model say which asset is more memorable, not just guess an absolute number.

## Calibration (can you trust the predicted scores)
|   bin |    n |   mean_pred |   mean_actual |    gap |
|------:|-----:|------------:|--------------:|-------:|
|     0 | 1000 |       0.598 |         0.59  |  0.007 |
|     1 | 1000 |       0.657 |         0.653 |  0.004 |
|     2 | 1000 |       0.694 |         0.693 |  0.001 |
|     3 | 1000 |       0.725 |         0.722 |  0.003 |
|     4 | 1000 |       0.75  |         0.758 | -0.008 |
|     5 | 1000 |       0.773 |         0.775 | -0.002 |
|     6 | 1000 |       0.797 |         0.8   | -0.004 |
|     7 | 1000 |       0.821 |         0.823 | -0.002 |
|     8 | 1000 |       0.849 |         0.845 |  0.005 |
|     9 | 1000 |       0.899 |         0.884 |  0.015 |

A calibrated model sits on the diagonal (mean actual = mean predicted per bin). A persistent `gap` means the model is systematically over- or under-confident in that score range. See `calibration.png`.

## Uncertainty coverage (if the model reports it)
|   nominal |   empirical_coverage |
|----------:|---------------------:|
|      0.5  |                0.319 |
|      0.8  |                0.561 |
|      0.9  |                0.675 |
|      0.95 |                0.762 |

Empirical coverage below nominal means the model's confidence intervals are too tight, i.e. it is overconfident.

**Does the uncertainty track error (the trust-layer test)**

- Spearman(std, |error|): **0.140** — higher means the model flags the items it actually gets wrong.
- MAE on its confident half: **0.062**, overall 0.069, unconfident half 0.077.

If error on the confident half is well below overall, the model knows when to be trusted: you act on its confident calls and route the rest to a human. That selective-trust behaviour is the product, not the single accuracy number.

**After split-conformal calibration** (held-out, distribution-free)

|   nominal |   conformal_coverage |
|----------:|---------------------:|
|      0.5  |                0.504 |
|      0.8  |                0.805 |
|      0.9  |                0.905 |
|      0.95 |                0.953 |

Conformal scales the interval widths using held-out normalized residuals, so empirical coverage meets the nominal rate with a finite-sample guarantee. This is the fix for the raw intervals being too tight above: same per-item std ordering, honest widths.

**Selective prediction** (auto-score the confident slice, route the rest)

|   coverage |     n |   mae |   rmse |
|-----------:|------:|------:|-------:|
|        1   | 10000 | 0.069 |  0.088 |
|        0.9 |  9000 | 0.068 |  0.086 |
|        0.8 |  8000 | 0.067 |  0.084 |
|        0.7 |  7000 | 0.065 |  0.082 |
|        0.6 |  6000 | 0.064 |  0.08  |
|        0.5 |  5000 | 0.062 |  0.078 |
|        0.4 |  4000 | 0.06  |  0.076 |
|        0.3 |  3000 | 0.058 |  0.073 |

`coverage` is the fraction you auto-accept, taking the model's most-confident items first; `mae` is the error on just that slice. Error falling as coverage drops means the uncertainty is usable for triage: you auto-score the confident creative and send only the uncertain rest to expensive human or neuro testing. That is the product economics, not the single accuracy number. See `risk_coverage.png`.

**Coverage by predicted-score group** (does the guarantee hold per segment)

|   pred_group |    n |   mean_pred |   cov_at_0.90 |   mae |
|-------------:|-----:|------------:|--------------:|------:|
|            0 | 2500 |       0.639 |         0.899 | 0.081 |
|            1 | 2500 |       0.73  |         0.9   | 0.074 |
|            2 | 2500 |       0.791 |         0.902 | 0.068 |
|            3 | 2500 |       0.865 |         0.92  | 0.055 |

Marginal conformal guarantees about 90% coverage on average. If `cov_at_0.90` is uneven across groups, the honest next step is group-conditional (Mondrian) conformal, which guarantees coverage within each creative segment rather than only overall.

## Where it breaks (largest errors)
| image        |   pred |   actual |   abs_err |
|:-------------|-------:|---------:|----------:|
| 00013281.jpg |  0.725 |    0.306 |     0.419 |
| 00036267.jpg |  0.69  |    0.286 |     0.404 |
| 00026509.jpg |  0.88  |    0.478 |     0.402 |
| 00012506.jpg |  0.777 |    0.412 |     0.365 |
| 00026939.jpg |  0.658 |    0.294 |     0.364 |
| 00055839.jpg |  0.712 |    0.349 |     0.363 |
| 00036776.jpg |  0.591 |    0.229 |     0.362 |
| 00034367.jpg |  0.685 |    0.333 |     0.352 |
| 00049982.jpg |  0.747 |    0.405 |     0.341 |
| 00040054.jpg |  0.689 |    0.351 |     0.338 |

These are the items to inspect for a failure taxonomy. Tag the test set with a category and call `metrics.residuals_by_group` to see which kinds of creative the model cannot be trusted on. For a brain-response product, that map of where-not-to-trust-it is the deliverable, not the single accuracy number.

## Scope
Public data, method illustration. Run against real LaMem for real numbers; the fixture scores are synthetic (brightness-driven) so the harness runs offline.
