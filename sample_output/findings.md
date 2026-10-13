# Memorability model: eval and trust report

Model: `brightness` · test items: 12

## Rank accuracy (does it order creative correctly)
- Spearman rho: **0.706** (p=0.01)
- MAE: 0.033 · RMSE: 0.046

Spearman is the metric that matters for creative testing: can the model say which asset is more memorable, not just guess an absolute number.

## Calibration (can you trust the predicted scores)
|   bin |   n |   mean_pred |   mean_actual |    gap |
|------:|----:|------------:|--------------:|-------:|
|     0 |   3 |       0.441 |         0.451 | -0.01  |
|     1 |   3 |       0.592 |         0.616 | -0.024 |
|     2 |   3 |       0.63  |         0.607 |  0.023 |
|     3 |   3 |       0.679 |         0.646 |  0.033 |

A calibrated model sits on the diagonal (mean actual = mean predicted per bin). A persistent `gap` means the model is systematically over- or under-confident in that score range. See `calibration.png`.

## Uncertainty coverage (if the model reports it)
|   nominal |   empirical_coverage |
|----------:|---------------------:|
|      0.5  |                0.667 |
|      0.8  |                0.833 |
|      0.9  |                0.833 |
|      0.95 |                0.833 |

Empirical coverage below nominal means the model's confidence intervals are too tight, i.e. it is overconfident.

## Where it breaks (largest errors)
| image        |   pred |   actual |   abs_err |
|:-------------|-------:|---------:|----------:|
| img_0037.png |  0.729 |    0.626 |     0.103 |
| img_0040.png |  0.572 |    0.664 |     0.092 |
| img_0025.png |  0.62  |    0.567 |     0.053 |
| img_0022.png |  0.602 |    0.561 |     0.04  |
| img_0044.png |  0.361 |    0.386 |     0.026 |
| img_0018.png |  0.601 |    0.622 |     0.021 |
| img_0019.png |  0.632 |    0.613 |     0.019 |
| img_0039.png |  0.655 |    0.673 |     0.017 |
| img_0031.png |  0.651 |    0.639 |     0.012 |
| img_0029.png |  0.45  |    0.46  |     0.01  |

These are the items to inspect for a failure taxonomy. Tag the test set with a category and call `metrics.residuals_by_group` to see which kinds of creative the model cannot be trusted on. For a brain-response product, that map of where-not-to-trust-it is the deliverable, not the single accuracy number.

## Scope
Public data, method illustration. Run against real LaMem for real numbers; the fixture scores are synthetic (brightness-driven) so the harness runs offline.
