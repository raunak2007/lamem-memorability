# Content-level failure map: where is the model least trustworthy, by creative type?

A single accuracy number hides where a model can and cannot be trusted, and for a creative-testing product that map is the deliverable. Each of the 5000 held-out LaMem test images was tagged with a content category by CLIP zero-shot, then rank accuracy, error, conformal coverage, and prediction bias were measured within each category.

## Reliability by creative type (least trustworthy first)
| category       |    n |   mae |   spearman |   cov_at_0.90 |   bias_pred_minus_actual |
|:---------------|-----:|------:|-----------:|--------------:|-------------------------:|
| outdoor/nature |  446 | 0.084 |      0.52  |         0.915 |                    0.014 |
| indoor scene   | 1210 | 0.077 |      0.598 |         0.908 |                   -0.001 |
| art/graphic    |  259 | 0.075 |      0.483 |         0.88  |                    0.001 |
| food           |  202 | 0.072 |      0.48  |         0.931 |                    0     |
| vehicle        |  573 | 0.071 |      0.62  |         0.923 |                    0.008 |
| animal         |  706 | 0.07  |      0.54  |         0.936 |                    0.001 |
| object/product |  670 | 0.067 |      0.575 |         0.919 |                    0.002 |
| people         |  604 | 0.066 |      0.578 |         0.922 |                    0.005 |
| text/document  |  110 | 0.063 |      0.525 |         0.909 |                   -0.005 |
| face/portrait  |  220 | 0.059 |      0.615 |         0.927 |                    0.004 |

Overall MAE is 0.072. The model is least trustworthy on **outdoor/nature** (MAE 0.084) and most trustworthy on **face/portrait** (MAE 0.059). A positive `bias_pred_minus_actual` means the model over-predicts memorability for that type and a negative one means it under-predicts. See `taxonomy/taxonomy.png`, where red bars are over-predicted types and blue are under-predicted.

## What this means for Presonance
This is the actionable trust map: auto-score the creative types where the model is reliable and well-calibrated, and route the types where it is not to human or neuro testing. It also exposes systematic bias, the categories the model consistently over- or under-rates, which a product corrects per segment rather than trusting one global number.

## Honest scope
CLIP zero-shot tags are approximate, each image gets a single best-matching category, and small categories are noisy, so this is a directional reliability map rather than a precise per-type audit. The categories reflect LaMem's natural-photo content, not ad creative; the same method would run on a creative library with the product's own taxonomy.
