# Ad-creative transfer and out-of-distribution check

Model `clip_ridge` was trained on LaMem natural photos, then run on 1500 real advertising images (Hugging Face mirror of the Pitt Image Ads dataset) with no retraining. There are no human response labels for these ads, so this is not an accuracy claim. It tests whether the trust layer behaves correctly on a new domain: does the model recognize creative unlike anything it was trained on.

## Does the model know these are unfamiliar
- Distance to the training distribution (median): ad creative **1.16** vs LaMem photos 1.00.
- Share of ads beyond the 95th percentile of in-distribution images: **70%** (5% would be chance).

A clearly higher ad distance and a flagged share well above 5% mean the model correctly treats ad creative as out-of-distribution. That is the safety property a creative-testing product needs: the system should know when it is being asked about creative unlike its training data and defer rather than guess. See `ads/ood_hist.png`.

## What it predicts on ads (context, not accuracy)
- Median predicted memorability: ads 0.818 vs LaMem test 0.762.
- Median predicted std: ads 0.051 vs LaMem test 0.051.

## Example ads (most and least out-of-distribution)
| image        |   ood |   pred_memorability |   std |
|:-------------|------:|--------------------:|------:|
| ad_01258.jpg | 1.383 |               0.676 | 0.071 |
| ad_00562.jpg | 1.345 |               0.815 | 0.04  |
| ad_01337.jpg | 1.345 |               0.888 | 0.094 |
| ad_00836.jpg | 1.345 |               0.74  | 0.147 |
| ad_00177.jpg | 1.341 |               0.939 | 0.049 |
| ad_00293.jpg | 1.341 |               0.875 | 0.043 |
| ad_00066.jpg | 0.909 |               0.875 | 0.052 |
| ad_01338.jpg | 0.879 |               0.7   | 0.059 |
| ad_00236.jpg | 0.836 |               0.806 | 0.048 |

These images are saved in `output/ads/`. Check that the most-out-of-distribution ones are genuinely unusual creative (heavy text, logos, collages) rather than noise, which is the qualitative confirmation the score is meaningful.

## Honest scope
No response or memorability ground truth exists for these ads, so this shows domain transfer and out-of-distribution behaviour, not predictive accuracy on ads. Validating accuracy would need a labeled ad-response set, which is exactly the kind of data a brain-response product collects.
