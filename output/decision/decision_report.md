# Creative selection under a testing budget

A creative-testing product rarely needs an image's absolute memorability score. It needs the right decision: which of these creatives is more memorable, and can I trust that call or should I pay to test it. This reframes the same `clip_ridge` model and trust layer as an A/B selection engine and measures it as one, on 5000 held-out LaMem images.

## The model is a far better chooser than scorer
- On 49742 random creative pairs, the model picks the more memorable one **74.2%** of the time (50% is a coin flip).
- Shown 5 candidate creatives, its top pick is the true best **50%** of the time (random is 20%).
- Shown 10 candidates, its top pick is the true best **36%** of the time (random is 10%) and lands in the true top 2 **58%** of the time (random is 20%).

The point estimate is a 0.699-Spearman regressor, strong but not exact. Expressed as a choice between creatives, that same signal is a reliable selector, because ranking only needs the sign of the gap, not its precise value.

## Accuracy rises with the model's confidence in the call
|   conf_bin |    n |   mean_confidence |   accuracy |
|-----------:|-----:|------------------:|-----------:|
|          0 | 4975 |             0.008 |      0.535 |
|          1 | 4974 |             0.024 |      0.566 |
|          2 | 4974 |             0.041 |      0.61  |
|          3 | 4974 |             0.058 |      0.661 |
|          4 | 4974 |             0.077 |      0.717 |
|          5 | 4974 |             0.098 |      0.77  |
|          6 | 4974 |             0.121 |      0.814 |
|          7 | 4974 |             0.148 |      0.868 |
|          8 | 4974 |             0.184 |      0.91  |
|          9 | 4975 |             0.256 |      0.97  |

`mean_confidence` is the predicted memorability gap the call rested on; `accuracy` is how often the pick was truly correct in that bin. The call is near coin-flip when the predicted gap is tiny (genuinely close creatives) and approaches certainty as the gap widens. That monotonic rise is what makes the confidence usable: the model knows which of its own calls to stand behind. See `decision/confidence_accuracy.png`.

## The test-budget tradeoff, and which confidence signal to gate on
Auto-handle the most-confident share of A/B calls and send the rest to a panel. Gating by the **predicted gap** (the decision's own margin) is compared against gating by **pooled per-item uncertainty** (the heteroscedastic std of the two items), the signal a naive system would reach for.

|   auto_fraction |   panel_fraction |   acc_gap_gate |   acc_std_gate |
|----------------:|-----------------:|---------------:|---------------:|
|             1   |              0   |          0.742 |          0.742 |
|             0.9 |              0.1 |          0.765 |          0.743 |
|             0.8 |              0.2 |          0.79  |          0.741 |
|             0.7 |              0.3 |          0.816 |          0.737 |
|             0.6 |              0.4 |          0.841 |          0.733 |
|             0.5 |              0.5 |          0.866 |          0.731 |
|             0.4 |              0.6 |          0.89  |          0.726 |
|             0.3 |              0.7 |          0.916 |          0.721 |
|             0.2 |              0.8 |          0.94  |          0.717 |

Gating by the decision margin is the clear winner. Automating the most-confident half of calls, the margin gate holds **86.6%** accuracy versus **73.1%** for the per-item std gate. This is the earlier finding put to use: the per-item variance head is a weak error-ranker, but the margin between two predictions is a strong decision-confidence signal. For an A/B product you gate on the margin, not the error bar. See `decision/budget_curve.png`.

## What this means for Presonance
This is the product rule stated in the units a creative team buys: the model makes the confident A/B calls itself at the accuracy shown, and the panel only ever sees the close calls it flags. A campaign choosing between a handful of creatives gets the winner picked automatically most of the time, with the budget spent exactly where the model is unsure rather than spread across every test.

## Honest scope
LaMem is image memorability from a memory-game study, not ad recall or neural response, so this demonstrates the decision method on the closest public proxy, not ad lift. Pairs and batches are drawn across the full memorability range; a single campaign's candidates sit in a narrower band and are harder, which is exactly why the accuracy-by-confidence table, which conditions on the predicted gap, is the transferable result rather than the headline pairwise number. On the lab's own response data the same selection curve would be measured directly.
