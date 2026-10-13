# Second response signal: does the model's uncertainty track real human disagreement?

Memorability is one human-response signal. To test whether the recipe (frozen `clip_ridge` features + ridge + heteroscedastic variance + conformal) is a general human-response engine rather than a memorability one-off, it was applied unchanged to a second signal, image aesthetics. The dataset is a 6000-image AVA subset where every image carries a histogram of 1-to-10 ratings from at least 50 human voters, so each image has a REAL human mean and a REAL human disagreement (the spread of those votes). That per-image disagreement is the quantity a brain-response product ultimately cares about, and it is almost never available; here it is.

(This stands in for OASIS valence/arousal, the closer affective construct, which is not cleanly obtainable programmatically. AVA is the robust equivalent and gives a cleaner disagreement measure: a full 50+ vote histogram per image rather than a single reported SD.)

## The recipe generalizes to a second signal
- Rank accuracy predicting the human mean aesthetic score: Spearman **0.631**.
- MAE on the 1-10 rating scale: 0.44 points.

Nothing changed but the training labels, so the trust layer is signal-agnostic: a calibrated human-response predictor, not a memorability model.

## Does the model's intrinsic uncertainty track human disagreement
- Spearman(predicted std, human disagreement) = **0.082**.
- Spearman(model error, human disagreement) = 0.153.
- Spearman(predicted std, model error) = 0.002 (the usual trust check).

The heteroscedastic std does not transfer to this signal: it tracks neither human disagreement nor, on this dataset, even its own error. That is expected and worth stating plainly. The variance head can only work when the model's error is predictable from the image features, as it is on memorability. Aesthetic error is largely idiosyncratic taste rather than something in the picture, so the intrinsic head has nothing to latch onto. See `ava/ava_disagreement.png`.

## A dedicated head predicts disagreement directly
- Spearman(dedicated head, human disagreement) = **0.279**, held out.

Trained straight on the vote-spread, a plain ridge on the same CLIP features predicts how much humans will disagree about a held-out image, well above the intrinsic std's 0.082. So the disagreement is partly in the creative itself, and the model can report it as a first-class output rather than hoping its own confidence encodes it. For a brain-response product that is the point: predict the response and the spread of the response. See `ava/ava_disagreement_head.png`.

## Calibration holds on the second dataset
|   nominal |   conformal_coverage |
|----------:|---------------------:|
|      0.5  |                0.515 |
|      0.8  |                0.799 |
|      0.9  |                0.894 |
|      0.95 |                0.95  |

Split-conformal coverage meets its nominal rate on aesthetics too, so the calibration is not an artifact of the memorability data.

## What this means for Presonance
Two things transfer cleanly to a second human-response signal: the point predictor (rank accuracy 0.631) and the conformal calibration. The model's intrinsic uncertainty does not, but a head trained directly on the human vote-spread recovers part of the disagreement from the creative alone. For a creative-testing product that is the shape of the system: predict the response, put an honest calibrated band on it, and separately predict how much the audience will disagree, routing the high-disagreement creative to real neuro testing.

## Honest scope
Aesthetics is a proxy for affective or neural response; the ideal target is the lab's own neuro data. The intrinsic heteroscedastic uncertainty does not capture disagreement on this signal, and the dedicated head captures it only to the degree reported above, so a production system would train the disagreement head on the real response distribution.
