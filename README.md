# Does an image-memorability model know when it's wrong?

A frozen-backbone model predicts image memorability at the human consistency ceiling,
wrapped in a layer that reports when its predictions can be trusted and when they
can't, then reframes that trust as the decision a creative-testing product actually
makes: which of two creatives is more memorable, and when is that call confident
enough to skip a panel.

Built on [LaMem](http://memorability.csail.mit.edu/) (~60k images with human
memorability scores). Everything runs on [Modal](https://modal.com).

## Results

| model | Spearman ρ | notes |
|---|---|---|
| **CLIP ViT-B/32 + ridge** | **0.699** | at the human split-half consistency ceiling (ρ ≈ 0.68) |
| ResNet18 + ridge | 0.619 ± 0.009 | mean over the 5 official splits |
| pixel brightness | 0.097 | trivial-feature floor |

Point-estimate error (CLIP, n = 10k held-out): MAE 0.069, RMSE 0.088. The backbone is
frozen; only a ridge head and the trust layer are fit, so the whole thing trains in
minutes and is cheap to reproduce.

## The trust layer

A score is only useful if you know when to believe it.

**Calibration.** A heteroscedastic variance head gives per-item intervals. Raw
intervals are overconfident, so split-conformal calibration corrects them to
near-exact coverage with a finite-sample guarantee:

| nominal | 0.50 | 0.80 | 0.90 | 0.95 |
|---|---|---|---|---|
| empirical | 0.504 | 0.805 | 0.905 | 0.953 |

**Selective prediction.** Keeping the most-confident 30% drops MAE from 0.069 to 0.058.

**Group-conditional coverage.** The 90% guarantee holds within every predicted-score
quartile (0.90 to 0.92), not just on average.

**An honest negative.** Per-item variance is a *weak* error ranker (std-vs-error
Spearman 0.14). That limitation is load-bearing below.

## Creative selection under a testing budget

The result that maps to a creative-testing product. A product rarely needs an image's
absolute memorability score; it needs the right **decision** between creatives and a
confidence it can act on. Reframing the same model as an A/B selector, on 5k held-out
images:

- On random creative pairs, it picks the more memorable one **74%** of the time (50% is chance).
- Shown 5 candidates, its top pick is the true best **50%** of the time (20% chance); shown 10, it lands in the true top 2 **58%** of the time.
- **Which confidence to gate on.** The per-item variance is weak, but the *margin* between two predictions is strong. Automating the most-confident half of A/B calls **by margin** holds **87%** accuracy; by per-item variance, only 73%. Gate on the margin, not the error bar.

So the model is a far better chooser than scorer, and it knows which of its own calls
to stand behind. That is the product rule: auto-decide the confident calls, route the
close ones to a panel, spend the testing budget only where the model is unsure.
(`output/decision/budget_curve.png`)

## Does it transfer?

Two tests that this is a general human-response recipe, not a memorability one-off.

**Out-of-distribution creative.** Trained on LaMem photos, run on real ad creative
(Pitt Image Ads). The feature-space OOD distance separates ads from in-distribution
photos at **AUROC 0.929**, while the model's own variance cannot (0.50). At a
5%-false-positive operating point it routes 70% of ad creative to deferral and leaves
in-domain accuracy intact. The deployment rule: a photo-trained model should defer on
creative unlike its training data, and this is the signal that catches it.

**A second human-response signal.** The same recipe applied to AVA aesthetics (50+
votes per image) predicts the human-mean rating at Spearman 0.631, and conformal
coverage still holds. The intrinsic variance doesn't capture human *disagreement*
(0.08), but a head trained directly on the vote-spread recovers part of it (0.28) from
the image alone.

## Where it fails

A content-level reliability map (CLIP zero-shot categories). The 90% coverage holds
across every content type (0.88 to 0.94); the error does not. The model is least
trustworthy on outdoor/nature (MAE 0.084, where it systematically over-predicts) and
most on faces/portraits (0.059). This is the actionable map: auto-score the reliable
content, route the rest. (`output/taxonomy/taxonomy.png`)

## Run it

```sh
pip install modal && modal setup
modal run modal_run.py --model-name clip_ridge                 # main: accuracy + trust layer
modal run modal_run.py --all-splits --model-name clip_ridge    # cross-split variance
modal run modal_run.py --decision --model-name clip_ridge      # creative selection under budget
modal run modal_run.py --shift --model-name clip_ridge         # OOD-gated routing on ad creative
modal run modal_run.py --ava --model-name clip_ridge           # second human-response signal
modal run modal_run.py --taxonomy --model-name clip_ridge      # content-level failure map
```

LaMem downloads into a Modal volume once (a few minutes); later runs reuse it. Reports
and plots land in `output/`.

## Honest scope

LaMem memorability comes from a memory-game study, not ad recall or neural response,
so this demonstrates the *method* on the closest public proxy, not ad lift. The point
is the shape of the system: a calibrated human-response predictor that reports its own
trust, defers on unfamiliar creative, and turns its confidence into a selection
decision. On a lab's own response data, every curve here (the selection budget, the
disagreement head, the content map) would be measured directly against that target.
