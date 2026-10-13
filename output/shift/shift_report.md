# OOD-gated routing on the real ad-creative shift

The trust layer's job on a new domain is to route creative it was not trained on to human or neuro testing and auto-score the rest. This measures how well the OOD distance does that on the genuine LaMem-to-ad shift the transfer demo established, rather than an artificial within-LaMem split. `clip_ridge` is trained on LaMem photos; the pool is 2000 held-out LaMem test images (in-distribution, labeled) mixed with 1500 real ad images (out-of-distribution).

An earlier version induced the shift by holding out a feature-space cluster inside LaMem. That is the wrong test: LaMem is a single photo domain, so its subpopulations are not out-of-distribution from each other, and the OOD distance correctly does not flag them. The detector is built for genuine domain novelty, which ad creative is and a LaMem cluster is not.

## The OOD distance detects the shift; the model's own confidence does not
- OOD distance separates ad creative from LaMem photos at **AUROC 0.929**.
- The model's own predicted std separates them at **AUROC 0.496** (0.5 is chance).

The heteroscedastic std is trained on in-domain residuals and is near-blind to a new domain; the feature-space OOD distance is the signal that flags it. You need both, for different jobs.

## Routing: defer the out-of-distribution creative, keep the rest
- At the operating point set by the 95th percentile of in-distribution OOD distance, **70%** of ad creative is routed to deferral, while only **5%** of in-distribution images are.
- MAE on the LaMem images that remain in the auto-scored pool is 0.074, versus 0.074 with no gating: in-domain accuracy is preserved while the novel creative is removed.

**Routing curve: as you defer more of the pool by OOD distance**

|   deferral |   ad_recall |   precision |   mae_kept_lamem |
|-----------:|------------:|------------:|-----------------:|
|        0   |       0     |     nan     |            0.074 |
|        0.1 |       0.224 |       0.96  |            0.074 |
|        0.2 |       0.445 |       0.953 |            0.074 |
|        0.3 |       0.649 |       0.928 |            0.074 |
|        0.4 |       0.806 |       0.864 |            0.074 |
|        0.5 |       0.903 |       0.774 |            0.074 |

`deferral` is the fraction of the pool sent to human or neuro testing, most-out-of-distribution first; `ad_recall` is the share of ad creative caught; `precision` is the share of the deferred slice that is actually ad creative; `mae_kept_lamem` is the error on the labeled LaMem images still in the auto-scored pool. See `shift/routing.png`.

## What this means for Presonance
This is the deployment rule, measured on the real shift: the OOD distance routes genuinely novel creative, where a photo-trained model should not be trusted, to expensive testing with high recall, and leaves in-domain accuracy intact, where the model's own confidence cannot tell the difference. That routing signal is the product, not the single accuracy number.

## Honest scope
Ad creative has no memorability labels, so this measures routing quality (does the gate send the right items to testing), not predictive accuracy on ads. A within-domain pocket the model fits poorly but that is not feature-novel would be missed by this signal, which is why in-domain creative still relies on the uncertainty-based selective prediction from the main report.
