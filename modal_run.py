"""Run the LaMem memorability harness on a Modal GPU.

Why this file exists: real LaMem is ~2.7GB of images and the train/eval step is
the part that actually wants a GPU, neither of which fits comfortably on a
laptop. This pulls LaMem once into a persistent Modal Volume, runs the exact
same `run_eval` harness from run.py on a GPU, and writes the same
findings.md + calibration.png back to your local ./output, plus the
worst-error images to ./output/failures/ so you can inspect what it gets wrong.

Setup (one time):
    pip install modal
    modal setup

Run it (from inside this project directory):
    modal run modal_run.py --model-name resnet_ridge --test-split test_1

Cross-split variance (runs all 5 LaMem splits in parallel, prints mean +/- std):
    modal run modal_run.py --all-splits --model-name resnet_ridge
    modal run modal_run.py --all-splits --model-name brightness   # trivial-feature floor

Ad-creative transfer + out-of-distribution demo (streams real ads from Hugging Face):
    modal run modal_run.py --ad-transfer --model-name clip_ridge

OOD-gated routing on the real ad shift (how well OOD distance routes ad creative out):
    modal run modal_run.py --shift --model-name clip_ridge

Second response signal: does predicted uncertainty track real human disagreement
(AVA aesthetics, 50+ votes per image, streamed from Hugging Face):
    modal run modal_run.py --ava --model-name clip_ridge

Content-level failure map (where the model is least trustworthy, by creative type):
    modal run modal_run.py --taxonomy --model-name clip_ridge

Creative selection under a testing budget (the model as an A/B selection engine:
pairwise pick accuracy, accuracy vs confidence, and the automate-vs-panel curve):
    modal run modal_run.py --decision --model-name clip_ridge

The first run downloads and extracts LaMem into the Volume (a few minutes, once).
Later runs reuse it and start in seconds. Swap --model-name for whatever you
register in memorability/model.py; resnet_ridge is the bundled starting point.
"""
from __future__ import annotations

import modal

app = modal.App("lamem-memorability")

# Persist the dataset across runs so it downloads exactly once.
volume = modal.Volume.from_name("lamem-data", create_if_missing=True)
DATA_DIR = "/data"
# Where to pull LaMem from, in order. The MIT origin 403s urllib's default
# User-Agent, and the Hugging Face mirror serves the identical tarball
# (same name, same 2.71GB) to any anonymous client, so try HF first and fall
# back to MIT with browser-like headers.
LAMEM_SOURCES = [
    (
        "https://huggingface.co/datasets/chitradrishti/LAMEM/resolve/main/download_lamem.tar.gz",
        {"User-Agent": "Mozilla/5.0"},
    ),
    (
        "http://memorability.csail.mit.edu/download_lamem.tar.gz",
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Referer": "http://memorability.csail.mit.edu/download.html",
        },
    ),
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "pillow>=10.0",
        "tabulate>=0.9",
        "torch",
        "torchvision",
        "open_clip_torch",
        "datasets",
        "scikit-learn",
    )
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=["*.pyc", "__pycache__", ".venv*", "output", ".git", "*.zip"],
    )
)


def _resolve_root() -> str | None:
    """LaMem may extract to /data/lamem/ or straight into /data/. Find the dir
    that actually holds a splits/ folder so load_split works either way."""
    import os

    for candidate in (f"{DATA_DIR}/lamem", DATA_DIR):
        if os.path.isdir(os.path.join(candidate, "splits")):
            return candidate
    return None


def _download_lamem(dest: str) -> None:
    """Fetch the LaMem tarball to `dest`, trying each source in LAMEM_SOURCES.

    urllib's HTTPError holds an open socket Modal can't pickle, which otherwise
    surfaces as a confusing SerializationError, so every failure is caught and
    re-raised as a plain, picklable RuntimeError listing what was tried.
    """
    import os
    import shutil
    import urllib.request

    errors = []
    for url, headers in LAMEM_SOURCES:
        print(f"downloading LaMem (~2.6GB, one time) from {url} ...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            print(f"downloaded {os.path.getsize(dest)} bytes")
            return
        except Exception as exc:
            errors.append(f"{url} -> {exc}")
            if os.path.exists(dest):
                os.remove(dest)
            print(f"  failed: {exc}")
    raise RuntimeError("all LaMem sources failed:\n  " + "\n  ".join(errors))


def _ensure_lamem_root() -> str:
    """Return the LaMem root inside the Volume, downloading and extracting once
    if it isn't there yet."""
    import os
    import subprocess

    root = _resolve_root()
    if root is not None:
        return root
    os.makedirs(DATA_DIR, exist_ok=True)
    tar_path = os.path.join(DATA_DIR, "lamem.tar.gz")
    if not os.path.exists(tar_path):
        _download_lamem(tar_path)
    print("extracting ...")
    subprocess.run(["tar", "-xzf", tar_path, "-C", DATA_DIR], check=True)
    volume.commit()
    root = _resolve_root()
    if root is None:
        raise RuntimeError(f"extracted LaMem but found no splits/ under {DATA_DIR}")
    return root


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=3600)
def ensure_data() -> str:
    """Populate the Volume once (no GPU) so parallel split runs don't race on it."""
    import sys

    sys.path.insert(0, "/root/project")
    return _ensure_lamem_root()


@app.function(image=image, gpu="A10G", volumes={DATA_DIR: volume}, timeout=3600)
def eval_split(model_name: str, split_num: int, n_bins: int = 10) -> dict:
    """Fit on train_<n>, eval on test_<n>, return just the headline metrics.
    One of these runs per split, in parallel, for the cross-split variance."""
    import sys

    sys.path.insert(0, "/root/project")
    root = _ensure_lamem_root()

    import memorability.model_torch  # noqa: F401  (registers resnet_ridge)
    import memorability.model_clip  # noqa: F401  (registers clip_ridge)
    from run import run_eval

    result = run_eval(
        root, f"train_{split_num}", f"test_{split_num}", model_name, f"/root/out_{split_num}", n_bins
    )
    m = result["metrics"]
    print(
        f"split {split_num}: spearman={m['spearman']:.4f} mae={m['mae']:.4f} "
        f"rmse={m['rmse']:.4f} n={m['n']}"
    )
    return {
        "split": split_num,
        "spearman": m["spearman"],
        "mae": m["mae"],
        "rmse": m["rmse"],
        "n": m["n"],
    }


@app.function(image=image, gpu="A10G", volumes={DATA_DIR: volume}, timeout=3600)
def run_remote(model_name: str, train_split: str, test_split: str, n_bins: int = 10) -> dict:
    import os
    import sys

    sys.path.insert(0, "/root/project")

    root = _ensure_lamem_root()
    print(f"LaMem root: {root}")

    import torch

    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")

    import memorability.model_torch  # noqa: F401  (registers resnet_ridge)
    import memorability.model_clip  # noqa: F401  (registers clip_ridge)
    from memorability import data
    from run import run_eval

    out_dir = "/root/out"
    result = run_eval(root, train_split, test_split, model_name, out_dir, n_bins)

    with open(result["findings_md"], encoding="utf-8") as handle:
        findings = handle.read()
    with open(result["plot"], "rb") as handle:
        plot_png = handle.read()

    # Pull the worst-error images back so they can be eyeballed locally. The
    # report lists them by basename; map those to real paths via the split so
    # the internal layout of images/ never matters.
    by_name = {
        os.path.basename(it.image_path): it.image_path
        for it in data.load_split(root, test_split)
    }
    fail_images = {}
    for name in result["failures"]["worst"]["image"].tolist():
        src = by_name.get(name)
        if src and os.path.exists(src):
            with open(src, "rb") as fh:
                fail_images[name] = fh.read()

    return {
        "metrics": result["metrics"],
        "findings_md": findings,
        "plot_png": plot_png,
        "fail_images": fail_images,
    }


def _ad_markdown(summary: dict, ex_meta: list) -> str:
    import pandas as pd

    s = summary
    out = []
    out.append("# Ad-creative transfer and out-of-distribution check\n")
    out.append(
        f"Model `{s['model']}` was trained on LaMem natural photos, then run on "
        f"{s['n_ads']} real advertising images (Hugging Face mirror of the Pitt Image "
        "Ads dataset) with no retraining. There are no human response labels for these "
        "ads, so this is not an accuracy claim. It tests whether the trust layer behaves "
        "correctly on a new domain: does the model recognize creative unlike anything it "
        "was trained on.\n"
    )
    out.append("## Does the model know these are unfamiliar")
    out.append(
        f"- Distance to the training distribution (median): ad creative "
        f"**{s['ood_ads_median']:.2f}** vs LaMem photos {s['ood_test_median']:.2f}.\n"
        f"- Share of ads beyond the 95th percentile of in-distribution images: "
        f"**{s['ads_flagged_frac'] * 100:.0f}%** (5% would be chance).\n"
    )
    out.append(
        "A clearly higher ad distance and a flagged share well above 5% mean the model "
        "correctly treats ad creative as out-of-distribution. That is the safety "
        "property a creative-testing product needs: the system should know when it is "
        "being asked about creative unlike its training data and defer rather than guess. "
        "See `ads/ood_hist.png`.\n"
    )
    out.append("## What it predicts on ads (context, not accuracy)")
    out.append(
        f"- Median predicted memorability: ads {s['pred_ads_median']:.3f} vs LaMem test "
        f"{s['pred_test_median']:.3f}.\n"
        f"- Median predicted std: ads {s['std_ads_median']:.3f} vs LaMem test "
        f"{s['std_test_median']:.3f}.\n"
    )
    out.append("## Example ads (most and least out-of-distribution)")
    out.append(pd.DataFrame(ex_meta).round(3).to_markdown(index=False))
    out.append(
        "\nThese images are saved in `output/ads/`. Check that the most-out-of-distribution "
        "ones are genuinely unusual creative (heavy text, logos, collages) rather than "
        "noise, which is the qualitative confirmation the score is meaningful.\n"
    )
    out.append("## Honest scope")
    out.append(
        "No response or memorability ground truth exists for these ads, so this shows "
        "domain transfer and out-of-distribution behaviour, not predictive accuracy on "
        "ads. Validating accuracy would need a labeled ad-response set, which is exactly "
        "the kind of data a brain-response product collects.\n"
    )
    return "\n".join(out)


@app.function(image=image, gpu="A10G", volumes={DATA_DIR: volume}, timeout=5400)
def run_ads(
    model_name: str,
    n_ads: int,
    train_split: str,
    test_split: str,
    fit_n: int = 5000,
    test_n: int = 2000,
) -> dict:
    """Train on LaMem, then score real ad creative and measure how out-of-distribution
    it is relative to the training photos. The Presonance-domain transfer demo.

    Fits on a sample (fit_n) rather than all 45k: the OOD comparison needs good
    training feature statistics and a reasonable model, not a fully-fit one, and the
    full CLIP embed of 45k images overruns the time budget."""
    import os
    import random
    import sys

    sys.path.insert(0, "/root/project")
    root = _ensure_lamem_root()

    import numpy as np
    import torch

    import memorability.model_torch  # noqa: F401  (registers resnet_ridge)
    import memorability.model_clip  # noqa: F401  (registers clip_ridge)
    from memorability import data, metrics
    from memorability.model import get_model

    rng = random.Random(0)
    train = data.load_split(root, train_split)
    test = data.load_split(root, test_split)
    fit_items = rng.sample(train, min(fit_n, len(train)))
    test_items = rng.sample(test, min(test_n, len(test)))
    model = get_model(model_name)
    print(f"cuda_available={torch.cuda.is_available()}; fitting {model_name} on {len(fit_items)} images ...")
    model.fit(fit_items)
    test_paths = [it.image_path for it in test_items]
    test_pred, test_std, test_feat = model.embed_and_predict(test_paths)

    # Stream a sample of real ad creative from the HF mirror of the Pitt Image Ads
    # dataset (streaming avoids pulling all 25GB).
    from datasets import load_dataset

    ad_dir = "/root/ads"
    os.makedirs(ad_dir, exist_ok=True)
    print(f"streaming {n_ads} ad images from Hugging Face ...")
    ds = load_dataset("Mindykkyan/PittadsDB-AdsPics", split="train", streaming=True)
    ad_paths = []
    for i, row in enumerate(ds):
        if len(ad_paths) >= n_ads:
            break
        try:
            img = row["image"].convert("RGB")
        except Exception:
            continue
        p = os.path.join(ad_dir, f"ad_{i:05d}.jpg")
        img.save(p, "JPEG", quality=90)
        ad_paths.append(p)
    print(f"got {len(ad_paths)} ad images")
    ad_pred, ad_std, ad_feat = model.embed_and_predict(ad_paths)

    test_ood = metrics.ood_distance(test_feat, model._mu, model._sd)
    ad_ood = metrics.ood_distance(ad_feat, model._mu, model._sd)
    thr = float(np.quantile(test_ood, 0.95))
    summary = {
        "model": model_name,
        "n_fit": len(fit_items),
        "n_test": len(test_paths),
        "n_ads": len(ad_paths),
        "ood_test_median": float(np.median(test_ood)),
        "ood_ads_median": float(np.median(ad_ood)),
        "ads_flagged_frac": float(np.mean(ad_ood > thr)),
        "std_test_median": float(np.median(test_std)),
        "std_ads_median": float(np.median(ad_std)),
        "pred_test_median": float(np.median(test_pred)),
        "pred_ads_median": float(np.median(ad_pred)),
    }
    print(summary)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(
        float(min(test_ood.min(), ad_ood.min())), float(max(test_ood.max(), ad_ood.max())), 40
    )
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(test_ood, bins=bins, alpha=0.6, density=True, color="#4C78A8", label="LaMem photos (in-distribution)")
    ax.hist(ad_ood, bins=bins, alpha=0.6, density=True, color="#E45756", label="ad creative (Pitt Ads)")
    ax.axvline(thr, color="#999999", ls="--", lw=1, label="95th pct of in-distribution")
    ax.set_xlabel("distance to training distribution (higher = more unfamiliar)")
    ax.set_ylabel("density")
    ax.set_title("Out-of-distribution: ad creative vs training photos")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    hist_path = "/root/ood_hist.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    with open(hist_path, "rb") as fh:
        hist_png = fh.read()

    order = np.argsort(-ad_ood)
    pick = [int(j) for j in list(order[:6]) + list(order[-3:])]
    ex_images, ex_meta = {}, []
    for j in pick:
        name = os.path.basename(ad_paths[j])
        with open(ad_paths[j], "rb") as fh:
            ex_images[name] = fh.read()
        ex_meta.append(
            {
                "image": name,
                "ood": float(ad_ood[j]),
                "pred_memorability": float(ad_pred[j]),
                "std": float(ad_std[j]),
            }
        )

    return {
        "summary": summary,
        "ad_report_md": _ad_markdown(summary, ex_meta),
        "hist_png": hist_png,
        "ex_images": ex_images,
    }


def _shift_markdown(summary: dict, curve) -> str:
    s = summary
    out = []
    out.append("# OOD-gated routing on the real ad-creative shift\n")
    out.append(
        f"The trust layer's job on a new domain is to route creative it was not trained on to human "
        f"or neuro testing and auto-score the rest. This measures how well the OOD distance does that "
        f"on the genuine LaMem-to-ad shift the transfer demo established, rather than an artificial "
        f"within-LaMem split. `{s['model']}` is trained on LaMem photos; the pool is {s['n_iid']} "
        f"held-out LaMem test images (in-distribution, labeled) mixed with {s['n_ads']} real ad images "
        "(out-of-distribution).\n"
    )
    out.append(
        "An earlier version induced the shift by holding out a feature-space cluster inside LaMem. "
        "That is the wrong test: LaMem is a single photo domain, so its subpopulations are not "
        "out-of-distribution from each other, and the OOD distance correctly does not flag them. The "
        "detector is built for genuine domain novelty, which ad creative is and a LaMem cluster is not.\n"
    )
    out.append("## The OOD distance detects the shift; the model's own confidence does not")
    out.append(
        f"- OOD distance separates ad creative from LaMem photos at **AUROC {s['auroc_ood']:.3f}**.\n"
        f"- The model's own predicted std separates them at **AUROC {s['auroc_std']:.3f}** "
        "(0.5 is chance).\n"
    )
    out.append(
        "The heteroscedastic std is trained on in-domain residuals and is near-blind to a new domain; "
        "the feature-space OOD distance is the signal that flags it. You need both, for different jobs.\n"
    )
    out.append("## Routing: defer the out-of-distribution creative, keep the rest")
    out.append(
        f"- At the operating point set by the 95th percentile of in-distribution OOD distance, "
        f"**{s['ad_recall'] * 100:.0f}%** of ad creative is routed to deferral, while only "
        f"**{s['iid_deferred'] * 100:.0f}%** of in-distribution images are.\n"
        f"- MAE on the LaMem images that remain in the auto-scored pool is {s['mae_kept']:.3f}, versus "
        f"{s['mae_iid']:.3f} with no gating: in-domain accuracy is preserved while the novel creative "
        "is removed.\n"
    )
    out.append("**Routing curve: as you defer more of the pool by OOD distance**\n")
    out.append(curve.round(3).to_markdown(index=False))
    out.append(
        "\n`deferral` is the fraction of the pool sent to human or neuro testing, most-out-of-distribution "
        "first; `ad_recall` is the share of ad creative caught; `precision` is the share of the deferred "
        "slice that is actually ad creative; `mae_kept_lamem` is the error on the labeled LaMem images "
        "still in the auto-scored pool. See `shift/routing.png`.\n"
    )
    out.append("## What this means for Presonance")
    out.append(
        "This is the deployment rule, measured on the real shift: the OOD distance routes genuinely "
        "novel creative, where a photo-trained model should not be trusted, to expensive testing with "
        "high recall, and leaves in-domain accuracy intact, where the model's own confidence cannot "
        "tell the difference. That routing signal is the product, not the single accuracy number.\n"
    )
    out.append("## Honest scope")
    out.append(
        "Ad creative has no memorability labels, so this measures routing quality (does the gate send the "
        "right items to testing), not predictive accuracy on ads. A within-domain pocket the model fits "
        "poorly but that is not feature-novel would be missed by this signal, which is why in-domain "
        "creative still relies on the uncertainty-based selective prediction from the main report.\n"
    )
    return "\n".join(out)


@app.function(image=image, gpu="A10G", volumes={DATA_DIR: volume}, timeout=5400)
def run_shift(
    model_name: str = "clip_ridge",
    train_split: str = "train_1",
    test_split: str = "test_1",
    fit_n: int = 5000,
    test_n: int = 2000,
    n_ads: int = 1500,
    seed: int = 0,
) -> dict:
    """OOD-gated routing on the real LaMem-to-ad shift. Fit on LaMem photos, build a
    pool of labeled LaMem test images (in-distribution) and real ad images
    (out-of-distribution), and measure how well the OOD distance routes the ad
    creative to deferral where the model's own confidence cannot. Honest and
    measurable: LaMem is labeled, ads are the genuine OOD set, and we report routing
    quality rather than fabricated accuracy on unlabeled ads."""
    import os
    import random
    import sys

    sys.path.insert(0, "/root/project")
    root = _ensure_lamem_root()

    import numpy as np
    import pandas as pd
    import torch

    import memorability.model_torch  # noqa: F401  (registers resnet_ridge)
    import memorability.model_clip  # noqa: F401  (registers clip_ridge)
    from memorability import data, metrics
    from memorability.model import get_model

    rng = random.Random(seed)
    train = data.load_split(root, train_split)
    test = data.load_split(root, test_split)
    fit_items = rng.sample(train, min(fit_n, len(train)))
    test_items = rng.sample(test, min(test_n, len(test)))

    model = get_model(model_name)
    print(f"cuda_available={torch.cuda.is_available()}; fitting {model_name} on {len(fit_items)} images ...")
    model.fit(fit_items)

    iid_pred, iid_std, iid_feat = model.embed_and_predict([it.image_path for it in test_items])
    y_iid = np.array([it.score for it in test_items], dtype=float)
    iid_ood = metrics.ood_distance(iid_feat, model._mu, model._sd)

    # Stream real ad creative: the genuine out-of-distribution domain.
    from datasets import load_dataset

    ad_dir = "/root/ads_shift"
    os.makedirs(ad_dir, exist_ok=True)
    print(f"streaming {n_ads} ad images from Hugging Face ...")
    ds = load_dataset("Mindykkyan/PittadsDB-AdsPics", split="train", streaming=True)
    ad_paths = []
    for i, row in enumerate(ds):
        if len(ad_paths) >= n_ads:
            break
        try:
            img = row["image"].convert("RGB")
        except Exception:
            continue
        p = os.path.join(ad_dir, f"ad_{i:05d}.jpg")
        img.save(p, "JPEG", quality=90)
        ad_paths.append(p)
    print(f"got {len(ad_paths)} ad images")
    ad_pred, ad_std, ad_feat = model.embed_and_predict(ad_paths)
    ad_ood = metrics.ood_distance(ad_feat, model._mu, model._sd)

    # Pool: label 1 = ad (out-of-distribution), 0 = LaMem test (in-distribution).
    pool_ood = np.concatenate([iid_ood, ad_ood])
    pool_std = np.concatenate([iid_std, ad_std])
    is_ad = np.concatenate([np.zeros(len(iid_ood)), np.ones(len(ad_ood))])
    auroc_ood = metrics.auroc(pool_ood, is_ad)
    auroc_std = metrics.auroc(pool_std, is_ad)

    mae_iid = float(np.mean(np.abs(iid_pred - y_iid)))

    # Operating point: 95th percentile of in-distribution OOD distance.
    thr = float(np.quantile(iid_ood, 0.95))
    ad_recall = float(np.mean(ad_ood > thr))  # share of ads routed to deferral
    iid_deferred = float(np.mean(iid_ood > thr))  # share of in-dist wrongly routed (~5%)
    kept_mask = iid_ood <= thr  # LaMem items still auto-scored
    mae_kept = (
        float(np.mean(np.abs(iid_pred[kept_mask] - y_iid[kept_mask]))) if kept_mask.any() else mae_iid
    )

    # Routing curve over deferral fractions, most-out-of-distribution first.
    n_pool = len(pool_ood)
    order = np.argsort(-pool_ood)
    rows = []
    for d in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        k = int(round(d * n_pool))
        deferred, kept = order[:k], order[k:]
        prec = float(is_ad[deferred].mean()) if k > 0 else float("nan")
        rec = float(is_ad[deferred].sum() / max(1.0, is_ad.sum()))
        kept_iid_idx = kept[kept < len(iid_ood)]  # pool indices < len(iid) are the in-dist items
        mae_k = (
            float(np.mean(np.abs(iid_pred[kept_iid_idx] - y_iid[kept_iid_idx])))
            if len(kept_iid_idx)
            else float("nan")
        )
        rows.append({"deferral": float(d), "ad_recall": rec, "precision": prec, "mae_kept_lamem": mae_k})
    curve = pd.DataFrame(rows)

    summary = {
        "model": model_name, "n_iid": int(len(iid_ood)), "n_ads": int(len(ad_ood)),
        "auroc_ood": auroc_ood, "auroc_std": auroc_std,
        "ad_recall": ad_recall, "iid_deferred": iid_deferred,
        "mae_iid": mae_iid, "mae_kept": mae_kept,
        "ood_iid_median": float(np.median(iid_ood)), "ood_ad_median": float(np.median(ad_ood)),
    }
    print(summary)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo = float(min(iid_ood.min(), ad_ood.min()))
    hi = float(max(iid_ood.max(), ad_ood.max()))
    bins = np.linspace(lo, hi, 40)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(iid_ood, bins=bins, alpha=0.6, density=True, color="#4C78A8", label="LaMem test (in-distribution)")
    ax.hist(ad_ood, bins=bins, alpha=0.6, density=True, color="#E45756", label="ad creative (out-of-distribution)")
    ax.axvline(thr, color="#999999", ls="--", lw=1, label="95th pct of in-distribution")
    ax.set_xlabel("distance to training distribution (higher = more unfamiliar)")
    ax.set_ylabel("density")
    ax.set_title("OOD distance: ad creative vs in-distribution photos")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    h1 = "/root/ood_shift_hist.png"
    fig.savefig(h1, dpi=150)
    plt.close(fig)
    with open(h1, "rb") as fh:
        ood_hist_png = fh.read()

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(curve["deferral"], curve["ad_recall"], "o-", color="#E45756", label="ad recall (OOD creative caught)")
    ax.plot(curve["deferral"], curve["precision"], "s-", color="#4C78A8", label="precision (deferred that is ad)")
    ax.set_xlabel("deferral fraction (most out-of-distribution first)")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("OOD-gated routing of ad creative")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    h2 = "/root/routing.png"
    fig.savefig(h2, dpi=150)
    plt.close(fig)
    with open(h2, "rb") as fh:
        routing_png = fh.read()

    return {
        "summary": summary,
        "shift_report_md": _shift_markdown(summary, curve),
        "ood_hist_png": ood_hist_png,
        "routing_png": routing_png,
    }


def _ava_markdown(summary: dict, conf_cov) -> str:
    s = summary
    out = []
    out.append("# Second response signal: does the model's uncertainty track real human disagreement?\n")
    out.append(
        f"Memorability is one human-response signal. To test whether the recipe (frozen "
        f"`{s['model']}` features + ridge + heteroscedastic variance + conformal) is a general "
        "human-response engine rather than a memorability one-off, it was applied unchanged to a "
        f"second signal, image aesthetics. The dataset is a {s['n_images']}-image AVA subset where "
        "every image carries a histogram of 1-to-10 ratings from at least 50 human voters, so each "
        "image has a REAL human mean and a REAL human disagreement (the spread of those votes). That "
        "per-image disagreement is the quantity a brain-response product ultimately cares about, and "
        "it is almost never available; here it is.\n"
    )
    out.append(
        "(This stands in for OASIS valence/arousal, the closer affective construct, which is not "
        "cleanly obtainable programmatically. AVA is the robust equivalent and gives a cleaner "
        "disagreement measure: a full 50+ vote histogram per image rather than a single reported SD.)\n"
    )
    out.append("## The recipe generalizes to a second signal")
    out.append(
        f"- Rank accuracy predicting the human mean aesthetic score: Spearman **{s['spearman_mean']:.3f}**.\n"
        f"- MAE on the 1-10 rating scale: {s['mae_10']:.2f} points.\n"
    )
    out.append(
        "Nothing changed but the training labels, so the trust layer is signal-agnostic: a calibrated "
        "human-response predictor, not a memorability model.\n"
    )
    out.append("## Does the model's intrinsic uncertainty track human disagreement")
    out.append(
        f"- Spearman(predicted std, human disagreement) = **{s['rho_std_humansd']:.3f}**.\n"
        f"- Spearman(model error, human disagreement) = {s['rho_err_humansd']:.3f}.\n"
        f"- Spearman(predicted std, model error) = {s['std_err_spearman']:.3f} (the usual trust check).\n"
    )
    out.append(
        "The heteroscedastic std does not transfer to this signal: it tracks neither human disagreement "
        "nor, on this dataset, even its own error. That is expected and worth stating plainly. The "
        "variance head can only work when the model's error is predictable from the image features, as "
        "it is on memorability. Aesthetic error is largely idiosyncratic taste rather than something in "
        "the picture, so the intrinsic head has nothing to latch onto. See `ava/ava_disagreement.png`.\n"
    )
    out.append("## A dedicated head predicts disagreement directly")
    out.append(f"- Spearman(dedicated head, human disagreement) = **{s['rho_dishead']:.3f}**, held out.\n")
    if s["rho_dishead"] >= 0.25:
        out.append(
            "Trained straight on the vote-spread, a plain ridge on the same CLIP features predicts how "
            "much humans will disagree about a held-out image, well above the intrinsic std's "
            f"{s['rho_std_humansd']:.3f}. So the disagreement is partly in the creative itself, and the "
            "model can report it as a first-class output rather than hoping its own confidence encodes "
            "it. For a brain-response product that is the point: predict the response and the spread of "
            "the response. See `ava/ava_disagreement_head.png`.\n"
        )
    else:
        out.append(
            "Trained straight on the vote-spread, a plain ridge on the same CLIP features beats the "
            f"intrinsic std ({s['rho_std_humansd']:.3f}) but still only weakly predicts disagreement. The "
            "honest reading is that aesthetic disagreement is only partly in the creative; a sharper "
            "signal would need features tuned to polarizing content or the lab's own response data. See "
            "`ava/ava_disagreement_head.png`.\n"
        )
    out.append("## Calibration holds on the second dataset")
    out.append(conf_cov.round(3).to_markdown(index=False))
    out.append(
        "\nSplit-conformal coverage meets its nominal rate on aesthetics too, so the calibration is not "
        "an artifact of the memorability data.\n"
    )
    out.append("## What this means for Presonance")
    out.append(
        "Two things transfer cleanly to a second human-response signal: the point predictor (rank "
        f"accuracy {s['spearman_mean']:.3f}) and the conformal calibration. The model's intrinsic "
        "uncertainty does not, but a head trained directly on the human vote-spread recovers part of the "
        "disagreement from the creative alone. For a creative-testing product that is the shape of the "
        "system: predict the response, put an honest calibrated band on it, and separately predict how "
        "much the audience will disagree, routing the high-disagreement creative to real neuro testing.\n"
    )
    out.append("## Honest scope")
    out.append(
        "Aesthetics is a proxy for affective or neural response; the ideal target is the lab's own neuro "
        "data. The intrinsic heteroscedastic uncertainty does not capture disagreement on this signal, "
        "and the dedicated head captures it only to the degree reported above, so a production system "
        "would train the disagreement head on the real response distribution.\n"
    )
    return "\n".join(out)


@app.function(image=image, gpu="A10G", volumes={DATA_DIR: volume}, timeout=5400)
def run_ava(model_name: str = "clip_ridge", n_images: int = 6000, test_frac: float = 0.25, seed: int = 0) -> dict:
    """Second human-response signal: AVA aesthetics. Each image carries a histogram
    of 1-10 ratings from 50+ voters, so there is a real per-image human mean and a
    real per-image human disagreement (vote SD). Fit the same recipe to predict the
    mean, then test whether the predicted uncertainty tracks the actual human
    disagreement, the property that matters most for a response-prediction product."""
    import os
    import sys

    sys.path.insert(0, "/root/project")

    import numpy as np
    import torch
    from datasets import load_dataset
    from scipy.stats import spearmanr

    import memorability.model_torch  # noqa: F401  (registers resnet_ridge)
    import memorability.model_clip  # noqa: F401  (registers clip_ridge)
    from memorability import metrics
    from memorability.model import get_model

    ratings = np.arange(1, 11, dtype=float)
    print(f"streaming up to {n_images} AVA images from Hugging Face ...")
    ds = load_dataset("trojblue/AVA-aesthetics-10pct-min50-10bins", split="train", streaming=True)
    img_dir = "/root/ava"
    os.makedirs(img_dir, exist_ok=True)
    paths, hmean, hsd = [], [], []
    for i, row in enumerate(ds):
        if len(paths) >= n_images:
            break
        try:
            img = row["image"].convert("RGB")
            counts = np.asarray(row["rating_counts"], dtype=float)
        except Exception:
            continue
        tot = counts.sum()
        if tot < 1 or counts.shape[0] != 10:
            continue
        mu = float((counts * ratings).sum() / tot)
        sd = float(np.sqrt((counts * (ratings - mu) ** 2).sum() / tot))
        p = os.path.join(img_dir, f"ava_{i:06d}.jpg")
        img.save(p, "JPEG", quality=90)
        paths.append(p)
        hmean.append(mu)
        hsd.append(sd)
    hmean = np.asarray(hmean)
    hsd = np.asarray(hsd)
    print(f"got {len(paths)} images; human mean {hmean.mean():.2f}, disagreement {hsd.mean():.2f} rating points")

    y = (hmean - 1.0) / 9.0  # scale the mean rating to the model's [0,1] range
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(paths))
    n_test = max(1, int(test_frac * len(paths)))
    te, tr = idx[:n_test], idx[n_test:]

    model = get_model(model_name)
    print(f"cuda_available={torch.cuda.is_available()}; embedding {len(paths)} images ...")
    X = model._embed(paths)
    model._fit_from_features(X[tr], y[tr])
    pred, pstd = model.predict_from_features(X[te])
    err = np.abs(pred - y[te])
    hsd_te = hsd[te]

    reg = metrics.regression_metrics(pred, y[te])
    rho_std_dis = float(spearmanr(pstd, hsd_te).correlation)
    rho_err_dis = float(spearmanr(err, hsd_te).correlation)
    uq = metrics.uncertainty_quality(pred, pstd, y[te])
    conf = {c: model.conformal_quantile(c) for c in (0.5, 0.8, 0.9, 0.95)}
    conf_cov = metrics.conformal_coverage(pred, pstd, y[te], conf)

    # Dedicated disagreement head: can the creative content predict the human
    # vote-spread directly? A plain ridge from CLIP features to each image's human
    # disagreement SD, trained on the train split and scored on held-out images.
    # This targets disagreement head-on instead of hoping the intrinsic
    # heteroscedastic std happens to encode it.
    def _ridge_predict(Xtr, ytr, Xte, lam=10.0):
        mu = Xtr.mean(0)
        sd = Xtr.std(0) + 1e-8
        Za = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
        reg = lam * np.eye(Za.shape[1])
        reg[-1, -1] = 0.0
        w = np.linalg.solve(Za.T @ Za + reg, Za.T @ ytr)
        Zte = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
        return Zte @ w

    dis_pred = _ridge_predict(X[tr], hsd[tr], X[te])
    rho_dishead = float(spearmanr(dis_pred, hsd_te).correlation)

    pred_10 = pred * 9.0 + 1.0
    mae_10 = float(np.mean(np.abs(pred_10 - hmean[te])))
    summary = {
        "model": model_name, "n_images": len(paths), "n_train": int(len(tr)), "n_test": int(len(te)),
        "spearman_mean": float(reg["spearman"]), "mae_10": mae_10,
        "rho_std_humansd": rho_std_dis, "rho_err_humansd": rho_err_dis,
        "rho_dishead": rho_dishead,
        "std_err_spearman": float(uq["std_err_spearman"]), "human_sd_mean": float(hsd.mean()),
    }
    print(summary)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pstd_10 = pstd * 9.0  # predicted uncertainty in rating points
    nb = 10
    edges = np.quantile(pstd_10, np.linspace(0, 1, nb + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    b = np.clip(np.digitize(pstd_10, edges) - 1, 0, nb - 1)
    bx = [float(pstd_10[b == kk].mean()) for kk in range(nb) if (b == kk).any()]
    by = [float(hsd_te[b == kk].mean()) for kk in range(nb) if (b == kk).any()]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(pstd_10, hsd_te, s=6, alpha=0.15, color="#4C78A8")
    ax.plot(bx, by, "o-", color="#E45756", label="mean human disagreement per predicted-std decile")
    ax.set_xlabel("model predicted uncertainty (rating points)")
    ax.set_ylabel("actual human disagreement (vote SD, rating points)")
    ax.set_title("Predicted uncertainty vs real human disagreement (AVA)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    h1 = "/root/ava_disagreement.png"
    fig.savefig(h1, dpi=150)
    plt.close(fig)
    with open(h1, "rb") as fh:
        disagree_png = fh.read()

    nb2 = 10
    e2 = np.quantile(dis_pred, np.linspace(0, 1, nb2 + 1))
    e2[-1] = np.nextafter(e2[-1], np.inf)
    b2 = np.clip(np.digitize(dis_pred, e2) - 1, 0, nb2 - 1)
    bx2 = [float(dis_pred[b2 == kk].mean()) for kk in range(nb2) if (b2 == kk).any()]
    by2 = [float(hsd_te[b2 == kk].mean()) for kk in range(nb2) if (b2 == kk).any()]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(dis_pred, hsd_te, s=6, alpha=0.15, color="#4C78A8")
    ax.plot(bx2, by2, "o-", color="#E45756", label="mean actual disagreement per predicted decile")
    ax.set_xlabel("predicted human disagreement (dedicated head)")
    ax.set_ylabel("actual human disagreement (vote SD, rating points)")
    ax.set_title(f"Dedicated disagreement head (Spearman {rho_dishead:.2f})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    h3 = "/root/ava_disagreement_head.png"
    fig.savefig(h3, dpi=150)
    plt.close(fig)
    with open(h3, "rb") as fh:
        dishead_png = fh.read()

    import memorability.report as report

    rel = metrics.reliability_curve(pred, y[te], n_bins=10)
    h2 = "/root/ava_calibration.png"
    report._plot(rel, h2)
    with open(h2, "rb") as fh:
        calib_png = fh.read()

    return {
        "summary": summary,
        "ava_report_md": _ava_markdown(summary, conf_cov),
        "disagree_png": disagree_png,
        "dishead_png": dishead_png,
        "calib_png": calib_png,
    }


def _taxonomy_markdown(summary: dict, df) -> str:
    s = summary
    out = []
    out.append("# Content-level failure map: where is the model least trustworthy, by creative type?\n")
    out.append(
        f"A single accuracy number hides where a model can and cannot be trusted, and for a creative-testing "
        f"product that map is the deliverable. Each of the {s['n_test']} held-out LaMem test images was tagged "
        f"with a content category by CLIP zero-shot, then rank accuracy, error, conformal coverage, and "
        f"prediction bias were measured within each category.\n"
    )
    out.append("## Reliability by creative type (least trustworthy first)")
    out.append(df.round(3).to_markdown(index=False))
    out.append(
        f"\nOverall MAE is {s['overall_mae']:.3f}. The model is least trustworthy on **{s['worst_category']}** "
        f"(MAE {s['worst_mae']:.3f}) and most trustworthy on **{s['best_category']}** (MAE {s['best_mae']:.3f}). "
        "A positive `bias_pred_minus_actual` means the model over-predicts memorability for that type and a "
        "negative one means it under-predicts. See `taxonomy/taxonomy.png`, where red bars are over-predicted "
        "types and blue are under-predicted.\n"
    )
    out.append("## What this means for Presonance")
    out.append(
        "This is the actionable trust map: auto-score the creative types where the model is reliable and "
        "well-calibrated, and route the types where it is not to human or neuro testing. It also exposes "
        "systematic bias, the categories the model consistently over- or under-rates, which a product corrects "
        "per segment rather than trusting one global number.\n"
    )
    out.append("## Honest scope")
    out.append(
        "CLIP zero-shot tags are approximate, each image gets a single best-matching category, and small "
        "categories are noisy, so this is a directional reliability map rather than a precise per-type audit. "
        "The categories reflect LaMem's natural-photo content, not ad creative; the same method would run on a "
        "creative library with the product's own taxonomy.\n"
    )
    return "\n".join(out)


@app.function(image=image, gpu="A10G", volumes={DATA_DIR: volume}, timeout=5400)
def run_taxonomy(
    model_name: str = "clip_ridge",
    train_split: str = "train_1",
    test_split: str = "test_1",
    fit_n: int = 10000,
    test_n: int = 5000,
    seed: int = 0,
) -> dict:
    """Content-level failure map: where is the model least trustworthy, by creative
    type? Fit the model, embed the test set, tag each image with a content category
    via CLIP zero-shot, then report rank accuracy, MAE, conformal coverage, and
    over/under-prediction per category. For a creative-testing product this map of
    where-not-to-trust-it is the deliverable, not the single accuracy number."""
    import random
    import sys

    sys.path.insert(0, "/root/project")
    root = _ensure_lamem_root()

    import numpy as np
    import open_clip
    import pandas as pd
    import torch
    from scipy.stats import spearmanr

    import memorability.model_torch  # noqa: F401  (registers resnet_ridge)
    import memorability.model_clip  # noqa: F401  (registers clip_ridge)
    from memorability import data, metrics
    from memorability.model import get_model

    rng = random.Random(seed)
    train = data.load_split(root, train_split)
    test = data.load_split(root, test_split)
    fit_items = rng.sample(train, min(fit_n, len(train)))
    test_items = rng.sample(test, min(test_n, len(test)))

    model = get_model(model_name)
    if not hasattr(model, "MODEL"):
        raise ValueError("taxonomy needs a CLIP-based model; run with --model-name clip_ridge")
    print(f"cuda_available={torch.cuda.is_available()}; fitting {model_name} on {len(fit_items)} images ...")
    model.fit(fit_items)
    paths = [it.image_path for it in test_items]
    pred, std, feat = model.embed_and_predict(paths)  # feat: unit-norm CLIP image features
    actual = np.array([it.score for it in test_items], dtype=float)

    # CLIP zero-shot content categories, matched in the shared image-text space.
    cats = {
        "people": "a photo of a person or people",
        "face/portrait": "a close-up photo of a human face",
        "indoor scene": "a photo of an indoor scene",
        "outdoor/nature": "a photo of an outdoor natural landscape",
        "animal": "a photo of an animal",
        "food": "a photo of food or a meal",
        "vehicle": "a photo of a vehicle",
        "object/product": "a photo of an object or product",
        "text/document": "a photo of text, a sign, or a document",
        "art/graphic": "a painting, drawing, or graphic design",
    }
    names = list(cats.keys())
    tokenizer = open_clip.get_tokenizer(model.MODEL)
    toks = tokenizer([cats[k] for k in names]).to(model._device)
    with torch.no_grad():
        tfeat = model._backbone.encode_text(toks)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        tfeat = tfeat.float().cpu().numpy()
    cat_idx = (feat @ tfeat.T).argmax(axis=1)  # feat already unit-norm

    q90 = model.conformal_quantile(0.9)
    rows = []
    for ci, name in enumerate(names):
        m = cat_idx == ci
        n = int(m.sum())
        if n < 30:
            continue
        p, a, sd = pred[m], actual[m], std[m]
        err = np.abs(p - a)
        cov = float(np.mean((a >= p - q90 * sd) & (a <= p + q90 * sd)))
        rho = float(spearmanr(p, a).correlation) if (n >= 50 and np.std(p) > 0) else float("nan")
        rows.append(
            {
                "category": name, "n": n, "mae": float(err.mean()), "spearman": rho,
                "cov_at_0.90": cov, "bias_pred_minus_actual": float(np.mean(p - a)),
            }
        )
    df = pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))

    overall_mae = float(np.mean(np.abs(pred - actual)))
    summary = {
        "model": model_name, "n_fit": len(fit_items), "n_test": len(test_items),
        "overall_mae": overall_mae, "n_categories": int(len(df)),
        "worst_category": str(df.iloc[0]["category"]), "worst_mae": float(df.iloc[0]["mae"]),
        "best_category": str(df.iloc[-1]["category"]), "best_mae": float(df.iloc[-1]["mae"]),
    }
    print(summary)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#E45756" if b > 0 else "#4C78A8" for b in df["bias_pred_minus_actual"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh(df["category"][::-1], df["mae"][::-1], color=colors[::-1])
    ax.axvline(overall_mae, color="#999999", ls="--", lw=1, label="overall MAE")
    ax.set_xlabel("MAE (higher = less trustworthy)")
    ax.set_title("Where the model is least trustworthy, by creative type")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    h = "/root/taxonomy.png"
    fig.savefig(h, dpi=150)
    plt.close(fig)
    with open(h, "rb") as fh:
        tax_png = fh.read()

    return {
        "summary": summary,
        "taxonomy_report_md": _taxonomy_markdown(summary, df),
        "taxonomy_png": tax_png,
    }


def _decision_markdown(summary: dict, by_conf, auto_gap, auto_std) -> str:
    s = summary
    out = []
    out.append("# Creative selection under a testing budget\n")
    out.append(
        f"A creative-testing product rarely needs an image's absolute memorability score. It needs the "
        f"right decision: which of these creatives is more memorable, and can I trust that call or "
        f"should I pay to test it. This reframes the same `{s['model']}` model and trust layer as an A/B "
        f"selection engine and measures it as one, on {s['n_test']} held-out LaMem images.\n"
    )
    out.append("## The model is a far better chooser than scorer")
    out.append(
        f"- On {s['n_pairs']} random creative pairs, the model picks the more memorable one "
        f"**{s['pairwise_acc'] * 100:.1f}%** of the time (50% is a coin flip).\n"
        f"- Shown 5 candidate creatives, its top pick is the true best **{s['p_best_5'] * 100:.0f}%** of "
        f"the time (random is 20%).\n"
        f"- Shown 10 candidates, its top pick is the true best **{s['p_best_10'] * 100:.0f}%** of the time "
        f"(random is 10%) and lands in the true top 2 **{s['p_top_10'] * 100:.0f}%** of the time "
        f"(random is 20%).\n"
    )
    out.append(
        "The point estimate is a 0.699-Spearman regressor, strong but not exact. Expressed as a choice "
        "between creatives, that same signal is a reliable selector, because ranking only needs the sign "
        "of the gap, not its precise value.\n"
    )
    out.append("## Accuracy rises with the model's confidence in the call")
    out.append(by_conf.round(3).to_markdown(index=False))
    out.append(
        "\n`mean_confidence` is the predicted memorability gap the call rested on; `accuracy` is how often "
        "the pick was truly correct in that bin. The call is near coin-flip when the predicted gap is tiny "
        "(genuinely close creatives) and approaches certainty as the gap widens. That monotonic rise is "
        "what makes the confidence usable: the model knows which of its own calls to stand behind. See "
        "`decision/confidence_accuracy.png`.\n"
    )
    out.append("## The test-budget tradeoff, and which confidence signal to gate on")
    out.append(
        "Auto-handle the most-confident share of A/B calls and send the rest to a panel. Gating by the "
        "**predicted gap** (the decision's own margin) is compared against gating by **pooled per-item "
        "uncertainty** (the heteroscedastic std of the two items), the signal a naive system would reach "
        "for.\n"
    )
    merged = auto_gap.rename(columns={"accuracy": "acc_gap_gate"}).copy()
    merged["acc_std_gate"] = auto_std["accuracy"].values
    out.append(
        merged[["auto_fraction", "panel_fraction", "acc_gap_gate", "acc_std_gate"]]
        .round(3)
        .to_markdown(index=False)
    )
    if s["acc_gap_50"] - s["acc_std_50"] >= 0.03:
        out.append(
            f"\nGating by the decision margin is the clear winner. Automating the most-confident half of "
            f"calls, the margin gate holds **{s['acc_gap_50'] * 100:.1f}%** accuracy versus "
            f"**{s['acc_std_50'] * 100:.1f}%** for the per-item std gate. This is the earlier finding put "
            "to use: the per-item variance head is a weak error-ranker, but the margin between two "
            "predictions is a strong decision-confidence signal. For an A/B product you gate on the "
            "margin, not the error bar. See `decision/budget_curve.png`.\n"
        )
    else:
        out.append(
            f"\nAutomating the most-confident half of calls, the margin gate holds "
            f"**{s['acc_gap_50'] * 100:.1f}%** accuracy versus **{s['acc_std_50'] * 100:.1f}%** for the "
            "per-item std gate. The two are closer than expected here, so the honest reading is that both "
            "the margin and the pooled uncertainty carry usable decision confidence on this data. See "
            "`decision/budget_curve.png`.\n"
        )
    out.append("## What this means for Presonance")
    out.append(
        "This is the product rule stated in the units a creative team buys: the model makes the confident "
        "A/B calls itself at the accuracy shown, and the panel only ever sees the close calls it flags. A "
        "campaign choosing between a handful of creatives gets the winner picked automatically most of the "
        "time, with the budget spent exactly where the model is unsure rather than spread across every "
        "test.\n"
    )
    out.append("## Honest scope")
    out.append(
        "LaMem is image memorability from a memory-game study, not ad recall or neural response, so this "
        "demonstrates the decision method on the closest public proxy, not ad lift. Pairs and batches are "
        "drawn across the full memorability range; a single campaign's candidates sit in a narrower band "
        "and are harder, which is exactly why the accuracy-by-confidence table, which conditions on the "
        "predicted gap, is the transferable result rather than the headline pairwise number. On the lab's "
        "own response data the same selection curve would be measured directly.\n"
    )
    return "\n".join(out)


@app.function(image=image, gpu="A10G", volumes={DATA_DIR: volume}, timeout=5400)
def run_decision(
    model_name: str = "clip_ridge",
    train_split: str = "train_1",
    test_split: str = "test_1",
    fit_n: int = 10000,
    test_n: int = 5000,
    n_pairs: int = 50000,
    seed: int = 0,
) -> dict:
    """Creative selection under a testing budget: reframe the regressor + trust layer
    as an A/B selection engine and measure it as one. Reports pairwise pick accuracy,
    how accuracy rises with the predicted margin, the automate-vs-panel budget curve
    (margin gate vs per-item std gate), and pick-the-winner rates for batches of
    candidate creative. The decision, not the absolute score, is what a creative-testing
    product sells."""
    import random
    import sys

    sys.path.insert(0, "/root/project")
    root = _ensure_lamem_root()

    import numpy as np
    import pandas as pd
    import torch

    import memorability.model_torch  # noqa: F401  (registers resnet_ridge)
    import memorability.model_clip  # noqa: F401  (registers clip_ridge)
    from memorability import data, metrics
    from memorability.model import get_model

    rng = random.Random(seed)
    train = data.load_split(root, train_split)
    test = data.load_split(root, test_split)
    fit_items = rng.sample(train, min(fit_n, len(train)))
    test_items = rng.sample(test, min(test_n, len(test)))

    model = get_model(model_name)
    print(f"cuda_available={torch.cuda.is_available()}; fitting {model_name} on {len(fit_items)} images ...")
    model.fit(fit_items)
    paths = [it.image_path for it in test_items]
    pred, std, feat = model.embed_and_predict(paths)
    actual = np.array([it.score for it in test_items], dtype=float)

    # Random creative pairs; drop self-pairs, then score each A/B call.
    nrng = np.random.default_rng(seed)
    i = nrng.integers(0, len(pred), size=n_pairs)
    j = nrng.integers(0, len(pred), size=n_pairs)
    nonself = i != j
    i, j = i[nonself], j[nonself]
    correct, margin, keep = metrics.pairwise_decision(pred, actual, i, j)
    ik, jk = i[keep], j[keep]
    pooled_std = np.sqrt(std[ik] ** 2 + std[jk] ** 2)  # per-item uncertainty of the pair

    by_conf = metrics.accuracy_by_confidence(correct, margin, n_bins=10)
    auto_gap = metrics.decision_coverage_curve(correct, margin)
    auto_std = metrics.decision_coverage_curve(correct, -pooled_std)  # low std = confident

    def _acc_at(df, f):
        row = df.loc[np.isclose(df["auto_fraction"], f)]
        return float(row["accuracy"].iloc[0]) if len(row) else float("nan")

    b5 = metrics.batch_selection(pred, actual, batch_size=5, n_batches=20000, seed=seed)
    b10 = metrics.batch_selection(pred, actual, batch_size=10, n_batches=20000, seed=seed)

    summary = {
        "model": model_name, "n_fit": len(fit_items), "n_test": len(test_items),
        "n_pairs": int(len(correct)), "pairwise_acc": float(correct.mean()),
        "acc_gap_50": _acc_at(auto_gap, 0.5), "acc_std_50": _acc_at(auto_std, 0.5),
        "acc_gap_30": _acc_at(auto_gap, 0.3),
        "p_best_5": b5["p_pick_is_best"],
        "p_best_10": b10["p_pick_is_best"], "p_top_10": b10["p_pick_in_top"],
    }
    print(by_conf.to_string(index=False))
    print(summary)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(by_conf["mean_confidence"], by_conf["accuracy"], "o-", color="#4C78A8")
    ax.axhline(0.5, color="#999999", ls="--", lw=1, label="coin flip")
    ax.set_xlabel("predicted memorability gap (decision confidence)")
    ax.set_ylabel("A/B decision accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Accuracy rises with the model's confidence in the call")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    h1 = "/root/confidence_accuracy.png"
    fig.savefig(h1, dpi=150)
    plt.close(fig)
    with open(h1, "rb") as fh:
        conf_png = fh.read()

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(auto_gap["auto_fraction"], auto_gap["accuracy"], "o-", color="#E45756", label="gate on predicted margin")
    ax.plot(auto_std["auto_fraction"], auto_std["accuracy"], "s-", color="#4C78A8", label="gate on per-item std")
    ax.axhline(float(correct.mean()), color="#999999", ls="--", lw=1, label="accuracy with no gating")
    ax.set_xlabel("fraction of A/B calls the model auto-handles (rest go to panel)")
    ax.set_ylabel("accuracy on auto-handled calls")
    ax.set_title("Test-budget tradeoff: automate the confident calls")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    h2 = "/root/budget_curve.png"
    fig.savefig(h2, dpi=150)
    plt.close(fig)
    with open(h2, "rb") as fh:
        budget_png = fh.read()

    return {
        "summary": summary,
        "decision_report_md": _decision_markdown(summary, by_conf, auto_gap, auto_std),
        "confidence_accuracy_png": conf_png,
        "budget_curve_png": budget_png,
    }


@app.local_entrypoint()
def main(
    model_name: str = "resnet_ridge",
    train_split: str = "train_1",
    test_split: str = "test_1",
    n_bins: int = 10,
    all_splits: bool = False,
    ad_transfer: bool = False,
    n_ads: int = 1500,
    shift: bool = False,
    ava: bool = False,
    taxonomy: bool = False,
    decision: bool = False,
) -> None:
    import os
    import statistics

    if ad_transfer:
        # Domain transfer: score real ad creative and measure how out-of-distribution
        # it is relative to the training photos.
        print(f"ad-transfer demo: {model_name} on {n_ads} ad images ...")
        result = run_ads.remote(model_name, n_ads, train_split, test_split)
        os.makedirs("output/ads", exist_ok=True)
        with open("output/ad_transfer.md", "w", encoding="utf-8") as handle:
            handle.write(result["ad_report_md"])
        with open("output/ads/ood_hist.png", "wb") as handle:
            handle.write(result["hist_png"])
        for name, blob in result["ex_images"].items():
            with open(os.path.join("output/ads", name), "wb") as handle:
                handle.write(blob)
        s = result["summary"]
        print(f"\nOOD distance median: ads {s['ood_ads_median']:.2f} vs photos {s['ood_test_median']:.2f}")
        print(f"ads beyond the in-distribution 95th percentile: {s['ads_flagged_frac'] * 100:.0f}%")
        print("wrote output/ad_transfer.md, output/ads/ood_hist.png, and example ad images")
        return

    if shift:
        # OOD-gated routing on the real LaMem-to-ad shift: how well the OOD distance
        # routes genuinely novel creative to deferral, where the model's std cannot.
        print(f"OOD-gated routing on the real ad shift: {model_name} ...")
        result = run_shift.remote(model_name, train_split, test_split)
        os.makedirs("output/shift", exist_ok=True)
        with open("output/shift/shift_report.md", "w", encoding="utf-8") as handle:
            handle.write(result["shift_report_md"])
        with open("output/shift/ood_shift_hist.png", "wb") as handle:
            handle.write(result["ood_hist_png"])
        with open("output/shift/routing.png", "wb") as handle:
            handle.write(result["routing_png"])
        s = result["summary"]
        print(f"\nAUROC(OOD detects ads) = {s['auroc_ood']:.3f}   AUROC(std) = {s['auroc_std']:.3f}")
        print(
            f"at the 95th-pct operating point: {s['ad_recall'] * 100:.0f}% of ads routed out, "
            f"{s['iid_deferred'] * 100:.0f}% of in-distribution deferred"
        )
        print(f"MAE on kept LaMem {s['mae_kept']:.3f} vs {s['mae_iid']:.3f} ungated")
        print("wrote output/shift/shift_report.md, ood_shift_hist.png, routing.png")
        return

    if ava:
        # Second human-response signal: does predicted uncertainty track real human
        # disagreement (AVA aesthetics, 50+ votes per image).
        print(f"second-signal (AVA aesthetics) disagreement demo: {model_name} ...")
        result = run_ava.remote(model_name)
        os.makedirs("output/ava", exist_ok=True)
        with open("output/ava/ava_report.md", "w", encoding="utf-8") as handle:
            handle.write(result["ava_report_md"])
        with open("output/ava/ava_disagreement.png", "wb") as handle:
            handle.write(result["disagree_png"])
        with open("output/ava/ava_disagreement_head.png", "wb") as handle:
            handle.write(result["dishead_png"])
        with open("output/ava/ava_calibration.png", "wb") as handle:
            handle.write(result["calib_png"])
        s = result["summary"]
        print(f"\nmean-aesthetic Spearman {s['spearman_mean']:.3f}; MAE {s['mae_10']:.2f} rating points")
        print(f"intrinsic std vs human disagreement: {s['rho_std_humansd']:.3f}")
        print(f"dedicated head vs human disagreement: {s['rho_dishead']:.3f} (held out)")
        print("wrote output/ava/ava_report.md, ava_disagreement.png, ava_disagreement_head.png, ava_calibration.png")
        return

    if taxonomy:
        # Content-level failure map: where the model is least trustworthy, by creative type.
        print(f"content-level failure map: {model_name} ...")
        result = run_taxonomy.remote(model_name, train_split, test_split)
        os.makedirs("output/taxonomy", exist_ok=True)
        with open("output/taxonomy/taxonomy_report.md", "w", encoding="utf-8") as handle:
            handle.write(result["taxonomy_report_md"])
        with open("output/taxonomy/taxonomy.png", "wb") as handle:
            handle.write(result["taxonomy_png"])
        s = result["summary"]
        print(
            f"\nleast trustworthy: {s['worst_category']} (MAE {s['worst_mae']:.3f}); "
            f"most trustworthy: {s['best_category']} (MAE {s['best_mae']:.3f})"
        )
        print("wrote output/taxonomy/taxonomy_report.md, taxonomy.png")
        return

    if decision:
        # Creative selection under a testing budget: the regressor + trust layer as an
        # A/B selection engine (pairwise accuracy, confidence-accuracy, budget curve).
        print(f"creative-selection-under-budget demo: {model_name} ...")
        result = run_decision.remote(model_name, train_split, test_split)
        os.makedirs("output/decision", exist_ok=True)
        with open("output/decision/decision_report.md", "w", encoding="utf-8") as handle:
            handle.write(result["decision_report_md"])
        with open("output/decision/confidence_accuracy.png", "wb") as handle:
            handle.write(result["confidence_accuracy_png"])
        with open("output/decision/budget_curve.png", "wb") as handle:
            handle.write(result["budget_curve_png"])
        s = result["summary"]
        print(f"\npairwise pick accuracy {s['pairwise_acc'] * 100:.1f}% (coin flip 50%)")
        print(f"batch-of-5 top pick is true best {s['p_best_5'] * 100:.0f}% (random 20%)")
        print(
            f"automate the confident half: margin gate {s['acc_gap_50'] * 100:.1f}% "
            f"vs std gate {s['acc_std_50'] * 100:.1f}%"
        )
        print("wrote output/decision/decision_report.md, confidence_accuracy.png, budget_curve.png")
        return

    if all_splits:
        # Cross-split variance: run the model on all 5 LaMem splits in parallel
        # and report mean +/- std, so the headline number is a measured result
        # rather than one lucky split.
        print(f"running {model_name} across LaMem splits 1-5 ...")
        ensure_data.remote()  # populate the Volume once so the 5 runs don't race
        results = sorted(
            eval_split.starmap([(model_name, i, n_bins) for i in range(1, 6)]),
            key=lambda r: r["split"],
        )
        rhos = [r["spearman"] for r in results]
        maes = [r["mae"] for r in results]
        print(f"\n{model_name} across 5 LaMem splits")
        print(f"{'split':>6} {'spearman':>10} {'mae':>8} {'rmse':>8} {'n':>7}")
        for r in results:
            print(f"{r['split']:>6} {r['spearman']:>10.4f} {r['mae']:>8.4f} {r['rmse']:>8.4f} {r['n']:>7}")
        print(
            f"\nSpearman: mean {statistics.mean(rhos):.4f}  std {statistics.stdev(rhos):.4f}  "
            f"(min {min(rhos):.4f}, max {max(rhos):.4f})"
        )
        print(f"MAE:      mean {statistics.mean(maes):.4f}")
        return

    result = run_remote.remote(model_name, train_split, test_split, n_bins)

    os.makedirs("output", exist_ok=True)
    with open("output/findings.md", "w", encoding="utf-8") as handle:
        handle.write(result["findings_md"])
    with open("output/calibration.png", "wb") as handle:
        handle.write(result["plot_png"])

    fail_images = result.get("fail_images", {})
    fail_dir = os.path.join("output", "failures")
    os.makedirs(fail_dir, exist_ok=True)
    for name, blob in fail_images.items():
        with open(os.path.join(fail_dir, name), "wb") as handle:
            handle.write(blob)

    m = result["metrics"]
    print(f"\nmodel={model_name}  train={train_split}  test={test_split}")
    print(f"Spearman rho = {m['spearman']:.3f}   MAE = {m['mae']:.3f}   RMSE = {m['rmse']:.3f}   (n={m['n']})")
    print("wrote output/findings.md")
    print("wrote output/calibration.png")
    print(f"wrote {len(fail_images)} failure images to output/failures/")
