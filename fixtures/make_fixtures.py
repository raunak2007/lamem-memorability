"""Generate a tiny synthetic stand-in for LaMem so the harness runs offline.

Real LaMem (http://memorability.csail.mit.edu) ships images under `images/` and
split files under `splits/` where each line is `<filename> <memorability_score>`.
This mirrors that exact layout with a handful of generated images whose score is
a monotonic function of brightness plus noise, so a trivial brightness model
shows real rank accuracy and the calibration plots are meaningful. Replace with
the real dataset for an actual run; nothing downstream changes.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "lamem")
IMAGES = os.path.join(ROOT, "images")
SPLITS = os.path.join(ROOT, "splits")

N = 44
RNG = np.random.default_rng(7)


def main() -> None:
    os.makedirs(IMAGES, exist_ok=True)
    os.makedirs(SPLITS, exist_ok=True)
    rows = []
    for i in range(1, N + 1):
        brightness = float(RNG.uniform(0.05, 0.95))
        # base image at that brightness with mild texture
        base = np.clip(
            brightness + RNG.normal(0, 0.05, size=(64, 64, 3)), 0, 1
        )
        Image.fromarray((base * 255).astype(np.uint8)).save(
            os.path.join(IMAGES, f"img_{i:04d}.png")
        )
        # synthetic "human" memorability: rises with brightness, with noise
        score = float(np.clip(0.30 + 0.50 * brightness + RNG.normal(0, 0.05), 0, 1))
        rows.append((f"img_{i:04d}.png", score))

    idx = RNG.permutation(N)
    train = [rows[i] for i in idx[:32]]
    test = [rows[i] for i in idx[32:]]
    _write(os.path.join(SPLITS, "train_tiny.txt"), train)
    _write(os.path.join(SPLITS, "test_tiny.txt"), test)
    print(f"wrote {N} images, {len(train)} train / {len(test)} test")


def _write(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for name, score in rows:
            handle.write(f"{name} {score:.5f}\n")


if __name__ == "__main__":
    main()
