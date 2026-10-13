"""The model interface you implement, plus two runnable baselines.

Your job is to replace / add to this file with a real memorability model (a
fine-tuned CNN, CLIP features with a regression head, whatever), then register it
in `get_model`. Everything else in the repo stays the same. Implement
`predict_with_uncertainty` too if your model can report a per-item standard
deviation; that unlocks the interval-coverage calibration in the report.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from memorability.data import Item


class MemorabilityModel:
    """Predict human memorability in [0, 1] for a list of image paths."""

    def fit(self, train_items: Sequence[Item]) -> None:
        """Train / calibrate on the train split. May be a no-op."""

    def predict(self, image_paths: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def predict_with_uncertainty(
        self, image_paths: Sequence[str]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (predictions, per-item std) or (predictions, None).

        Override to return a real std and the report will add interval-coverage
        calibration, which is the part Presonance most needs to trust.
        """
        return self.predict(image_paths), None


class MeanBaseline(MemorabilityModel):
    """Predicts the train-set mean for everything. The floor any model must beat."""

    def fit(self, train_items: Sequence[Item]) -> None:
        self.mean = float(np.mean([it.score for it in train_items])) if train_items else 0.5

    def predict(self, image_paths: Sequence[str]) -> np.ndarray:
        return np.full(len(image_paths), getattr(self, "mean", 0.5), dtype=float)


class BrightnessBaseline(MemorabilityModel):
    """A trivial worked example of the interface: mean pixel intensity, linearly
    mapped to memorability. Not good, just enough to show the pipeline end to end
    and give you a working reference for the method signatures."""

    def _feature(self, path: str) -> float:
        img = Image.open(path).convert("L").resize((64, 64))
        return float(np.asarray(img, dtype=float).mean()) / 255.0

    def fit(self, train_items: Sequence[Item]) -> None:
        x = np.array([self._feature(it.image_path) for it in train_items])
        y = np.array([it.score for it in train_items])
        design = np.vstack([x, np.ones_like(x)]).T
        self.w, self.b = np.linalg.lstsq(design, y, rcond=None)[0]
        self.resid_std = float(np.std(y - (self.w * x + self.b)))

    def predict(self, image_paths: Sequence[str]) -> np.ndarray:
        x = np.array([self._feature(p) for p in image_paths])
        return np.clip(self.w * x + self.b, 0.0, 1.0)

    def predict_with_uncertainty(self, image_paths):
        pred = self.predict(image_paths)
        # a crude constant std from training residuals, so coverage is exercised
        std = np.full_like(pred, getattr(self, "resid_std", 0.1))
        return pred, std


_REGISTRY = {
    "mean": MeanBaseline,
    "brightness": BrightnessBaseline,
    # TODO: register your real model here, e.g. "cnn": MyCNNModel
}


def get_model(name: str) -> MemorabilityModel:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
