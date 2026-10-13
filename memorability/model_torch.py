"""A reference torch model: ResNet18 features + a ridge-regression head, plus a
heteroscedastic variance head for per-item uncertainty.

This is a STARTING POINT, not the finished model. It gives the Modal run
something real to execute and a reasonable LaMem baseline to beat. The mean head
is a linear probe on frozen features; the variance head is a second ridge fit to
the log squared-residual, so the model reports where it cannot be trusted and the
interval-coverage part of the report lights up. The work that is yours, and what
you'll actually talk about, is improving the mean model: a stronger backbone
(CLIP) or real fine-tuning instead of a linear probe.

Importing this module registers the model as "resnet_ridge", so run.py can use
it with no edit to model.py. torch is imported lazily inside the methods, so the
fixture pipeline still runs on a machine with no torch installed.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from memorability.data import Item
from memorability.model import _REGISTRY, MemorabilityModel


class ResNetRidge(MemorabilityModel):
    def __init__(
        self,
        batch: int = 64,
        ridge_lambda: float = 1.0,
        var_lambda: float = 10.0,
        var_floor: float = 1e-4,
        cal_frac: float = 0.2,
    ):
        self.batch = batch
        self.lam = ridge_lambda
        self.var_lambda = var_lambda  # regularization for the variance head
        self.var_floor = var_floor  # noise floor so log(residual^2) stays finite
        self.cal_frac = cal_frac  # held-out fraction for conformal calibration
        self._backbone = None
        self._preprocess = None
        self._device = None

    def _ensure_backbone(self) -> None:
        if self._backbone is not None:
            return
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = ResNet18_Weights.DEFAULT
        net = resnet18(weights=weights)
        net.fc = torch.nn.Identity()  # expose the 512-d penultimate features
        net.eval().to(self._device)
        self._backbone = net
        self._preprocess = weights.transforms()

    def _embed(self, image_paths: Sequence[str]) -> np.ndarray:
        import torch

        self._ensure_backbone()
        feats = []
        with torch.no_grad():
            for i in range(0, len(image_paths), self.batch):
                chunk = image_paths[i : i + self.batch]
                batch = torch.stack(
                    [self._preprocess(Image.open(p).convert("RGB")) for p in chunk]
                ).to(self._device)
                feats.append(self._backbone(batch).float().cpu().numpy())
        return np.concatenate(feats, axis=0)

    def fit(self, train_items: Sequence[Item]) -> None:
        x = self._embed([it.image_path for it in train_items])
        y = np.array([it.score for it in train_items], dtype=float)
        self._fit_from_features(x, y)

    def _fit_from_features(self, x, y) -> None:
        """Fit the mean, variance, and conformal heads from precomputed features.

        `fit` embeds then calls this. Exposing it lets a caller embed a pool
        once and refit on an arbitrary subset without re-paying the embedding
        cost, which the distribution-shift experiment relies on (hold out a
        cluster, refit on the rest, all on features already in hand)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self._mu = x.mean(axis=0)
        self._sd = x.std(axis=0) + 1e-8
        xa = self._design_from_features(x)  # standardize + intercept column
        gram = xa.T @ xa

        reg = self.lam * np.eye(xa.shape[1])
        reg[-1, -1] = 0.0  # do not regularize the intercept
        self._w = np.linalg.solve(gram + reg, xa.T @ y)

        # Heteroscedastic variance head: a second ridge that predicts the log of
        # the squared residual from the same features. It learns which regions of
        # feature space the model cannot predict well and reports a larger std
        # there, so the model can say when not to trust it. The mean head (and so
        # the Spearman number) is untouched; this only adds a per-item std.
        resid = y - xa @ self._w
        z = np.log(resid ** 2 + self.var_floor)
        reg_v = self.var_lambda * np.eye(xa.shape[1])
        reg_v[-1, -1] = 0.0
        self._v = np.linalg.solve(gram + reg_v, xa.T @ z)

        # Split-conformal calibration. The raw heteroscedastic std is informative
        # but overconfident (its intervals cover below nominal). Fit a shadow model
        # on a subset, measure the held-out normalized residuals |y-yhat|/std, and
        # keep them; their conformal quantile gives interval widths with a finite-
        # sample, distribution-free coverage guarantee. Point predictions still use
        # the full-data head above, so the Spearman number is unchanged.
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        n_cal = max(1, int(self.cal_frac * len(y)))
        cal, fit = idx[:n_cal], idx[n_cal:]
        gram_fit = xa[fit].T @ xa[fit]
        w_cal = np.linalg.solve(gram_fit + reg, xa[fit].T @ y[fit])
        resid_fit = y[fit] - xa[fit] @ w_cal
        z_fit = np.log(resid_fit ** 2 + self.var_floor)
        v_cal = np.linalg.solve(gram_fit + reg_v, xa[fit].T @ z_fit)
        std_cal = np.sqrt(np.exp(np.clip(xa[cal] @ v_cal, np.log(1e-6), np.log(1.0))))
        pred_cal = np.clip(xa[cal] @ w_cal, 0.0, 1.0)
        self._cal_scores = np.sort(np.abs(y[cal] - pred_cal) / std_cal)

    def conformal_quantile(self, coverage: float) -> float:
        """Split-conformal interval half-width multiplier for a target coverage:
        the calibrated interval is pred +/- conformal_quantile(cov) * std."""
        s = self._cal_scores
        n = len(s)
        level = min(1.0, np.ceil((n + 1) * coverage) / n)
        return float(np.quantile(s, level, method="higher"))

    def _design_from_features(self, x) -> np.ndarray:
        xs = (np.asarray(x, dtype=float) - self._mu) / self._sd
        return np.hstack([xs, np.ones((len(xs), 1))])  # intercept column

    def _design(self, image_paths: Sequence[str]) -> np.ndarray:
        return self._design_from_features(self._embed(image_paths))

    def predict_from_features(self, x):
        """(mean, std) from precomputed features, using the already-fit heads.
        The counterpart to _fit_from_features: score a pool that was embedded
        once without re-embedding."""
        xa = self._design_from_features(x)
        mean = np.clip(xa @ self._w, 0.0, 1.0)
        log_var = np.clip(xa @ self._v, np.log(1e-6), np.log(1.0))
        std = np.sqrt(np.exp(log_var))
        return mean, std

    def predict(self, image_paths: Sequence[str]) -> np.ndarray:
        return np.clip(self._design(image_paths) @ self._w, 0.0, 1.0)

    def predict_with_uncertainty(self, image_paths: Sequence[str]):
        return self.predict_from_features(self._embed(image_paths))

    def embed_and_predict(self, image_paths: Sequence[str]):
        """Embed once and return (mean, std, raw_features). The raw features feed
        out-of-distribution scoring: how far new creative sits from the training
        distribution."""
        x = self._embed(image_paths)
        mean, std = self.predict_from_features(x)
        return mean, std, x


_REGISTRY["resnet_ridge"] = ResNetRidge
