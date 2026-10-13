"""Reference model: a frozen CLIP (ViT-B/32) image encoder + the same ridge head,
heteroscedastic uncertainty, and split-conformal calibration as resnet_ridge.
Registers as "clip_ridge".

This is the controlled test of the failure-taxonomy hypothesis: that the ResNet18
ceiling comes from borrowing ImageNet class features, which encode "what object is
this" rather than "how distinctive is this". CLIP features carry more of the
semantic and stylistic signal that drives memorability, so under the identical
head clip_ridge should beat resnet_ridge. Everything except the backbone is
inherited, so the comparison is clean.

open_clip is imported lazily inside the methods, so importing this module costs
nothing until the model is actually used (the fixture pipeline and resnet_ridge
still run with no open_clip installed).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from memorability.model import _REGISTRY
from memorability.model_torch import ResNetRidge


class CLIPRidge(ResNetRidge):
    """Same ridge + variance head + conformal logic as ResNetRidge; only the
    feature extractor changes to a frozen CLIP image encoder."""

    MODEL = "ViT-B-32"
    PRETRAINED = "laion2b_s34b_b79k"

    def _ensure_backbone(self) -> None:
        if self._backbone is not None:
            return
        import open_clip
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.MODEL, pretrained=self.PRETRAINED
        )
        model.eval().to(self._device)
        self._backbone = model
        self._preprocess = preprocess

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
                out = self._backbone.encode_image(batch)
                out = out / out.norm(dim=-1, keepdim=True)  # unit-normalize CLIP features
                feats.append(out.float().cpu().numpy())
        return np.concatenate(feats, axis=0)


_REGISTRY["clip_ridge"] = CLIPRidge
