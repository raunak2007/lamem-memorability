"""Load LaMem (or the offline fixtures) in the dataset's native split format."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_ROOT = os.path.join(_HERE, "fixtures", "lamem")


@dataclass
class Item:
    image_path: str
    score: float


def load_split(root: str, split: str) -> List[Item]:
    """Read `<root>/splits/<split>.txt`; each line is `<filename> <score>`.

    Real LaMem ships splits named train_1..train_5, val_1.., test_1..; the
    fixtures use train_tiny / test_tiny. Image paths resolve to
    `<root>/images/<filename>`, so nothing downstream changes between them.
    """
    split_path = os.path.join(root, "splits", f"{split}.txt")
    images_dir = os.path.join(root, "images")
    items: List[Item] = []
    with open(split_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            items.append(Item(os.path.join(images_dir, parts[0]), float(parts[1])))
    return items
