"""Run a memorability model through the eval + calibration harness.

Offline on the bundled fixtures:
    python run.py --fixtures --model brightness

Real LaMem (download + extract first, see README):
    python run.py --root /path/to/lamem --train-split train_1 --test-split test_1 --model <yours>
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from memorability import data, report
from memorability.model import get_model

_HERE = os.path.dirname(os.path.abspath(__file__))


def run_eval(root, train_split, test_split, model_name, out_dir, n_bins=10) -> dict:
    train = data.load_split(root, train_split)
    test = data.load_split(root, test_split)

    model = get_model(model_name)
    model.fit(train)

    paths = [it.image_path for it in test]
    actual = [it.score for it in test]
    pred, std = model.predict_with_uncertainty(paths)
    pred = np.asarray(pred, dtype=float)

    conformal = None
    if std is not None and hasattr(model, "conformal_quantile"):
        conformal = {c: model.conformal_quantile(c) for c in (0.5, 0.8, 0.9, 0.95)}

    return report.write_report(
        paths, pred, actual, model_name, out_dir, std=std, n_bins=n_bins, conformal=conformal
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", action="store_true", help="use the bundled offline fixtures")
    parser.add_argument("--root", default=None, help="path to the extracted LaMem directory")
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--model", default="brightness", help="model name registered in model.get_model")
    parser.add_argument("--out-dir", default=os.path.join(_HERE, "output"))
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    if args.fixtures:
        root = data.FIXTURE_ROOT
        train_split = args.train_split or "train_tiny"
        test_split = args.test_split or "test_tiny"
    else:
        if not args.root:
            parser.error("pass --fixtures, or --root pointing at an extracted LaMem directory")
        root = args.root
        train_split = args.train_split or "train_1"
        test_split = args.test_split or "test_1"

    print(f"model={args.model}  root={root}  train={train_split}  test={test_split}")
    result = run_eval(root, train_split, test_split, args.model, args.out_dir, args.n_bins)
    m = result["metrics"]
    print(f"\nSpearman rho = {m['spearman']:.3f}   MAE = {m['mae']:.3f}   RMSE = {m['rmse']:.3f}   (n={m['n']})")
    print(f"wrote {result['findings_md']}")
    print(f"wrote {result['plot']}")


if __name__ == "__main__":
    main()
