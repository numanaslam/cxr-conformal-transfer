"""Per-concept Platt calibration (protocol §5).

At CLIP's temperature, sigmoid(cosine/tau) saturates and one global temperature cannot
calibrate K concepts with different base rates. We fit a per-concept scale+bias
    l_k = w_k * s_k + b_k ,  p_k = sigmoid(l_k)
on the calibration split (cosine sim s_k vs. binary label y_k). `apply_platt` is pure-python
(testable); `fit_platt` lazily uses scikit-learn.
"""
from __future__ import annotations
import json
import math


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def apply_platt(sim: float, w: float, b: float) -> float:
    """Calibrated present-probability p_k = sigmoid(w*sim + b)."""
    return sigmoid(w * sim + b)


def fit_platt(sims, labels, max_iter=200):
    """Fit (w, b) for one concept via 1-D logistic regression. Lazy scikit-learn."""
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    X = np.asarray(sims, dtype="float64").reshape(-1, 1)
    y = np.asarray(labels, dtype="int64")
    if len(set(y.tolist())) < 2:  # degenerate concept in this split
        return 1.0, 0.0
    clf = LogisticRegression(max_iter=max_iter, C=1.0)
    clf.fit(X, y)
    return float(clf.coef_[0][0]), float(clf.intercept_[0])


def fit_bank(sims_by_concept, labels_by_concept, max_iter=200):
    """{concept: (w,b)} over all concepts. Inputs are {concept: [sims]}, {concept: [labels]}."""
    params = {}
    for c in sims_by_concept:
        params[c] = fit_platt(sims_by_concept[c], labels_by_concept[c], max_iter=max_iter)
    return params


def save_bank(params, path):
    with open(path, "w") as f:
        json.dump({c: list(wb) for c, wb in params.items()}, f, indent=2)


def load_bank(path):
    with open(path) as f:
        return {c: tuple(wb) for c, wb in json.load(f).items()}
