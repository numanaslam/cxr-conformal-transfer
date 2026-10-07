"""Trained linear concept heads (ADR-0005) — the default frozen-encoder concept predictor.

Heads are logistic-regression weights per concept fit on frozen train embeddings
(`scripts/s09_train_heads.py` -> `processed/concept_heads.json`): {concept: {coef:[D], intercept}}.
"""
from __future__ import annotations
import json
import math


def load_heads(path):
    with open(path) as f:
        return json.load(f)


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def predict_one(emb_row, head):
    """Present-probability for one concept from one embedding row (pure-python; testable).
    Applies optional Platt calibration `head["platt"]=[a,b]` on the logit."""
    z = float(head["intercept"]) + sum(float(e) * float(c) for e, c in zip(emb_row, head["coef"]))
    if "platt" in head:
        a, b = head["platt"]
        z = float(a) * z + float(b)
    return sigmoid(z)


def predict_matrix(emb, heads, order):
    """[N, K] present-probabilities. emb: numpy [N, D]; order: concept names. Applies optional
    per-head Platt calibration. Concepts absent from `heads` get a constant 0.5 column."""
    import numpy as np
    cols = []
    for c in order:
        h = heads.get(c)
        if h is None:
            cols.append(np.full(emb.shape[0], 0.5))
            continue
        z = emb @ np.asarray(h["coef"], dtype="float64") + float(h["intercept"])
        if "platt" in h:
            a, b = h["platt"]
            z = float(a) * z + float(b)
        cols.append(1.0 / (1.0 + np.exp(-z)))
    return np.stack(cols, axis=1)
