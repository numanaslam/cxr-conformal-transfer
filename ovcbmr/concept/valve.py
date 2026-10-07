"""§2.4 test-time open-set decision — the contribution, assembled.

Given per-concept calibrated probabilities p_k (and optional evidential vacuity u_k) over
the active vocabulary V, plus an image abnormality signal A(x), decide whether the image
contains an UNANCHORED finding (nothing in V explains it) and abstain, else emit the anchored
concepts. The conformal threshold q_hat comes from `ovcbmr.ood.conformal` (G1).

Pure-python (no torch/numpy) so the decision logic is unit-testable.
"""
from __future__ import annotations

UNANCHORED = "UNANCHORED"
ANCHORED = "ANCHORED"


def anchoring(p, u=None):
    """Per-concept anchoring confidence a_k = p_k * (1 - u_k).

    p : list of calibrated present-probabilities over V.
    u : optional list of per-concept vacuities (evidential backbone); None -> treat as 0.
    """
    if u is None:
        return [float(pk) for pk in p]
    if len(u) != len(p):
        raise ValueError("p and u must have equal length")
    return [float(pk) * (1.0 - float(uk)) for pk, uk in zip(p, u)]


def unexplained_statistic(abnormality, anchoring_scores):
    """s(x) = A(x) * (1 - max_k a_k). High when abnormal yet nothing in V anchors it."""
    if not anchoring_scores:
        return float(abnormality)
    return float(abnormality) * (1.0 - max(anchoring_scores))


def image_unexplained(p, concept_names, normal_name="Normal"):
    """Per-image (abnormality A, unexplained s) from calibrated concept probs `p`.

    p            : per-concept present-probabilities aligned to `concept_names`.
    A(x)         : 1 - p(Normal) if a Normal/no-finding concept exists, else max_k p_k.
    anchoring    : a_k = p_k over the *abnormal* concepts (Normal excluded — a confident
                   'Normal' must not count as anchoring a finding).
    s(x)         : A * (1 - max_k a_k).  Pure-python; unit-testable.
    """
    idx = {c: i for i, c in enumerate(concept_names)}
    if normal_name in idx:
        A = 1.0 - float(p[idx[normal_name]])
        anch = [float(p[i]) for i, c in enumerate(concept_names) if c != normal_name]
    else:
        A = max((float(x) for x in p), default=0.0)
        anch = [float(x) for x in p]
    return A, unexplained_statistic(A, anch)


def decide(s, q_hat):
    """G1 rule: flag UNANCHORED iff the unexplained statistic exceeds the conformal q_hat."""
    return UNANCHORED if s > q_hat else ANCHORED


def valve_predict(p, abnormality, q_hat, concept_names=None, u=None, anchor_thresh=0.5):
    """End-to-end per-image decision.

    Returns a dict with the decision, the unexplained statistic, and — when anchored —
    the concepts that pass `anchor_thresh` (present + evidenced), sorted by confidence.
    """
    a = anchoring(p, u)
    s = unexplained_statistic(abnormality, a)
    decision = decide(s, q_hat)
    out = {"decision": decision, "unexplained": s, "abnormality": float(abnormality)}
    if decision == ANCHORED:
        idx = sorted(range(len(a)), key=lambda i: a[i], reverse=True)
        picks = [i for i in idx if a[i] >= anchor_thresh]
        if concept_names is not None:
            out["concepts"] = [(concept_names[i], round(a[i], 4)) for i in picks]
        else:
            out["concepts"] = [(i, round(a[i], 4)) for i in picks]
    return out
