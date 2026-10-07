"""OOD baseline suite (protocol §3), corrected for the MULTI-LABEL setting.

Pure-python scores (testable): `msp_margin`, `joint_energy`, `mcm_score`, `emptyset_reject`.
Embedding-based scores (`mahalanobis`, `knn`, `neglabel`) are runnable skeletons that lazily
use numpy and require fitted statistics / banks from cached embeddings.

Baselines to beat: MCM (Ming 2022), NegLabel (Jiang 2024), JointEnergy (Wang 2021), and the
class-level empty-set conformal rejection (2510.00773) — the nearest abstention prior art (B8b).
"""
from __future__ import annotations
import math


def msp_margin(logits):
    """B4 (corrected). Multi-label MSP must NOT be 1 - max_k p_k (that flags 'Normal' as OOD).
    Use distance to the per-concept decision boundary: uncertainty = -min_k |logit_k|.
    Higher = more OOD.
    """
    if not logits:
        return 0.0
    return -float(min(abs(l) for l in logits))


def joint_energy(logits):
    """B5. Multi-label energy = sum of per-label binary free energies (Wang et al. 2021),
    NOT a single logsumexp over concepts. Returns the ID-ness score (higher = in-distribution);
    negate for an OOD score.
    """
    return sum(math.log1p(math.exp(-abs(l))) + max(l, 0.0) for l in logits)


def mcm_score(sims, temp=0.01):
    """B2/B8-style. MCM (Ming 2022): negative max softmax over temperature-scaled cosine sims.
    Higher = more OOD. Pure-python softmax for testability.
    """
    if not sims:
        return 0.0
    zs = [s / temp for s in sims]
    m = max(zs)
    exps = [math.exp(z - m) for z in zs]
    denom = sum(exps)
    return -max(e / denom for e in exps)


def emptyset_reject(conformal_class_set):
    """B8b (2510.00773). Class-level conformal rejection: reject iff the prediction set is
    empty. `conformal_class_set` is the set/list of classes retained after thresholding.
    """
    return len(conformal_class_set) == 0


# --------------------------------------------------------------------------- #
# Embedding-based scores (numpy; fit on cached image embeddings).
# --------------------------------------------------------------------------- #
def fit_mahalanobis(train_emb, concept_masks, min_count=2):
    """B6. Per-concept means + a single shared Ledoit-Wolf precision on TRAIN embeddings.
    train_emb : np.ndarray [N, D]. concept_masks : {name: bool mask [N]} (positive for concept).
    Returns (names, means [K, D], precision [D, D]).
    """
    import numpy as np
    from sklearn.covariance import LedoitWolf
    names = [c for c, m in concept_masks.items() if int(m.sum()) >= min_count]
    means = np.stack([train_emb[concept_masks[c]].mean(axis=0) for c in names])
    centered = np.concatenate(
        [train_emb[concept_masks[c]] - means[i] for i, c in enumerate(names)], axis=0)
    cov = LedoitWolf().fit(centered).covariance_
    precision = np.linalg.pinv(cov)
    return names, means, precision


def mahalanobis_score(test_emb, means, precision, chunk=2048):
    """B6 score. OOD = min_c Mahalanobis distance to the class means. Higher = OOD. [N]."""
    import numpy as np
    out = np.empty(test_emb.shape[0], dtype="float64")
    for i in range(0, test_emb.shape[0], chunk):
        d = test_emb[i:i + chunk][:, None, :] - means[None, :, :]   # [c, K, D]
        md = np.einsum("ckd,de,cke->ck", d, precision, d)
        out[i:i + chunk] = np.sqrt(np.clip(md.min(axis=1), 0, None))
    return out


def knn_score_batch(test_emb, bank_emb, k=50, chunk=1024):
    """B7 score. kth-NN Euclidean distance to the (L2-normalized) train bank. Higher = OOD. [N].
    Uses cosine to get squared-Euclidean on unit vectors: ||q-b||^2 = 2 - 2 q.b .
    """
    import numpy as np
    out = np.empty(test_emb.shape[0], dtype="float64")
    for i in range(0, test_emb.shape[0], chunk):
        sims = test_emb[i:i + chunk] @ bank_emb.T                   # [c, B]
        d2 = np.clip(2.0 - 2.0 * sims, 0, None)
        kth = np.partition(d2, k, axis=1)[:, k]                     # kth smallest (0-indexed)
        out[i:i + chunk] = np.sqrt(kth)
    return out


def neglabel_score(image_emb, known_text_emb, negative_text_emb):
    """B8. NegLabel (Jiang 2024): OOD via alignment to negative/outlier text labels vs. known.
    Needs a cached negative-text bank; omitted from the s05 diagnostic set for now."""
    raise NotImplementedError("NegLabel negative-text alignment — add a negative-text bank")
