"""Evaluation metrics (protocol §7). Pure-python cores so they are testable with no numpy:
AUROC (Mann-Whitney), FPR@95TPR, risk-coverage / AURC, bootstrap CI over patients.
DeLong significance lazily uses scipy.

Two abstention axes are kept separate (protocol §1.3):
  * Axis A — unanchored detection: `auroc`, `fpr_at_tpr` on the unanchored labels.
  * Axis B — selective disease: `risk_coverage_curve`, `aurc`.
"""
from __future__ import annotations
import math
import random


def auroc(scores, labels):
    """AUROC via the Mann-Whitney U statistic with tie handling. labels in {0,1}."""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n = len(pairs)
    # average ranks (1..n) with ties
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    pos = sum(1 for _, y in pairs if y == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")
    sum_pos_ranks = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)


def fpr_at_tpr(scores, labels, tpr=0.95):
    """FPR when the detection threshold achieves at least `tpr` recall on positives.
    Higher score = more likely positive (OOD/unanchored)."""
    pos = sorted((s for s, y in zip(scores, labels) if y == 1), reverse=True)
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    k = max(1, math.ceil(tpr * len(pos)))
    thr = pos[k - 1]                       # threshold that captures >= tpr of positives
    fp = sum(1 for s in neg if s >= thr)
    return fp / len(neg)


def risk_coverage_curve(losses, confidences):
    """Selective-prediction curve. Sort by DEScending confidence; accept a growing prefix.
    Returns list of (coverage, risk) with risk = mean loss over accepted."""
    order = sorted(zip(confidences, losses), key=lambda t: t[0], reverse=True)
    n = len(order)
    out, run = [], 0.0
    for i, (_, l) in enumerate(order, start=1):
        run += l
        out.append((i / n, run / i))
    return out


def aurc(losses, confidences):
    """Area under the risk-coverage curve (lower is better)."""
    curve = risk_coverage_curve(losses, confidences)
    if len(curve) < 2:
        return float("nan")
    area = 0.0
    for (c0, r0), (c1, r1) in zip(curve[:-1], curve[1:]):
        area += (c1 - c0) * (r0 + r1) / 2.0
    return area


def bootstrap_ci(values, statistic=None, n_boot=1000, alpha=0.05, seed=0, groups=None):
    """Percentile bootstrap CI. If `groups` (e.g. patient ids) is given, resample GROUPS
    (cluster bootstrap) so CIs respect patient-level correlation (protocol §7)."""
    statistic = statistic or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    values = list(values)
    if groups is not None:
        by_g = {}
        for v, g in zip(values, groups):
            by_g.setdefault(g, []).append(v)
        keys = list(by_g)
        stats = []
        for _ in range(n_boot):
            samp = []
            for _ in range(len(keys)):
                samp.extend(by_g[keys[rng.randrange(len(keys))]])
            stats.append(statistic(samp))
    else:
        n = len(values)
        stats = [statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)]
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return statistic(values), lo, hi


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE for a binary concept: |mean predicted prob - fraction positive| averaged over
    equal-width probability bins, weighted by bin count. Pure-python."""
    probs, labels = list(probs), list(labels)
    n = len(probs)
    if n == 0:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        bins[min(n_bins - 1, int(float(p) * n_bins))].append((float(p), float(y)))
    ece = 0.0
    for b in bins:
        if b:
            conf = sum(p for p, _ in b) / len(b)
            acc = sum(y for _, y in b) / len(b)
            ece += (len(b) / n) * abs(conf - acc)
    return ece


def delong_test(scores_a, scores_b, labels):
    """DeLong test for two correlated AUROCs (same test set). Lazy scipy; returns p-value.
    Use to test 'ours > strongest baseline' (protocol §7)."""
    from scipy import stats  # noqa: F401
    raise NotImplementedError("DeLong covariance on paired AUROCs — implement in s06 eval")
