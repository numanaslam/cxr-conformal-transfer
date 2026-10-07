"""Distribution-free guarantees for the safety valve (protocol §2.3).

G1 — split conformal, one-sided: calibrate a threshold q_hat on the unexplained statistic
     over KNOWN-abnormal calibration cases so that
         P( known-abnormal case falsely flagged UNANCHORED ) <= alpha   (finite-sample).
G2 — Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster 2023): choose an
     abstention threshold lambda_hat so the selective risk obeys  E[risk(lambda_hat)] <= alpha.

Pure-python (math + sorting) so the guarantees are unit-testable with no heavy stack.
The guarantee is deliberately ONE-SIDED (control false-flags on *known* findings; report
empirical recall on unseen) — a distribution-free bound on never-seen classes is impossible
without calibration data for them.
"""
from __future__ import annotations
import math


def split_conformal_threshold(cal_scores, alpha=0.05):
    """G1 threshold. Flag UNANCHORED iff score > q_hat.

    cal_scores : unexplained statistics on known-abnormal calibration cases (higher = more
                 likely to be (wrongly) flagged).
    Returns q_hat s.t. the false-flag rate on exchangeable known cases is <= alpha.
    """
    xs = sorted(float(s) for s in cal_scores)
    n = len(xs)
    if n == 0:
        return float("inf")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:                      # alpha too small for this calibration size
        return float("inf")
    return xs[rank - 1]


def empirical_flag_rate(scores, q_hat):
    """Fraction of `scores` strictly above q_hat (empirical false-flag rate on known cases)."""
    scores = list(scores)
    if not scores:
        return 0.0
    return sum(1 for s in scores if s > q_hat) / len(scores)


def coverage_vs_alpha(cal_scores, test_known_scores, alphas):
    """For each target alpha, return (alpha, q_hat, empirical_false_flag_rate_on_test).

    The headline guarantee figure: empirical rate should track <= alpha for ours, and drift
    above alpha for fixed-threshold baselines under multi-center shift (protocol §2.3, §8).
    """
    rows = []
    for a in alphas:
        q = split_conformal_threshold(cal_scores, a)
        rows.append((a, q, empirical_flag_rate(test_known_scores, q)))
    return rows


def crc_threshold(risks, lambdas, n, alpha=0.10, B=1.0):
    """G2 — Conformal Risk Control.

    risks   : empirical mean selective risks R_hat(lambda_i) on calibration, aligned to
              `lambdas` and assumed non-increasing as lambda relaxes (more coverage).
    n       : calibration set size.
    Returns (lambda_hat, index) — the smallest lambda whose CRC bound
        (n * R_hat(lambda) + B) / (n + 1) <= alpha
    holds; or (None, -1) if no lambda satisfies it.
    """
    for i, lam in enumerate(lambdas):
        bound = (n * float(risks[i]) + B) / (n + 1)
        if bound <= alpha:
            return lam, i
    return None, -1


def selective_risk(losses, keep_mask):
    """Mean loss over kept (non-abstained) examples; 0.0 if everything is abstained."""
    kept = [l for l, k in zip(losses, keep_mask) if k]
    if not kept:
        return 0.0
    return sum(kept) / len(kept)


# --------------------------------------------------------------------------- #
# Shift-robust conformal (weighted, Tibshirani-Barber-Candes-Ramdas 2019).
# Restores G1 coverage under covariate shift where the unweighted quantile fails.
# --------------------------------------------------------------------------- #
def weighted_quantile(scores, weights, level):
    """Smallest score whose normalized cumulative weight reaches `level` in [0, 1].
    weights >= 0; returns +inf if `level` exceeds the total normalized mass. Pure-python."""
    pairs = sorted(zip(scores, weights), key=lambda t: t[0])
    total = float(sum(weights))
    if total <= 0.0:
        return float("inf")
    cum = 0.0
    for s, w in pairs:
        cum += float(w)
        if cum / total >= level:
            return float(s)
    return float("inf")


def weighted_split_conformal_threshold(scores, weights, alpha=0.05):
    """Shift-robust G1 threshold: weighted (1-alpha) quantile of calibration scores with
    w_i proportional to dP_test/dP_cal(x_i). Restores coverage under covariate shift where the
    unweighted quantile breaks. (Finite-sample (n+1) test atom approximated; validate with the
    empirical coverage-vs-alpha curve.)"""
    return weighted_quantile(scores, weights, 1.0 - alpha)


def effective_sample_size(weights):
    """Kish effective sample size (sum w)^2 / sum(w^2). Small ESS relative to n signals weight
    degeneracy -> the weighted quantile is unstable and shift-robust coverage is unreliable."""
    s1 = float(sum(weights))
    s2 = float(sum(float(w) * float(w) for w in weights))
    return (s1 * s1 / s2) if s2 > 0.0 else 0.0


def estimate_shift_weights(cal_emb, ref_emb, C=0.1, max_iter=1000, clip_quantile=0.95):
    """Likelihood-ratio weights w(x) = dP_ref/dP_cal(x) for the calibration points, via a
    logistic domain classifier (cal=0, ref=1) on the frozen embeddings: w ∝ c(x)/(1-c(x)).

    Stabilized against effective-sample-size collapse when the two distributions over-separate:
    C regularizes the classifier (default strong), and weights above `clip_quantile` are capped.
    Returns a numpy array aligned to cal_emb rows.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([cal_emb, ref_emb], axis=0)
    y = np.concatenate([np.zeros(len(cal_emb)), np.ones(len(ref_emb))])
    clf = LogisticRegression(max_iter=max_iter, C=C)
    clf.fit(X, y)
    c = np.clip(clf.predict_proba(cal_emb)[:, 1], 1e-4, 1 - 1e-4)
    w = c / (1.0 - c)
    if clip_quantile and 0.0 < clip_quantile < 1.0:          # cap extreme weights
        w = np.minimum(w, float(np.quantile(w, clip_quantile)))
    return w
