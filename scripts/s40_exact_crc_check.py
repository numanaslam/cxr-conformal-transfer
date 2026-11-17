"""[check] Exact check of Proposition 3 against the implemented _lam: for a fixed multiset of n+1 studies,
average the test loss over all n+1 choices of the held-out study (exact expectation under exchangeability
given the multiset) and check it is <= alpha; also check the deterministic step L(lam_hat) <= L(lam')."""
import sys, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from scripts.s31_clean_protocol import _lam
rng = np.random.default_rng(1)
tau, alpha = 0.5, 0.10
worst, n_cases, det_fail = 0.0, 0, 0
for trial in range(4000):
    n = int(rng.integers(9, 40))
    y = (rng.random(n + 1) < rng.uniform(0.2, 0.6)).astype(int)
    A = np.where(y == 1, rng.beta(3, 2.5, n + 1), rng.beta(2, 3.5, n + 1))
    dec = [1, 2, 3][trial % 3]                       # heavy ties: 1-3 decimals
    A = np.round(A, dec)
    m = np.abs(A - tau); err = ((A >= tau).astype(int) != y).astype(float)
    M = sorted(set([0.0] + m.tolist() + [np.inf]))
    R = lambda lam: float(((m >= lam) * err).mean())
    lam_p = min(l for l in M if R(l) <= alpha)        # oracle lambda' on all n+1
    losses = []
    for t in range(n + 1):                           # every study as the test point
        idx = np.r_[0:t, t + 1:n + 1]
        lam_h = _lam({"A": A[idx], "y": y[idx]}, tau, alpha, "symmetric")
        L = float(m[t] >= lam_h and err[t] == 1)
        Lp = float(m[t] >= lam_p and err[t] == 1)
        det_fail += L > Lp
        losses.append(L)
    worst = max(worst, float(np.mean(losses))); n_cases += 1
print(f"multisets {n_cases}; max exact E[L_test] = {worst:.4f} (alpha {alpha}); deterministic violations {det_fail}")
