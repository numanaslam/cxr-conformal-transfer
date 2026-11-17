"""[check] Marginal validity of the implemented G1 and G2 threshold rules on synthetic exchangeable data.

Calls the same code as the experiments: `split_conformal_threshold` for G1 and `_lam` (the 201-quantile
grid with the (n R + B)/(n + 1) <= alpha condition and the lambda = inf fallback) for G2. For each
calibration size n, with continuous scores and with heavily tied scores (rounded to two decimals),
it draws R calibration sets, fits the threshold, and measures the realised rate on a fresh test set
from the same distribution. Validity means that the mean over draws is at most the target, up to
Monte Carlo error. No data are needed.

    python scripts/s38_validity_simulation.py --reps 4000
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ovcbmr.ood.conformal import split_conformal_threshold
from scripts.s31_clean_protocol import _lam

SIZES = (20, 50, 100, 500, 2000)


def _draw(rng, n, ties):
    """Abnormality score A in (0,1), label y with P(y=1 | A) rising in A; valve-like score s."""
    import numpy as np
    y = rng.random(n) < 0.35
    A = np.where(y, rng.beta(4, 2.5, n), rng.beta(2, 4.5, n))
    s = A * (1 - rng.beta(3, 2, n))
    if ties:
        A, s = np.round(A, 2), np.round(s, 2)
    return {"A": A, "y": y.astype(int), "s": s}


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--ntest", type=int, default=5000)
    ap.add_argument("--out", default="experiments/final/s38_validity_simulation.json")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    tau, a1, a2 = 0.5, 0.05, 0.10
    out = {}
    print(f"{'scores':<10}{'n':>6}  {'G1 mean':>8} {'(MC se)':>8}  {'G2 mean':>8} {'(MC se)':>8}  {'defer-all':>9}  {'G2 accept':>9}")
    for ties in (False, True):
        for n in SIZES:
            g1, g2, acc, inf = [], [], [], 0
            for _ in range(a.reps):
                cal, te = _draw(rng, n, ties), _draw(rng, a.ntest, ties)
                q = split_conformal_threshold(cal["s"][cal["y"] == 1].tolist(), a1)
                g1.append(float((te["s"][te["y"] == 1] > q).mean()))
                lam = _lam(cal, tau, a2, "symmetric")
                inf += lam == float("inf")
                ok = np.abs(te["A"] - tau) >= lam
                err = (te["A"] >= tau).astype(int) != te["y"]
                g2.append(float((ok & err).mean()))
                acc.append(float(ok.mean()))
            g1, g2 = np.array(g1), np.array(g2)
            key = f"{'tied' if ties else 'continuous'}_n{n}"
            out[key] = {"g1_mean": float(g1.mean()), "g1_se": float(g1.std(ddof=1) / np.sqrt(len(g1))),
                        "g2_mean": float(g2.mean()), "g2_se": float(g2.std(ddof=1) / np.sqrt(len(g2))),
                        "defer_all_share": inf / a.reps, "g2_accept": float(np.mean(acc))}
            o = out[key]
            print(f"{'tied' if ties else 'continuous':<10}{n:>6}  {o['g1_mean']:8.4f} {o['g1_se']:8.4f}  "
                  f"{o['g2_mean']:8.4f} {o['g2_se']:8.4f}  {o['defer_all_share']:9.3f}  {o['g2_accept']:9.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"tau": tau, "alpha_g1": a1, "alpha_g2": a2, "reps": a.reps, "ntest": a.ntest, "results": out}, f, indent=2)
    print(f"[s38] wrote {a.out}")


if __name__ == "__main__":
    main()
