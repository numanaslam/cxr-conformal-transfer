"""[analysis] Conditional-risk comparator, G1 workload and score baselines, prevalence-matched Indiana.

Run on the three sites of s35 (cached embeddings only), primary predictor fixed.

  A  A rule that bounds the CONDITIONAL error among accepted studies, P(error | accept) <= r with
     probability >= 1 - delta, in the manner of selection with guaranteed risk (Geifman and El-Yaniv
     2017): each margin of a fixed grid (0 to 0.5 in steps of 0.005) is tested with a one-sided
     Clopper-Pearson bound at level delta / 101 (Bonferroni), and the most permissive margin that
     passes is kept (none passes: defer every study).
     Fit on the same half as lambda. Reported for r = 0.10 and 0.15, delta = 0.05, on the primary
     split and over R calibration redraws (share of draws whose realised conditional error > r).
  B  What G1 defers at each deployment site: total deferral, deferral among normal, in-vocabulary
     abnormal (the G1 rate) and out-of-vocabulary abnormal studies.
  C  G1 with simpler scores, each calibrated by the same split-conformal rule at alpha = 0.05 on the
     in-vocabulary abnormal calibration studies: the valve statistic s(x), the unanchored gap
     1 - max_k p_k, the abnormality A(x), the mean binary entropy of the eight finding probabilities,
     and the binary uncertainty 1 - |2A - 1|. For every pair: G1 rate, total deferral, deferral of
     normal and of out-of-vocabulary abnormal studies.
  D  Prevalence-matched Indiana: Indiana calibration and test sets subsampled to the NIH abnormal
     prevalence by dropping abnormal studies at random (R draws); G2 for every pair involving Indiana.

    python -m scripts.s37_conditional_and_g1 --config configs\\vindr.yaml ^
        --sites vindr=configs\\vindr.yaml iuxray=configs\\iuxray_nihtax.yaml --redraws 200
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ovcbmr.config import load_config
from scripts.s31_clean_protocol import _tau, _lam
from scripts.s35_multisite import _primary, _pairs
from scripts.s36_revision2 import _sub, _cat, _summ, _rates

RISKS = (0.10, 0.15)
DELTA = 0.05


# ------------------------------------------------------------------ A: conditional-risk rule
def _cp_upper(k, m, delta):
    """One-sided Clopper-Pearson upper bound for k errors in m accepted studies."""
    from scipy.stats import beta
    if m == 0:
        return 1.0
    return 1.0 if k >= m else float(beta.ppf(1 - delta, k + 1, m - k))


GRID = [round(0.005 * i, 3) for i in range(101)]      # fixed margins 0, 0.005, ..., 0.5 (data-independent)


def _lam_cond(D, tau, r, delta=DELTA):
    """Most permissive margin on a FIXED grid whose Clopper-Pearson upper bound on P(error | accept)
    is <= r at level delta / len(GRID) (Bonferroni). With probability >= 1 - delta every margin kept
    has conditional error <= r; if none qualifies, defer every study."""
    import numpy as np
    conf = np.abs(D["A"] - tau)
    err = ((D["A"] >= tau).astype(int) != D["y"])
    for lam in GRID:                                   # increasing margin = decreasing acceptance
        acc = conf >= lam
        if _cp_upper(int((acc & err).sum()), int(acc.sum()), delta / len(GRID)) <= r:
            return float(lam)
    return float("inf")


def _cond(D, tau, lam):
    import numpy as np
    acc = np.abs(D["A"] - tau) >= lam
    err = ((D["A"] >= tau).astype(int) != D["y"])
    a = float(acc.mean())
    return {"accept": a, "err_given_accept": float((acc & err).sum() / acc.sum()) if acc.sum() else 0.0,
            "cer": float((acc & err).mean())}


# ------------------------------------------------------------------ B, C: G1 scores
def _scores(D):
    import numpy as np
    P = D["P"]
    abn = [j for j in range(P.shape[1]) if j != D["nidx"]]
    pa = np.clip(P[:, abn], 1e-6, 1 - 1e-6)
    return {"valve": D["s"], "gap": 1.0 - pa.max(axis=1), "abnormality": D["A"],
            "entropy": (-(pa * np.log(pa) + (1 - pa) * np.log(1 - pa))).mean(axis=1),
            "binary_uncertainty": 1.0 - np.abs(2 * D["A"] - 1)}


def _qhat(x, alpha):
    import numpy as np
    s = np.sort(x)
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(s[k - 1]) if 0 < k <= n else float("inf")


def _defer_profile(D, sc, q):
    import numpy as np
    d = sc > q
    oov = (D["y"] == 1) & ~D["known"]
    f = lambda m: float(d[m].mean()) if m.sum() else float("nan")   # noqa: E731
    return {"g1": f(D["known"]), "total": float(d.mean()), "normal": f(D["y"] == 0), "oov": f(oov)}


# ------------------------------------------------------------------ D: prevalence matching
def _match_prev(D, prev, rng):
    import numpy as np
    pos, neg = np.where(D["y"] == 1)[0], np.where(D["y"] == 0)[0]
    k = int(round(prev * len(neg) / (1 - prev)))
    keep = np.concatenate([neg, rng.choice(pos, size=min(k, len(pos)), replace=False)])
    return _sub(D, np.sort(keep))


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vindr.yaml")
    ap.add_argument("--sites", nargs="+", default=["vindr=configs/vindr.yaml", "iuxray=configs/iuxray_nihtax.yaml"])
    ap.add_argument("--redraws", type=int, default=200)
    ap.add_argument("--out", default="s37_conditional_and_g1.json")
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config)
    sites = dict(s.split("=", 1) for s in args.sites)
    a1 = float(getattr(cfg.conformal, "alpha_flag", 0.05))
    a2 = float(getattr(cfg.conformal, "alpha_risk", 0.10))
    seed = int(getattr(cfg, "seed", 0))
    prim, S = _primary(cfg, sites, a1, a2, 50, seed)
    names = prim["sites"]
    with open(os.path.join(cfg.cross_hospital.nih_processed, "meta.json")) as f:
        nidx = json.load(f)["vocabulary"].index("Normal")
    for s in names:
        for part in ("q", "fit", "tune", "te"):
            S[s][part]["nidx"] = nidx
    out = {"alpha_flag": a1, "alpha_risk": a2, "delta": DELTA}

    # ---------------- A on the primary split
    print(f"\n=== A. conditional-risk rule, P(error | accept) <= r with prob. >= {1 - DELTA} (primary split) ===")
    lamc = {s: {r: _lam_cond(S[s]["fit"], S[s]["tau"], r) for r in RISKS} for s in names}
    A = {}
    for c, d in _pairs(names):
        row = {f"r{r}": _cond(S[d]["te"], S[c]["tau"], lamc[c][r]) for r in RISKS}
        row["crc"] = _cond(S[d]["te"], S[c]["tau"], S[c]["lam"])
        A[f"{c}->{d}"] = row
        print(f"  {c+' -> '+d:<18} CRC: acc {row['crc']['accept']:.2f} err|acc {row['crc']['err_given_accept']:.3f} | "
              + " | ".join(f"cond r={r}: acc {row[f'r{r}']['accept']:.2f} err|acc {row[f'r{r}']['err_given_accept']:.3f}"
                           for r in RISKS))
    out["conditional_primary"] = A

    # ---------------- A over redraws (heads fixed)
    rng = np.random.RandomState(seed)
    pools, ncal = {}, {}
    for s in names:
        cal = S[s]["fit"] if s == "nih" else S[s]["q"]
        pools[s] = _cat({k: v for k, v in cal.items() if k != "nidx"}, {k: v for k, v in S[s]["te"].items() if k != "nidx"})
        ncal[s] = len(cal["y"])
    rows = []
    for _ in range(args.redraws):
        fits, tests = {}, {}
        for s in names:
            P = pools[s]
            p = rng.permutation(len(P["y"]))
            cal, te = _sub(P, p[: ncal[s]]), _sub(P, p[ncal[s]:])
            if s == "nih":
                tune, fit = S["nih"]["tune"], cal
            else:
                h = rng.permutation(len(cal["y"]))
                tune, fit = _sub(cal, h[: len(h) // 2]), _sub(cal, h[len(h) // 2:])
            tau = _tau(tune)
            fits[s] = {"tau": tau, **{r: _lam_cond(fit, tau, r) for r in RISKS}}
            tests[s] = te
        row = {}
        for c, d in _pairs(names):
            for r in RISKS:
                g = _cond(tests[d], fits[c]["tau"], fits[c][r])
                row[f"acc_r{r}|{c}->{d}"], row[f"eca_r{r}|{c}->{d}"] = g["accept"], g["err_given_accept"]
        rows.append(row)
    red = {}
    for k in rows[0]:
        r = float(k.split("|")[0].split("_r")[1])
        red[k] = _summ([x[k] for x in rows], r if k.startswith("eca") else None)
    out["conditional_redraws"] = red
    print(f"\n=== A. conditional-risk rule over {args.redraws} redraws: mean acceptance, mean err|acc, share of draws err|acc > r ===")
    for c, d in _pairs(names):
        print(f"  {c+' -> '+d:<18} " + " | ".join(
            f"r={r}: acc {red[f'acc_r{r}|{c}->{d}']['mean']:.2f} err|acc {red[f'eca_r{r}|{c}->{d}']['mean']:.3f} "
            f"above {red[f'eca_r{r}|{c}->{d}']['frac_above']:.2f}" for r in RISKS))

    # ---------------- B, C: G1 deferral profile for every score
    print("\n=== B/C. G1 at alpha = 0.05 for five scores: G1 rate, total deferral, deferral of normal and out-of-vocabulary abnormal ===")
    G = {}
    qs = {s: {k: _qhat(v[S[s]["q"]["known"]], a1) for k, v in _scores(S[s]["q"]).items()} for s in names}
    for c, d in _pairs(names):
        sc = _scores(S[d]["te"])
        G[f"{c}->{d}"] = {k: _defer_profile(S[d]["te"], sc[k], qs[c][k]) for k in sc}
    out["g1_scores"] = G
    for k in ("valve", "gap", "abnormality", "entropy", "binary_uncertainty"):
        print(f"  [{k}]")
        for c, d in _pairs(names):
            g = G[f"{c}->{d}"][k]
            print(f"    {c+' -> '+d:<18} G1 {g['g1']:.3f}  total {g['total']:.3f}  normal {g['normal']:.3f}  oov {g['oov']:.3f}")

    # ---------------- D: prevalence-matched Indiana
    if "iuxray" in names:
        target = float(S["nih"]["te"]["y"].mean())
        print(f"\n=== D. Indiana subsampled to abnormal prevalence {target:.2f}, {args.redraws} draws: mean G2 acc / cer ===")
        rng = np.random.RandomState(seed + 11)
        acc = {}
        for _ in range(args.redraws):
            cal = _match_prev({k: v for k, v in S["iuxray"]["q"].items() if k != "nidx"}, target, rng)
            te = _match_prev({k: v for k, v in S["iuxray"]["te"].items() if k != "nidx"}, target, rng)
            h = rng.permutation(len(cal["y"]))
            tune, fit = _sub(cal, h[: len(h) // 2]), _sub(cal, h[len(h) // 2:])
            tau = _tau(tune)
            fiu = {"tau": tau, "lam": _lam(fit, tau, a2, "symmetric")}
            tests = {s: (te if s == "iuxray" else S[s]["te"]) for s in names}
            fits = {s: ({"tau": S[s]["tau"], "lam": S[s]["lam"]} if s != "iuxray" else fiu) for s in names}
            for c, d in _pairs(names):
                if "iuxray" not in (c, d):
                    continue
                g = _rates(tests[d], fits[c]["tau"], fits[c]["lam"])
                acc.setdefault(f"{c}->{d}", []).append((g["accept"], g["cer"]))
        D = {}
        for k, v in acc.items():
            v = np.array(v)
            D[k] = {"accept": _summ(v[:, 0]), "cer": _summ(v[:, 1], a2)}
            print(f"  {k:<18} acc {v[:, 0].mean():.2f}  cer {v[:, 1].mean():.3f} [{np.percentile(v[:, 1], 2.5):.3f}, "
                  f"{np.percentile(v[:, 1], 97.5):.3f}]  above {np.mean(v[:, 1] > a2):.2f}")
        out["prevalence_matched_indiana"] = {"target_prevalence": target, "pairs": D}

    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, args.out)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[s37] wrote {op}")


if __name__ == "__main__":
    main()
