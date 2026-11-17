"""[re-splits] Repeated re-splits, repartitions and checks on the three sites of s35 (cached embeddings only).

  A  20 re-splits with retrained heads (s35 protocol): mean, sd, 2.5 and 97.5 percentiles, and the
     fraction of splits above the target, for every ordered pair of sites.
  B  Calibration redraws with the heads fixed: the primary predictor is kept and each site's pooled
     calibration and test studies are re-partitioned at random R times, so calibration and test
     sampling variability is separated from head retraining. (The bootstrap intervals of s35 cover
     test sampling only.)
  C  Sample size under the clean protocol at VinDr and Indiana: m local calibration studies drawn over
     the R redraws, tau and lambda fit on the two halves; on-site G2 and G1 against m.
  D  Stricter one-sided targets (accepted-and-missed rate at 0.10, 0.05, 0.02), primary split and redraws.
  E  Baselines without conformal calibration: a fixed confidence threshold (accept if the calibrated
     abnormality probability is >= 0.8 or <= 0.2) and an empirical margin without the finite-sample
     correction (smallest lambda with R_hat <= alpha).
  F  Workload per 1000 deployed studies: cleared as normal, flagged abnormal, deferred, missed abnormal.
  G  Measured shift per pair: abnormal prevalence, domain AUROC (from s35), and Kolmogorov-Smirnov
     distances of A(x) among normal and among abnormal studies and of s(x) among known abnormal studies.

    python -m scripts.s36_resplits --config configs\\vindr.yaml ^
        --sites vindr=configs\\vindr.yaml iuxray=configs\\iuxray_nihtax.yaml --seeds 20 --redraws 200
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ovcbmr.config import load_config
from scripts.s31_clean_protocol import _tau, _lam
from scripts.s35_multisite import _primary, _resplits, _pairs

ALPHAS_1S = (0.10, 0.05, 0.02)
SIZES = (100, 200, 400, 800, 1500)


# ------------------------------------------------------------------ fast rates (no bootstrap)
def _sub(D, idx):
    import numpy as np
    out = {}
    for k, v in D.items():
        out[k] = v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx]
    return out


def _cat(a, b):
    import numpy as np
    out = {}
    for k in a:
        out[k] = np.concatenate([a[k], b[k]]) if isinstance(a[k], np.ndarray) else list(a[k]) + list(b[k])
    return out


def _ffr(D, q):
    s = D["s"][D["known"]]
    return float((s > q).mean()) if len(s) else float("nan")


def _q(D, alpha):
    import math
    import numpy as np
    s = np.sort(D["s"][D["known"]])
    n = len(s)
    r = math.ceil((n + 1) * (1 - alpha))
    return float(s[r - 1]) if 0 < r <= n else float("inf")


def _rates(D, tau, lam):
    """acceptance, confident-error, accepted-and-missed, and per-study workload counts."""
    import numpy as np
    acc = np.abs(D["A"] - tau) >= lam
    yhat = (D["A"] >= tau).astype(int)
    y = D["y"]
    err, miss = yhat != y, (yhat == 0) & (y == 1)
    return {"accept": float(acc.mean()), "cer": float((acc & err).mean()), "miss": float((acc & miss).mean()),
            "cleared": float((acc & (yhat == 0)).mean()), "flagged": float((acc & (yhat == 1)).mean()),
            "deferred": float((~acc).mean()), "flagged_true": float((acc & (yhat == 1) & (y == 1)).mean())}


def _lam_naive(D, tau, alpha):
    """Smallest lambda with empirical confident-error rate <= alpha (no (n R + B)/(n + 1) correction)."""
    import numpy as np
    err = ((D["A"] >= tau).astype(int) != D["y"])
    conf = np.abs(D["A"] - tau)
    for lam in sorted(set(np.quantile(conf, np.linspace(0, 1, 201)).tolist())):
        if ((conf >= lam) & err).mean() <= alpha:
            return float(lam)
    return float(conf.max())


def _ks(a, b):
    from scipy.stats import ks_2samp
    return float(ks_2samp(a, b).statistic) if len(a) and len(b) else float("nan")


def _summ(v, target=None):
    import numpy as np
    v = np.asarray([x for x in v if x == x], dtype=float)
    out = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
           "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5)),
           "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
    if target is not None:
        out["frac_above"] = float((v > target).mean())
    return out


# ------------------------------------------------------------------ B, C, D: redraws with fixed heads
def _fit_site(cal, name, a1, a2, rng, nih_tune=None):
    """q_hat from the whole calibration set; tau and lambda from two random halves (NIH: tau from val)."""
    import numpy as np
    n = len(cal["y"])
    if nih_tune is not None:
        tune, fit = nih_tune, cal
    else:
        p = rng.permutation(n)
        tune, fit = _sub(cal, p[: n // 2]), _sub(cal, p[n // 2:])
    tau = _tau(tune)
    f = {"qhat": _q(cal, a1), "tau": tau, "lam": _lam(fit, tau, a2, "symmetric"),
         "lam_naive": _lam_naive(fit, tau, a2)}
    for al in ALPHAS_1S:
        f[f"lam1_{al}"] = _lam(fit, tau, al, "one-sided")
    return f


def _redraws(S, names, a1, a2, R, seed):
    import numpy as np
    rng = np.random.RandomState(seed)
    pools, ncal = {}, {}
    for s in names:
        cal = S[s]["fit"] if s == "nih" else S[s]["q"]
        pools[s] = _cat(cal, S[s]["te"])
        ncal[s] = len(cal["y"])
    rows = []
    for r in range(R):
        fits, tests = {}, {}
        for s in names:
            P = pools[s]
            p = rng.permutation(len(P["y"]))
            cal, te = _sub(P, p[: ncal[s]]), _sub(P, p[ncal[s]:])
            fits[s] = _fit_site(cal, s, a1, a2, rng, nih_tune=S["nih"]["tune"] if s == "nih" else None)
            tests[s] = te
        row = {}
        for c, d in _pairs(names):
            f, T = fits[c], tests[d]
            g = _rates(T, f["tau"], f["lam"])
            row[f"g1|{c}->{d}"] = _ffr(T, f["qhat"])
            row[f"acc|{c}->{d}"], row[f"cer|{c}->{d}"] = g["accept"], g["cer"]
            for al in ALPHAS_1S:
                g1s = _rates(T, f["tau"], f[f"lam1_{al}"])
                row[f"acc1s_{al}|{c}->{d}"], row[f"miss1s_{al}|{c}->{d}"] = g1s["accept"], g1s["miss"]
            gn = _rates(T, f["tau"], f["lam_naive"])
            row[f"accnaive|{c}->{d}"], row[f"cernaive|{c}->{d}"] = gn["accept"], gn["cer"]
        rows.append(row)
    return rows, pools, ncal


def _sample_size(pools, ncal, names, a1, a2, R, seed):
    import numpy as np
    rng = np.random.RandomState(seed + 7)
    out = {}
    for s in names:
        if s == "nih":
            continue
        P = pools[s]
        res = {}
        for m in [m for m in SIZES if m <= ncal[s]] + ["full"]:
            mm = ncal[s] if m == "full" else m
            acc, cer, g1, nk = [], [], [], []
            for _ in range(R):
                p = rng.permutation(len(P["y"]))
                pool_cal, te = _sub(P, p[: ncal[s]]), _sub(P, p[ncal[s]:])
                cal = _sub(pool_cal, rng.permutation(ncal[s])[:mm])
                f = _fit_site(cal, s, a1, a2, rng)
                g = _rates(te, f["tau"], f["lam"])
                acc.append(g["accept"]); cer.append(g["cer"])
                g1.append(_ffr(te, f["qhat"])); nk.append(int(cal["known"].sum()))
            res[str(m)] = {"accept": _summ(acc), "cer": _summ(cer, a2), "g1": _summ(g1, a1),
                           "n_known_mean": float(np.mean(nk))}
            print(f"[s36] sample size {s} m={m}: acc {np.mean(acc):.2f}  cer mean {np.mean(cer):.3f} "
                  f"p95 {np.percentile(cer, 95):.3f} above {np.mean(np.array(cer) > a2):.2f}  "
                  f"G1 mean {np.nanmean(g1):.3f} p95 {np.nanpercentile(g1, 95):.3f} (known {np.mean(nk):.0f})")
        out[s] = res
    return out


# ------------------------------------------------------------------ main
def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vindr.yaml")
    ap.add_argument("--sites", nargs="+", default=["vindr=configs/vindr.yaml", "iuxray=configs/iuxray_nihtax.yaml"])
    ap.add_argument("--seeds", type=int, default=20, help="number of re-splits with retrained heads (seeds 0..n-1)")
    ap.add_argument("--redraws", type=int, default=200)
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--out", default="s36_resplits.json")
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config)
    sites = dict(s.split("=", 1) for s in args.sites)
    a1 = float(getattr(cfg.conformal, "alpha_flag", 0.05))
    a2 = float(getattr(cfg.conformal, "alpha_risk", 0.10))
    seed = int(getattr(cfg, "seed", 0))
    out = {"alpha_flag": a1, "alpha_risk": a2}

    prim, S = _primary(cfg, sites, a1, a2, args.nboot, seed)
    names = prim["sites"]
    rng = np.random.RandomState(seed)

    # ---------------- D, E, F on the primary split
    print("\n=== primary split: one-sided targets, baselines, workload per 1000 deployed studies ===")
    prim2 = {}
    fits = {}
    for s in names:
        f = {"qhat": S[s]["qhat"], "tau": S[s]["tau"], "lam": S[s]["lam"],
             "lam_naive": _lam_naive(S[s]["fit"], S[s]["tau"], a2)}
        for al in ALPHAS_1S:
            f[f"lam1_{al}"] = _lam(S[s]["fit"], S[s]["tau"], al, "one-sided")
        fits[s] = f
    for c, d in _pairs(names):
        f, T = fits[c], S[d]["te"]
        r = {"symmetric": _rates(T, f["tau"], f["lam"]),
             "naive_margin": _rates(T, f["tau"], f["lam_naive"])}
        # fixed confidence: accept if A >= 0.8 or A <= 0.2, predicted class at tau
        acc = (T["A"] >= 0.8) | (T["A"] <= 0.2)
        yhat = (T["A"] >= f["tau"]).astype(int)
        r["fixed_confidence"] = {"accept": float(acc.mean()), "cer": float((acc & (yhat != T["y"])).mean()),
                                 "miss": float((acc & (yhat == 0) & (T["y"] == 1)).mean())}
        for al in ALPHAS_1S:
            r[f"onesided_{al}"] = _rates(T, f["tau"], f[f"lam1_{al}"])
        prim2[f"{c}->{d}"] = r
        w = r["symmetric"]
        print(f"  {c+' -> '+d:<18} G2 acc {w['accept']:.2f} cer {w['cer']:.3f} | per 1000: cleared {1000*w['cleared']:.0f} "
              f"flagged {1000*w['flagged']:.0f} deferred {1000*w['deferred']:.0f} missed {1000*w['miss']:.0f} | "
              + "  ".join(f"1s@{al}: acc {r[f'onesided_{al}']['accept']:.2f} miss {r[f'onesided_{al}']['miss']:.3f}"
                          for al in ALPHAS_1S)
              + f" | naive cer {r['naive_margin']['cer']:.3f} acc {r['naive_margin']['accept']:.2f}"
              + f" | fixed-0.8 cer {r['fixed_confidence']['cer']:.3f} acc {r['fixed_confidence']['accept']:.2f}")
    out["primary"] = prim2

    # ---------------- G shift per pair
    shift = {}
    for c, d in _pairs(names):
        if c >= d:
            continue
        A, B = S[c]["te"], S[d]["te"]
        shift[f"{c}|{d}"] = {
            "prev": [float(A["y"].mean()), float(B["y"].mean())],
            "ks_A_normal": _ks(A["A"][A["y"] == 0], B["A"][B["y"] == 0]),
            "ks_A_abnormal": _ks(A["A"][A["y"] == 1], B["A"][B["y"] == 1]),
            "ks_s_known": _ks(A["s"][A["known"]], B["s"][B["known"]]),
            "domain_auroc": None}
    from scripts.s35_multisite import _domain_auc
    dom = _domain_auc(S, names, seed)
    for k, v in dom.items():
        a, b = k.split("|")
        key = f"{a}|{b}" if f"{a}|{b}" in shift else f"{b}|{a}"
        shift[key]["domain_auroc"] = v
    out["shift"] = shift
    print("\n=== measured shift per pair of sites ===")
    for k, v in shift.items():
        print(f"  {k:<14} prevalence {v['prev'][0]:.2f}/{v['prev'][1]:.2f}  KS A|normal {v['ks_A_normal']:.3f}  "
              f"KS A|abnormal {v['ks_A_abnormal']:.3f}  KS s|known {v['ks_s_known']:.3f}  domain AUROC {v['domain_auroc']:.3f}")

    # ---------------- B, D, E over calibration redraws with fixed heads
    print(f"\n=== {args.redraws} calibration redraws, heads fixed: mean (sd) [2.5, 97.5] fraction above target ===")
    rows, pools, ncal = _redraws(S, names, a1, a2, args.redraws, seed)
    red = {}
    for k in rows[0]:
        kind = k.split("|")[0]
        tgt = a1 if kind == "g1" else a2 if kind in ("cer", "cernaive") else None
        if kind.startswith("miss1s_"):
            tgt = float(kind.split("_")[1])
        red[k] = _summ([r[k] for r in rows], tgt)
    out["redraws"] = red
    for c, d in _pairs(names):
        g, e = red[f"g1|{c}->{d}"], red[f"cer|{c}->{d}"]
        print(f"  {c+' -> '+d:<18} G1 {g['mean']:.3f} ({g['sd']:.3f}) [{g['p2.5']:.3f}, {g['p97.5']:.3f}] above {g['frac_above']:.2f}"
              f" | G2 {e['mean']:.3f} ({e['sd']:.3f}) [{e['p2.5']:.3f}, {e['p97.5']:.3f}] above {e['frac_above']:.2f}"
              f" | naive G2 {red[f'cernaive|{c}->{d}']['mean']:.3f} above {red[f'cernaive|{c}->{d}']['frac_above']:.2f}")

    # ---------------- C sample size
    print("\n=== sample size, clean protocol, over redraws ===")
    out["sample_size"] = _sample_size(pools, ncal, names, a1, a2, args.redraws, seed)

    # ---------------- A re-splits with retrained heads
    print(f"\n=== {args.seeds} re-splits with retrained heads ===")
    rs = _resplits(cfg, sites, list(range(args.seeds)), a1, a2)
    summ = {}
    for k in rs["per_seed"][0]:
        if k == "seed":
            continue
        kind = k.split("|")[0]
        summ[k] = _summ([r[k] for r in rs["per_seed"]], a1 if kind == "g1" else a2 if kind == "cer" else None)
    out["resplits"] = {"seeds": rs["seeds"], "per_seed": rs["per_seed"], "summary": summ}
    for c, d in _pairs(names):
        g, e, ac = summ[f"g1|{c}->{d}"], summ[f"cer|{c}->{d}"], summ[f"acc|{c}->{d}"]
        print(f"  {c+' -> '+d:<18} G1 {g['mean']:.3f} ({g['sd']:.3f}) [{g['p2.5']:.3f}, {g['p97.5']:.3f}] above {g['frac_above']:.2f}"
              f" | G2 {e['mean']:.3f} ({e['sd']:.3f}) [{e['p2.5']:.3f}, {e['p97.5']:.3f}] above {e['frac_above']:.2f}"
              f" | acc {ac['mean']:.2f}")

    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, args.out)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[s36] wrote {op}")


if __name__ == "__main__":
    main()
