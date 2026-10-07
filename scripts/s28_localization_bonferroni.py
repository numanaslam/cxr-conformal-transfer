"""[localization multiplicity] The referee notes that eight per-concept pointing-game tests
against eight nulls need a multiplicity correction, and that the random-point null is weak. This
script reads the existing s15_localization.json and re-reports each concept's significance under
a Bonferroni correction for the eight tests, alongside the 95% result. It rescales each 95%
bootstrap half-width to the Bonferroni one-sided level (alpha/2m) with a normal approximation, so
no re-bootstrapping of the embeddings is needed. Runs locally from the results JSON, no GPU.

    python -m scripts.s28_localization_bonferroni --config configs/vindr.yaml
    # or: python scripts/s28_localization_bonferroni.py --results experiments/results/s15_localization.json
"""
from __future__ import annotations
import argparse
import json
import os


def _z(p):
    """Standard-normal quantile via the Moro/Beasley-Springer rational approximation (math-only)."""
    import math
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--config", default=None)  # accepted for uniformity; results path is enough
    args, _ = ap.parse_known_args()
    path = args.results or "experiments/results/s15_localization.json"
    if not os.path.exists(path):
        print(f"[s28] {path} not found."); return
    d = json.load(open(path))
    pc = d["per_concept"]
    m = len(pc)
    z95 = _z(0.975)
    zbonf = _z(1 - 0.05 / (2 * m))               # two-sided Bonferroni -> one-sided alpha/2m
    scale = zbonf / z95
    print(f"[s28] {m} concepts, Bonferroni z={zbonf:.3f} vs 95% z={z95:.3f} (widen x{scale:.3f})\n")
    print(f"{'concept':18}{'hit':>6}{'null':>6}{'lo95':>7}{'loBonf':>8}{'95%':>6}{'Bonf':>6}")
    n_clear95 = n_clearb = 0
    rows = {}
    for c, v in pc.items():
        hit, null = v["hit_rate"], v["null"]
        lo, hi = v["ci"]
        half = hit - lo
        lo_b = hit - half * scale
        c95 = lo > null
        cb = lo_b > null
        n_clear95 += c95; n_clearb += cb
        print(f"{c:18}{hit:6.3f}{null:6.3f}{lo:7.3f}{lo_b:8.3f}{('yes' if c95 else 'no'):>6}{('yes' if cb else 'no'):>6}")
        rows[c] = {"hit": hit, "null": null, "lo95": lo, "lo_bonf": lo_b,
                   "clears_95": bool(c95), "clears_bonferroni": bool(cb), "n": v.get("n")}
    print(f"\n[s28] clears null: {n_clear95}/{m} at 95%, {n_clearb}/{m} under Bonferroni.")
    out = "experiments/results/s28_localization_bonferroni.json"
    with open(out, "w") as f:
        json.dump({"m": m, "z95": z95, "z_bonferroni": zbonf, "per_concept": rows}, f, indent=2)
    print(f"[s28] wrote {out}")


if __name__ == "__main__":
    main()
