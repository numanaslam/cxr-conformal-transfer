"""[one image per patient] One image per patient at NIH, so that the NIH guarantees are exact.

NIH has 3.6 images per patient, so image-level calibration and test scores are exchangeable only
between patients, not between images, and the image-level G1 and G2 statements at NIH are
approximate. The remedy cited in the paper (Dunn, Wasserman and Ramdas, JASA 2023) is to keep one
randomly chosen image per patient in the NIH calibration and test splits. This script reruns the
clean protocol of s31 that way, on the primary (seed 42) split:

    NIH val            -> Platt scalars and tau (all images; not part of the conformal statement)
    NIH calib, 1/pat   -> G1 quantile, G2 lambda
    NIH test,  1/pat   -> every rate measured at NIH
    VinDr              -> unchanged (one image per study already)

The choice of image is repeated over --draws random draws. Reported per setting: the seed-42
all-images value (as in s31), and the mean, sd, min and max over the draws, for the G1
false-deferral rate and the G2 (symmetric) acceptance and confident-error rate.

It also reports the VinDr AUROC of a single merged Nodule/Mass score,
max(p_Nodule, p_Mass), against the single native VinDr Nodule/Mass label.

Cached embeddings, CPU only, about a minute per draw.

    python -m scripts.s34_one_image_per_patient --config configs/vindr.yaml --draws 20
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from anywhere

from ovcbmr.config import load_config
from ovcbmr.concept.heads import load_heads
from ovcbmr.eval.metrics import auroc
from scripts.s31_clean_protocol import (_split, _take, _halves, _refit_platt, _derive,
                                        _q, _ffr, _tau, _lam, _g2, one_per_patient_mask)

SETTINGS = ("NIH on-site", "NIH->VinDr", "VinDr on-site", "VinDr->NIH")


def _one_per_patient(D, draw):
    """Keep, for every patient, the image with the smallest hash of (draw, patient, position)."""
    return _take(D, one_per_patient_mask(D["pat"], draw))


def _run(heads, vocab, nidx, abn_idx, n_tune, n_fit, n_te, v_tune, v_fit, v_cal, v_te, a1, a2, nb, seed):
    dv = lambda D: _derive(D, heads, vocab, nidx, abn_idx)   # noqa: E731
    N_tune, N_fit, N_te = dv(n_tune), dv(n_fit), dv(n_te)
    V_tune, V_fit, V_q, V_te = dv(v_tune), dv(v_fit), dv(v_cal), dv(v_te)
    qN, qV = _q(N_fit, a1), _q(V_q, a1)
    g1 = {"NIH on-site": _ffr(N_te, qN, nb, seed), "NIH->VinDr": _ffr(V_te, qN, nb, seed),
          "VinDr on-site": _ffr(V_te, qV, nb, seed), "VinDr->NIH": _ffr(N_te, qV, nb, seed)}
    tN, tV = _tau(N_tune), _tau(V_tune)
    lN, lV = _lam(N_fit, tN, a2, "symmetric"), _lam(V_fit, tV, a2, "symmetric")
    g2 = {"NIH on-site": _g2(N_te, tN, lN, "symmetric", a2, nb, seed),
          "NIH->VinDr": _g2(V_te, tN, lN, "symmetric", a2, nb, seed),
          "VinDr on-site": _g2(V_te, tV, lV, "symmetric", a2, nb, seed),
          "VinDr->NIH": _g2(N_te, tV, lV, "symmetric", a2, nb, seed)}
    return {s: {"g1": g1[s]["ffr"], "g1_n": g1[s]["n"], "g1_ci": g1[s]["ci"],
                "accept": g2[s]["accept"], "cer": g2[s]["controlled"], "cer_ci": g2[s]["ci"]}
            for s in SETTINGS}


def _summ(xs):
    import numpy as np
    a = np.asarray(xs, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vindr.yaml")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--nboot", type=int, default=1000, help="bootstrap for the all-images row and draw 0")
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config)
    ch = cfg.cross_hospital
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    nidx = vocab.index("Normal")
    abn_idx = [i for i, c in enumerate(vocab) if c != "Normal"]
    a1 = float(getattr(cfg.conformal, "alpha_flag", 0.05))
    a2 = float(getattr(cfg.conformal, "alpha_risk", 0.10))
    seed = int(getattr(cfg, "seed", 0))

    n_cal = _split(ch.nih_embeddings, ch.nih_processed, "calib")
    n_val = _split(ch.nih_embeddings, ch.nih_processed, "val")
    n_te = _split(ch.nih_embeddings, ch.nih_processed, "test")
    v_cal = _split(cfg.paths.embeddings, cfg.paths.processed, "calib")
    v_te = _split(cfg.paths.embeddings, cfg.paths.processed, "test")
    if any(x is None for x in (n_cal, n_val, n_te, v_cal, v_te)):
        print("[s34] missing an embeddings split (need NIH calib/val/test and VinDr calib/test)."); return
    print(f"[s34] NIH images: calib {len(n_cal['pat'])}, test {len(n_te['pat'])}; "
          f"patients: calib {len(set(n_cal['pat']))}, test {len(set(n_te['pat']))}")

    clean_heads = _refit_platt(heads, vocab, n_val)
    v_tune, v_fit = _halves(v_cal, "vindr-clean")
    common = dict(v_tune=v_tune, v_fit=v_fit, v_cal=v_cal, v_te=v_te, a1=a1, a2=a2, seed=seed)

    allimg = _run(clean_heads, vocab, nidx, abn_idx, n_val, n_cal, n_te, nb=args.nboot, **common)
    draws = []
    for d in range(args.draws):
        r = _run(clean_heads, vocab, nidx, abn_idx, n_val, _one_per_patient(n_cal, d),
                 _one_per_patient(n_te, d), nb=(args.nboot if d == 0 else 1), **common)
        draws.append(r)
        print(f"  draw {d:>2}: " + "  ".join(f"{s} G1 {r[s]['g1']:.3f} CER {r[s]['cer']:.3f}"
                                            for s in SETTINGS))

    out = {"alpha_flag": a1, "alpha_risk": a2, "draws": args.draws, "all_images": allimg,
           "one_per_patient_draw0": draws[0], "one_per_patient": {}}
    print(f"\n=== one image per patient at NIH, {args.draws} draws (clean protocol, seed-42 split) ===")
    print(f"  {'setting':<15}{'quantity':<18}{'all images':>11}   one per patient: mean (sd) [min, max]")
    for s in SETTINGS:
        out["one_per_patient"][s] = {}
        for k, name in (("g1", "G1 false deferral"), ("accept", "G2 acceptance"), ("cer", "G2 conf-error")):
            sm = _summ([r[s][k] for r in draws])
            out["one_per_patient"][s][k] = sm
            print(f"  {s:<15}{name:<18}{allimg[s][k]:>11.3f}   {sm['mean']:.3f} ({sm['sd']:.3f}) "
                  f"[{sm['min']:.3f}, {sm['max']:.3f}]")
    print(f"  NIH G1 denominators, one per patient (draw 0): test n={draws[0]['NIH on-site']['g1_n']}")

    # R2.7: one merged Nodule/Mass score against the single native VinDr class
    if "Nodule" in vocab and "Mass" in vocab:
        import numpy as np
        V = _derive(v_te, clean_heads, vocab, nidx, abn_idx)
        jn, jm = vocab.index("Nodule"), vocab.index("Mass")
        y = ((V["Y"][:, jn] + V["Y"][:, jm]) > 0).astype(int)
        res = {"merged_max": auroc(np.maximum(V["P"][:, jn], V["P"][:, jm]).tolist(), y.tolist()),
               "nodule_only": auroc(V["P"][:, jn].tolist(), y.tolist()),
               "mass_only": auroc(V["P"][:, jm].tolist(), y.tolist()), "n_pos": int(y.sum())}
        out["vindr_nodule_mass"] = res
        print(f"\n=== VinDr Nodule/Mass as one class (n+={res['n_pos']}) ===\n"
              f"  AUROC  max(Nodule, Mass) {res['merged_max']:.3f} | Nodule head {res['nodule_only']:.3f}"
              f" | Mass head {res['mass_only']:.3f}")

    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, "s34_one_image_per_patient.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[s34] wrote {op}")


if __name__ == "__main__":
    main()
