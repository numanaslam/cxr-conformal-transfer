"""[clean protocol, both directions] Two things reviewers asked for, in one run.

(1) Disjoint calibration. In the paper protocol the Platt scalars, the G2 decision threshold tau,
    the G1 quantile and the G2 stringency are all fit on the SAME calibration split, so the
    calibration scores are not exactly exchangeable with test scores. The clean protocol is

        NIH   val    -> Platt scalars and tau          (NIH calib half A if no val cache exists)
        NIH   calib  -> G1 quantile q_hat, G2 lambda   (NIH calib half B in that fallback)
        VinDr calib half A -> tau ; half B -> lambda ; full VinDr calib -> q_hat
        Platt stays the NIH fit: the predictor is shared and nothing is fit on VinDr.

(2) Both deployment directions, for G1 and for G2 under the symmetric and the one-sided
    (missed-abnormal) loss: on-site at each hospital, NIH->VinDr, and VinDr->NIH.

Every row reports acceptance, the CRC-controlled joint rate with a clustered bootstrap interval,
and the conditional error and missed-abnormal rates among accepted cases. Mean abnormal-concept
ECE and Brier score are printed for both protocols. Cached embeddings, CPU, no GPU.

    python -m scripts.s31_clean_protocol --config configs/vindr.yaml
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from anywhere

from ovcbmr.config import load_config
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.concept.calibrate import fit_platt
from ovcbmr.ood.conformal import split_conformal_threshold, crc_threshold
from ovcbmr.eval.metrics import bootstrap_ci, expected_calibration_error


def _h01(key, salt=""):
    return int(hashlib.md5(f"{salt}:{key}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _split(emb_dir, proc, split):
    """{'E': [N,D], 'Y': [N,K] labels over the vocabulary, 'pat': [N]} or None if not cached."""
    import numpy as np
    import torch
    p = os.path.join(emb_dir, f"{split}_img.pt")
    if not os.path.exists(p):
        return None
    d = torch.load(p, weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    keep = [i for i, iid in enumerate(d["ids"]) if iid in items]
    ids = [d["ids"][i] for i in keep]
    return {"E": d["emb"].numpy()[keep],
            "Y": np.array([items[i]["labels_known"] for i in ids], dtype=int),
            "pat": [str(items[i].get("patient_id", i)) for i in ids]}


def _take(D, mask):
    return {"E": D["E"][mask], "Y": D["Y"][mask], "pat": [p for p, m in zip(D["pat"], mask) if m]}


def _halves(D, salt):
    """Deterministic split by patient into two disjoint halves."""
    import numpy as np
    m = np.array([_h01(p, salt) < 0.5 for p in D["pat"]])
    return _take(D, m), _take(D, ~m)


def _refit_platt(heads, vocab, D):
    """Copy of `heads` whose two Platt scalars per concept are refit on split D."""
    import numpy as np
    new = copy.deepcopy(heads)
    for j, c in enumerate(vocab):
        h = new.get(c)
        if h is None:
            continue
        y = D["Y"][:, j]
        if len(np.unique(y)) < 2:
            continue
        z0 = D["E"] @ np.asarray(h["coef"], dtype="float64") + float(h["intercept"])
        a, b = fit_platt(z0.tolist(), y.tolist())
        h["platt"] = [a, b]
    return new


def _derive(D, heads, vocab, nidx, abn_idx):
    P = predict_matrix(D["E"], heads, vocab)
    A = 1.0 - P[:, nidx]
    return {"P": P, "A": A, "s": A * (1.0 - P[:, abn_idx].max(axis=1)),
            "known": D["Y"][:, abn_idx].sum(axis=1) > 0,       # in-vocabulary known-abnormal
            "y": (D["Y"][:, nidx] != 1).astype(int),           # abnormal = not No-finding
            "Y": D["Y"], "pat": D["pat"]}


# ---------------------------------------------------------------- G1
def _q(D, alpha):
    return split_conformal_threshold(D["s"][D["known"]].tolist(), alpha)


def _ffr(D, q, nb, seed):
    k = D["known"]
    flags = (D["s"][k] > q).astype(float).tolist()
    pats = [p for p, kk in zip(D["pat"], k) if kk]
    r, lo, hi = bootstrap_ci(flags, groups=pats, n_boot=nb, seed=seed)
    return {"ffr": float(r), "ci": [lo, hi], "n": int(k.sum())}


# ---------------------------------------------------------------- G2
def _tau(D):
    import numpy as np
    A, y = D["A"], D["y"]
    P, N = int(y.sum()), int(len(y) - y.sum())
    ts = np.linspace(float(A.min()), float(A.max()), 201)
    ba = [0.5 * (((A >= t) & (y == 1)).sum() / max(P, 1) + ((A < t) & (y == 0)).sum() / max(N, 1))
          for t in ts]
    return float(ts[int(np.argmax(ba))])


def _losses(D, tau):
    yhat = (D["A"] >= tau).astype(int)
    err = (yhat != D["y"]).astype(float)
    miss = ((yhat == 0) & (D["y"] == 1)).astype(float)
    return err, miss


def _lam(D, tau, alpha, kind):
    import numpy as np
    err, miss = _losses(D, tau)
    loss = miss if kind == "one-sided" else err
    conf = np.abs(D["A"] - tau)
    lambdas = sorted(set(np.quantile(conf, np.linspace(0, 1, 201)).tolist()))
    risks = [float(((conf >= lam) * loss).mean()) for lam in lambdas]
    lam, _ = crc_threshold(risks, lambdas, n=len(conf), alpha=alpha, B=1.0)
    return float(lam if lam is not None else lambdas[-1])


def _g2(D, tau, lam, kind, alpha, nb, seed):
    import numpy as np
    err, miss = _losses(D, tau)
    acc = np.abs(D["A"] - tau) >= lam
    ctrl = miss if kind == "one-sided" else err
    j, lo, hi = bootstrap_ci((acc * ctrl).tolist(), groups=D["pat"], n_boot=nb, seed=seed)
    n_acc = max(int(acc.sum()), 1)
    return {"accept": float(acc.mean()), "controlled": float(j), "ci": [lo, hi],
            "bound_met": bool(j <= alpha + 1e-9),
            "err_given_accept": float(err[acc].sum() / n_acc),
            "miss_given_accept": float(miss[acc].sum() / n_acc),
            "joint_error": float((acc * err).mean()), "joint_miss": float((acc * miss).mean()),
            "tau": tau, "lam": lam}


def _calib(D, abn_idx):
    import numpy as np
    eces, briers = [], []
    for j in abn_idx:
        p, y = D["P"][:, j], D["Y"][:, j]
        eces.append(expected_calibration_error(p.tolist(), y.tolist()))
        briers.append(float(np.mean((p - y) ** 2)))
    return float(np.mean(eces)), float(np.mean(briers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vindr.yaml")
    ap.add_argument("--nboot", type=int, default=0)
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
    nb = args.nboot or int(getattr(getattr(cfg, "eval", object()), "bootstrap", 1000))
    seed = int(getattr(cfg, "seed", 0))

    n_cal = _split(ch.nih_embeddings, ch.nih_processed, "calib")
    n_val = _split(ch.nih_embeddings, ch.nih_processed, "val")
    n_te = _split(ch.nih_embeddings, ch.nih_processed, "test")
    v_cal = _split(cfg.paths.embeddings, cfg.paths.processed, "calib")
    v_te = _split(cfg.paths.embeddings, cfg.paths.processed, "test")
    if any(x is None for x in (n_cal, n_te, v_cal, v_te)):
        print("[s31] missing an embeddings split (need NIH calib/test and VinDr calib/test)."); return

    if n_val is not None:
        n_tune, n_fit, src = n_val, n_cal, "NIH val"
    else:
        n_tune, n_fit = _halves(n_cal, "nih-clean")
        src = "NIH calib half A (no val_img.pt cached; calib split in two)"
    v_tune, v_fit = _halves(v_cal, "vindr-clean")
    print(f"[s31] clean protocol: Platt + tau from {src} (n={len(n_tune['pat'])}); "
          f"q_hat + lambda from n={len(n_fit['pat'])}. VinDr tau/lambda halves "
          f"{len(v_tune['pat'])}/{len(v_fit['pat'])}. boot={nb}\n")

    protocols = {
        "paper": {"heads": heads, "n_tune": n_cal, "n_fit": n_cal, "v_tune": v_cal, "v_fit": v_cal},
        "clean": {"heads": _refit_platt(heads, vocab, n_tune),
                  "n_tune": n_tune, "n_fit": n_fit, "v_tune": v_tune, "v_fit": v_fit},
    }
    out = {"alpha_flag": a1, "alpha_risk": a2, "n_boot": nb, "clean_tune_source": src, "protocols": {}}
    settings = ("NIH on-site", "NIH->VinDr", "VinDr on-site", "VinDr->NIH")

    for pname, pr in protocols.items():
        dv = lambda D: _derive(D, pr["heads"], vocab, nidx, abn_idx)   # noqa: E731
        N_tune, N_fit, N_te = dv(pr["n_tune"]), dv(pr["n_fit"]), dv(n_te)
        V_tune, V_fit, V_q, V_te = dv(pr["v_tune"]), dv(pr["v_fit"]), dv(v_cal), dv(v_te)
        res = {"g1": {}, "g2": {}}

        qN, qV = _q(N_fit, a1), _q(V_q, a1)
        g1 = {"NIH on-site": _ffr(N_te, qN, nb, seed), "NIH->VinDr": _ffr(V_te, qN, nb, seed),
              "VinDr on-site": _ffr(V_te, qV, nb, seed), "VinDr->NIH": _ffr(N_te, qV, nb, seed)}
        res["g1"] = g1
        print(f"=== [{pname}] G1 in-vocabulary false deferral (target {a1}) ===")
        for s in settings:
            r = g1[s]
            print(f"  {s:<15}{r['ffr']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  n={r['n']}")

        tN, tV = _tau(N_tune), _tau(V_tune)
        for kind in ("symmetric", "one-sided"):
            lN, lV = _lam(N_fit, tN, a2, kind), _lam(V_fit, tV, a2, kind)
            g2 = {"NIH on-site": _g2(N_te, tN, lN, kind, a2, nb, seed),
                  "NIH->VinDr": _g2(V_te, tN, lN, kind, a2, nb, seed),
                  "VinDr on-site": _g2(V_te, tV, lV, kind, a2, nb, seed),
                  "VinDr->NIH": _g2(N_te, tV, lV, kind, a2, nb, seed)}
            res["g2"][kind] = g2
            what = "accept AND miss" if kind == "one-sided" else "accept AND error"
            print(f"\n=== [{pname}] G2 {kind} loss: controlled = P({what}) (target {a2}) ===")
            print(f"  {'setting':<15}{'accept':>7}{'controlled [CI]':>24}{'ok':>5}"
                  f"{'err|acc':>9}{'miss|acc':>10}")
            for s in settings:
                r = g2[s]
                print(f"  {s:<15}{r['accept']:>7.2f}{r['controlled']:>10.3f} "
                      f"[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]{('yes' if r['bound_met'] else 'NO'):>5}"
                      f"{r['err_given_accept']:>9.3f}{r['miss_given_accept']:>10.3f}")

        eN, bN = _calib(N_te, abn_idx)
        eV, bV = _calib(V_te, abn_idx)
        res["calibration"] = {"nih_ece": eN, "nih_brier": bN, "vindr_ece": eV, "vindr_brier": bV}
        print(f"\n=== [{pname}] mean abnormal-concept calibration (source Platt) ===")
        print(f"  NIH test   ECE {eN:.3f}  Brier {bN:.4f}\n  VinDr test ECE {eV:.3f}  Brier {bV:.4f}\n")
        out["protocols"][pname] = res

    p, c = out["protocols"]["paper"], out["protocols"]["clean"]
    print("=== paper protocol vs clean protocol (what moves when calibration data are disjoint) ===")
    for s in settings:
        print(f"  G1 {s:<15} {p['g1'][s]['ffr']:.3f} -> {c['g1'][s]['ffr']:.3f}")
    for s in settings:
        a, b = p["g2"]["symmetric"][s], c["g2"]["symmetric"][s]
        print(f"  G2 {s:<15} accept {a['accept']:.2f} -> {b['accept']:.2f} | "
              f"confident-error {a['controlled']:.3f} -> {b['controlled']:.3f}")
    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, "s31_clean_protocol.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[s31] wrote {op}")


if __name__ == "__main__":
    main()
