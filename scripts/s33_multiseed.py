"""[multi-seed robustness] The paper's numbers come from one split (seed 42). This repeats the
whole downstream pipeline over several split seeds, with no GPU: the cached embeddings do not
depend on the split, so each seed re-partitions them and refits everything after the encoder.

Per seed:
  NIH   pooled train/calib/val/test embeddings are re-split BY PATIENT (0.70/0.10/0.10/0.10)
        with the project's own hash rule; the nine linear heads are retrained (same recipe as
        s09: balanced logistic regression); Platt scalars and the G2 decision threshold come from
        the VAL split and the conformal thresholds from the CALIB split (the clean protocol).
  VinDr pooled calib/test studies are re-split in half; half of the new calib sets tau, the
        other half the G2 stringency; the whole new calib sets the G1 quantile.
Reported per seed and as mean / sd / min / max: mean abnormal-concept AUROC and ECE at both
sites, abnormality AUROC, G1 false deferral and G2 (symmetric) acceptance and confident-error in
all four settings (on-site at each hospital and both transfer directions).

Nine logistic fits on ~78k x 512 per seed: expect a few minutes per seed on CPU.

    python -m scripts.s33_multiseed --config configs/vindr.yaml --seeds 0 1 2 3 4
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from anywhere

from ovcbmr.config import load_config
from ovcbmr.concept.heads import predict_matrix
from ovcbmr.ood.conformal import split_conformal_threshold, empirical_flag_rate, crc_threshold
from ovcbmr.eval.metrics import auroc, expected_calibration_error

FRACTIONS = (("train", 0.70), ("calib", 0.10), ("val", 0.10), ("test", 0.10))


def _unit_hash(seed, key):
    """Same rule as ovcbmr.io.splits: deterministic value in [0, 1) for a patient key."""
    return int(hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _pool(emb_dir, proc, splits):
    """Concatenate cached splits -> (E [N,D], Y [N,K], patient ids)."""
    import numpy as np
    import torch
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    Es, Ys, pats = [], [], []
    for sp in splits:
        p = os.path.join(emb_dir, f"{sp}_img.pt")
        if not os.path.exists(p):
            continue
        d = torch.load(p, weights_only=False)
        keep = [i for i, iid in enumerate(d["ids"]) if iid in items]
        ids = [d["ids"][i] for i in keep]
        Es.append(d["emb"].numpy()[keep])
        Ys.append(np.array([items[i]["labels_known"] for i in ids], dtype=int))
        pats.extend(str(items[i].get("patient_id", i)) for i in ids)
    return np.concatenate(Es), np.concatenate(Ys), np.array(pats, dtype=object)


def _assign(pats, seed):
    """Patient-level split labels for one seed."""
    import numpy as np
    bounds, acc = [], 0.0
    for name, frac in FRACTIONS:
        acc += frac
        bounds.append((name, acc))
    cache, out = {}, np.empty(len(pats), dtype=object)
    for i, p in enumerate(pats):
        if p not in cache:
            u = _unit_hash(seed, p)
            cache[p] = next((n for n, b in bounds if u < b), FRACTIONS[-1][0])
        out[i] = cache[p]
    return out


def _train_heads(Xtr, Ytr, Xpl, Ypl, vocab):
    """s09 recipe: balanced logistic head per concept, Platt scalars fit on (Xpl, Ypl)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    heads = {}
    for j, c in enumerate(vocab):
        ytr = Ytr[:, j]
        if ytr.sum() < 10 or ytr.sum() == len(ytr):
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced").fit(Xtr, ytr)
        raw, ypl = clf.decision_function(Xpl), Ypl[:, j]
        if len(np.unique(ypl)) < 2:
            pa, pb = 1.0, 0.0
        else:
            pl = LogisticRegression(max_iter=1000).fit(raw.reshape(-1, 1), ypl)
            pa, pb = float(pl.coef_[0][0]), float(pl.intercept_[0])
        heads[c] = {"coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "platt": [pa, pb]}
    return heads


def _derive(E, Y, heads, vocab, nidx, abn_idx):
    P = predict_matrix(E, heads, vocab)
    A = 1.0 - P[:, nidx]
    return {"P": P, "A": A, "s": A * (1.0 - P[:, abn_idx].max(axis=1)),
            "known": Y[:, abn_idx].sum(axis=1) > 0, "y": (Y[:, nidx] != 1).astype(int), "Y": Y}


def _tau(D):
    import numpy as np
    A, y = D["A"], D["y"]
    P, N = int(y.sum()), int(len(y) - y.sum())
    ts = np.linspace(float(A.min()), float(A.max()), 201)
    ba = [0.5 * (((A >= t) & (y == 1)).sum() / max(P, 1) + ((A < t) & (y == 0)).sum() / max(N, 1))
          for t in ts]
    return float(ts[int(np.argmax(ba))])


def _lam(D, tau, alpha):
    import numpy as np
    loss = ((D["A"] >= tau).astype(int) != D["y"]).astype(float)
    conf = np.abs(D["A"] - tau)
    lambdas = sorted(set(np.quantile(conf, np.linspace(0, 1, 201)).tolist()))
    risks = [float(((conf >= lam) * loss).mean()) for lam in lambdas]
    lam, _ = crc_threshold(risks, lambdas, n=len(conf), alpha=alpha, B=1.0)
    return float(lam if lam is not None else lambdas[-1])


def _g2(D, tau, lam):
    import numpy as np
    acc = np.abs(D["A"] - tau) >= lam
    err = ((D["A"] >= tau).astype(int) != D["y"]).astype(float)
    return float(acc.mean()), float((acc * err).mean())


def _ffr(D, q):
    return empirical_flag_rate(D["s"][D["known"]].tolist(), q)


def _concept_stats(D, abn_idx):
    import numpy as np
    aus, eces = [], []
    for j in abn_idx:
        y = D["Y"][:, j]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        aus.append(auroc(D["P"][:, j].tolist(), y.tolist()))
        eces.append(expected_calibration_error(D["P"][:, j].tolist(), y.tolist()))
    return float(np.mean(aus)), float(np.mean(eces))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vindr.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="s33_multiseed.json", help="file name under the results folder")
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config)
    import numpy as np
    ch = cfg.cross_hospital
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    nidx = vocab.index("Normal")
    abn_idx = [i for i, c in enumerate(vocab) if c != "Normal"]
    a1 = float(getattr(cfg.conformal, "alpha_flag", 0.05))
    a2 = float(getattr(cfg.conformal, "alpha_risk", 0.10))

    En, Yn, Pn = _pool(ch.nih_embeddings, ch.nih_processed, ("train", "calib", "val", "test"))
    Ev, Yv, Pv = _pool(cfg.paths.embeddings, cfg.paths.processed, ("calib", "test"))
    print(f"[s33] pooled NIH {En.shape[0]} images / {len(set(Pn.tolist()))} patients, "
          f"VinDr {Ev.shape[0]} studies; seeds {args.seeds}\n")

    rows = []
    for seed in args.seeds:
        sp = _assign(Pn, seed)
        m = {k: sp == k for k in ("train", "calib", "val", "test")}
        heads = _train_heads(En[m["train"]], Yn[m["train"]], En[m["val"]], Yn[m["val"]], vocab)
        vocab_s = [c for c in vocab if c in heads]
        if "Normal" not in vocab_s:
            print(f"[s33] seed {seed}: no Normal head, skipped"); continue
        dv = lambda E, Y: _derive(E, Y, heads, vocab, nidx, abn_idx)   # noqa: E731
        N_val, N_cal, N_te = (dv(En[m[k]], Yn[m[k]]) for k in ("val", "calib", "test"))

        vcal = np.array([_unit_hash(seed, p) < 0.5 for p in Pv])
        vtun = np.array([_unit_hash(seed + 1000, p) < 0.5 for p in Pv])
        V_cal, V_te = dv(Ev[vcal], Yv[vcal]), dv(Ev[~vcal], Yv[~vcal])
        V_tune, V_fit = dv(Ev[vcal & vtun], Yv[vcal & vtun]), dv(Ev[vcal & ~vtun], Yv[vcal & ~vtun])

        qN = split_conformal_threshold(N_cal["s"][N_cal["known"]].tolist(), a1)
        qV = split_conformal_threshold(V_cal["s"][V_cal["known"]].tolist(), a1)
        tN, tV = _tau(N_val), _tau(V_tune)
        lN, lV = _lam(N_cal, tN, a2), _lam(V_fit, tV, a2)
        auN, ecN = _concept_stats(N_te, abn_idx)
        auV, ecV = _concept_stats(V_te, abn_idx)
        r = {"seed": seed, "auroc_nih": auN, "auroc_vindr": auV, "ece_nih": ecN, "ece_vindr": ecV,
             "abn_auroc_nih": auroc(N_te["A"].tolist(), N_te["y"].tolist()),
             "abn_auroc_vindr": auroc(V_te["A"].tolist(), V_te["y"].tolist()),
             "g1_nih_onsite": _ffr(N_te, qN), "g1_nih_to_vindr": _ffr(V_te, qN),
             "g1_vindr_onsite": _ffr(V_te, qV), "g1_vindr_to_nih": _ffr(N_te, qV)}
        for name, D, t, l in (("nih_onsite", N_te, tN, lN), ("nih_to_vindr", V_te, tN, lN),
                              ("vindr_onsite", V_te, tV, lV), ("vindr_to_nih", N_te, tV, lV)):
            r[f"g2_accept_{name}"], r[f"g2_conferr_{name}"] = _g2(D, t, l)
        rows.append(r)
        print(f"[s33] seed {seed}: AUROC {auN:.3f}->{auV:.3f} | ECE {ecN:.3f}->{ecV:.3f} | "
              f"G1 fwd {r['g1_nih_to_vindr']:.3f} site {r['g1_vindr_onsite']:.3f} rev {r['g1_vindr_to_nih']:.3f} | "
              f"G2 conf-err fwd {r['g2_conferr_nih_to_vindr']:.3f} rev {r['g2_conferr_vindr_to_nih']:.3f}")

    if not rows:
        print("[s33] no seed completed."); return
    keys = [k for k in rows[0] if k != "seed"]
    print(f"\n{'metric':<28}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}")
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        summary[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "min": float(v.min()), "max": float(v.max())}
        s = summary[k]
        print(f"{k:<28}{s['mean']:>8.3f}{s['sd']:>8.3f}{s['min']:>8.3f}{s['max']:>8.3f}")
    print(f"\n[s33] targets: G1 {a1}, G2 confident-error {a2}. Stable across seeds if the sd is small "
          "against the gap between naive and on-site values.")
    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, args.out)
    with open(op, "w") as f:
        json.dump({"seeds": args.seeds, "per_seed": rows, "summary": summary}, f, indent=2)
    print(f"[s33] wrote {op}")


if __name__ == "__main__":
    main()
