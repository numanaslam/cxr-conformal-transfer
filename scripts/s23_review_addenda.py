"""[review addenda] CPU-only extras requested in review, using the paper's exact machinery
(same heads, valve statistic, split-conformal q-hat, and CRC rule as s16/s17). Prints four
labelled blocks to paste back:

  (A) G1 false-deferral, IN-VOCABULARY known-abnormal, BOTH directions (forward is a sanity
      check that should reproduce ~0.085 / 0.053; reverse VinDr->NIH answers review #10).
  (D) Weighted (importance-weighted) conformal vs naive vs group-conditional, NIH->VinDr
      in-vocab G1, with domain-classifier AUROC and Kish ESS (review #11).
  (B) G2 accepted sensitivity/specificity + acceptance among normal vs abnormal (review #5).
  (C) G2 sample-size: how much local VinDr calibration recovers the acceptance rate (review #7).

Cached embeddings only, no GPU.

    python -m scripts.s23_review_addenda --config configs/vindr.yaml
"""
from __future__ import annotations
import argparse
import json
import os

from ovcbmr.config import load_config
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.concept.valve import image_unexplained
from ovcbmr.ood.conformal import (split_conformal_threshold, empirical_flag_rate, crc_threshold,
                                   weighted_split_conformal_threshold, effective_sample_size,
                                   estimate_shift_weights)
from ovcbmr.eval.metrics import bootstrap_ci, auroc


def _items(proc):
    with open(os.path.join(proc, "index.json")) as f:
        return {it["id"]: it for it in json.load(f)}


def _emb(emb_dir, split):
    import torch
    d = torch.load(os.path.join(emb_dir, f"{split}_img.pt"), weights_only=False)
    return d["emb"].numpy(), d["ids"]


def _g1(emb_dir, proc, split, heads, vocab, ab_cols, want_emb=False):
    """Valve scores (and embeddings) for IN-VOCABULARY known-abnormal cases (>=1 of the 8 heads)."""
    import numpy as np
    E, ids = _emb(emb_dir, split)
    items = _items(proc)
    Pm = predict_matrix(E, heads, vocab)
    S, pat, keep = [], [], []
    for i, iid in enumerate(ids):
        it = items.get(iid)
        if it and any(int(it["labels_known"][j]) == 1 for j in ab_cols):
            S.append(image_unexplained(Pm[i].tolist(), vocab, "Normal")[1])
            pat.append(it.get("patient_id", iid))
            keep.append(i)
    S = np.array(S)
    return (S, pat, E[keep]) if want_emb else (S, pat)


def _flag(scores, q, pat, nb, seed):
    r = empirical_flag_rate(scores, q)
    _, lo, hi = bootstrap_ci([1.0 if s > q else 0.0 for s in scores], groups=pat, n_boot=nb, seed=seed)
    return r, lo, hi


# ---- G2 (identical rule to s16) ----
def _g2_load(emb_dir, proc, split, heads, vocab, nidx):
    import numpy as np
    E, ids = _emb(emb_dir, split)
    items = _items(proc)
    Pm = predict_matrix(E, heads, vocab)
    A, y, pat = [], [], []
    for i, iid in enumerate(ids):
        it = items.get(iid)
        if it is None:
            continue
        A.append(1.0 - float(Pm[i][nidx]))
        y.append(0 if int(it["labels_known"][nidx]) == 1 else 1)
        pat.append(it.get("patient_id", iid))
    return np.array(A), np.array(y, dtype=int), pat


def _g2_fit(Ac, yc, alpha):
    import numpy as np
    P, N = int(yc.sum()), int(len(yc) - yc.sum())
    ts = np.linspace(float(Ac.min()), float(Ac.max()), 201)
    ba = [0.5 * (((Ac >= t) & (yc == 1)).sum() / max(P, 1) + ((Ac < t) & (yc == 0)).sum() / max(N, 1))
          for t in ts]
    tau = float(ts[int(np.argmax(ba))])
    conf = np.abs(Ac - tau)
    loss = ((Ac >= tau).astype(int) != yc).astype(float)
    lambdas = sorted(set(np.quantile(conf, np.linspace(0, 1, 201)).tolist()))
    risks = [float(((conf >= lam) * loss).mean()) for lam in lambdas]
    lam, _ = crc_threshold(risks, lambdas, n=len(conf), alpha=alpha, B=1.0)
    return tau, (lam if lam is not None else lambdas[-1])


def _g2_full(dp, tau, lam):
    import numpy as np
    A, y, _ = dp
    yhat = (A >= tau).astype(int)
    acc = (np.abs(A - tau) >= lam)
    ab, no = (y == 1), (y == 0)

    def rate(num, den):
        return float(num.sum() / max(den.sum(), 1))

    return dict(coverage=float(acc.mean()),
                acc_among_normal=float(acc[no].mean()) if no.sum() else 0.0,
                acc_among_abnormal=float(acc[ab].mean()) if ab.sum() else 0.0,
                accepted_sensitivity=rate((yhat == 1) & acc & ab, acc & ab),
                accepted_specificity=rate((yhat == 0) & acc & no, acc & no),
                accepted_fn_rate=float(((yhat == 0) & (y == 1))[acc].mean()) if acc.sum() else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vindr.yaml")
    ap.add_argument("--alpha", type=float, default=0.05)        # G1 target
    ap.add_argument("--alpha-risk", type=float, default=0.10)   # G2 target
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--ksub", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config)
    import numpy as np

    ch = cfg.cross_hospital
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    nidx = vocab.index("Normal")
    ab_cols = [j for j, c in enumerate(vocab) if c != "Normal"]
    nb, seed = args.nboot, int(getattr(cfg, "seed", 0))
    NE, NP = ch.nih_embeddings, ch.nih_processed
    VE, VP = cfg.paths.embeddings, cfg.paths.processed

    # ===================== (A) G1 in-vocab, both directions =====================
    print("=" * 74)
    print(f"(A) G1 false-deferral, IN-VOCAB known-abnormal (alpha={args.alpha})   [review #10]")
    nih_cal_S, nih_cal_pat, nih_cal_E = _g1(NE, NP, "calib", heads, vocab, ab_cols, True)
    nih_te_S, nih_te_pat = _g1(NE, NP, "test", heads, vocab, ab_cols)
    vin_cal_S, vin_cal_pat = _g1(VE, VP, "calib", heads, vocab, ab_cols)
    vin_te_S, vin_te_pat, vin_te_E = _g1(VE, VP, "test", heads, vocab, ab_cols, True)
    qf_naive = split_conformal_threshold(nih_cal_S, args.alpha)   # cal NIH  -> deploy VinDr
    qf_gc = split_conformal_threshold(vin_cal_S, args.alpha)      # cal VinDr-> deploy VinDr
    qr_naive = split_conformal_threshold(vin_cal_S, args.alpha)   # cal VinDr-> deploy NIH
    qr_gc = split_conformal_threshold(nih_cal_S, args.alpha)      # cal NIH  -> deploy NIH
    print("  NIH ->VinDr  naive      : %.3f [%.3f,%.3f]" % _flag(vin_te_S, qf_naive, vin_te_pat, nb, seed))
    print("  NIH ->VinDr  group-cond : %.3f [%.3f,%.3f]" % _flag(vin_te_S, qf_gc, vin_te_pat, nb, seed))
    print("  VinDr->NIH   naive      : %.3f [%.3f,%.3f]" % _flag(nih_te_S, qr_naive, nih_te_pat, nb, seed))
    print("  VinDr->NIH   group-cond : %.3f [%.3f,%.3f]" % _flag(nih_te_S, qr_gc, nih_te_pat, nb, seed))
    print("  n in-vocab: NIH-cal=%d NIH-test=%d VinDr-cal=%d VinDr-test=%d" %
          (len(nih_cal_S), len(nih_te_S), len(vin_cal_S), len(vin_te_S)))

    # ===================== (D) weighted conformal NIH->VinDr =====================
    print("=" * 74)
    print("(D) Weighted (importance-weighted) conformal, NIH->VinDr in-vocab G1   [review #11]")
    w = estimate_shift_weights(nih_cal_E, vin_te_E)
    w = np.asarray(w[0] if isinstance(w, tuple) else w).ravel()
    qw = weighted_split_conformal_threshold(nih_cal_S, w, args.alpha)
    ess = float(effective_sample_size(w))
    from sklearn.linear_model import LogisticRegression
    X = np.vstack([nih_cal_E, vin_te_E])
    yd = np.r_[np.zeros(len(nih_cal_E)), np.ones(len(vin_te_E))]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    cut = len(X) // 2
    clf = LogisticRegression(C=0.1, max_iter=1000).fit(X[perm[:cut]], yd[perm[:cut]])
    dom_auc = auroc(clf.predict_proba(X[perm[cut:]])[:, 1].tolist(), yd[perm[cut:]].astype(int).tolist())
    a = args.alpha
    print("  domain-classifier AUROC : %.3f   (0.5 = no shift, 1.0 = fully separable)" % dom_auc)
    print("  weights: clipped 95th pct; Kish ESS = %.0f of %d NIH-cal (%.0f%%)" % (ess, len(w), 100 * ess / len(w)))
    print("  VinDr false-deferral    :   naive %.3f (|dev| %.3f) | weighted %.3f (|dev| %.3f) | group-cond %.3f (|dev| %.3f)" %
          (empirical_flag_rate(vin_te_S, qf_naive), abs(empirical_flag_rate(vin_te_S, qf_naive) - a),
           empirical_flag_rate(vin_te_S, qw), abs(empirical_flag_rate(vin_te_S, qw) - a),
           empirical_flag_rate(vin_te_S, qf_gc), abs(empirical_flag_rate(vin_te_S, qf_gc) - a)))

    # ===================== (B) G2 accepted sensitivity/specificity =====================
    print("=" * 74)
    print(f"(B) G2 accepted sensitivity/specificity (alpha_risk={args.alpha_risk})   [review #5]")
    n_cal = _g2_load(NE, NP, "calib", heads, vocab, nidx)
    n_te = _g2_load(NE, NP, "test", heads, vocab, nidx)
    v_cal = _g2_load(VE, VP, "calib", heads, vocab, nidx)
    v_te = _g2_load(VE, VP, "test", heads, vocab, nidx)
    tau_n, lam_n = _g2_fit(n_cal[0], n_cal[1], args.alpha_risk)
    tau_v, lam_v = _g2_fit(v_cal[0], v_cal[1], args.alpha_risk)
    print("  %-24s %7s %8s %8s %9s %9s %8s" %
          ("setting", "accept", "acc|norm", "acc|abn", "sens|acc", "spec|acc", "FN|acc"))
    for name, dp, tau, lam in [("NIH  (cal NIH)", n_te, tau_n, lam_n),
                               ("VinDr naive (cal NIH)", v_te, tau_n, lam_n),
                               ("VinDr site  (cal VinDr)", v_te, tau_v, lam_v)]:
        r = _g2_full(dp, tau, lam)
        print("  %-24s %7.2f %8.2f %8.2f %9.3f %9.3f %8.3f" %
              (name, r["coverage"], r["acc_among_normal"], r["acc_among_abnormal"],
               r["accepted_sensitivity"], r["accepted_specificity"], r["accepted_fn_rate"]))

    # ===================== (C) G2 sample-size =====================
    print("=" * 74)
    print(f"(C) G2 sample-size: local VinDr calib m -> acceptance + accepted-error, K={args.ksub}   [review #7]")
    Ac, yc, _ = v_cal
    A, y, _ = v_te
    rng = np.random.default_rng(seed)
    print("  %8s %26s %26s" % ("m", "acceptance [5th,95th]", "accepted-error [5th,95th]"))
    for m in [100, 200, 400, 800, 1500]:
        if m > len(Ac):
            continue
        covs, errs = [], []
        for _ in range(args.ksub):
            idx = rng.choice(len(Ac), m, replace=False)
            if yc[idx].sum() < 2 or (len(idx) - yc[idx].sum()) < 2:
                continue
            tau, lam = _g2_fit(Ac[idx], yc[idx], args.alpha_risk)
            acc = (np.abs(A - tau) >= lam)
            covs.append(float(acc.mean()))
            errs.append(float((acc * ((A >= tau).astype(int) != y)).mean()))
        c, e = np.array(covs), np.array(errs)
        print("  %8d       %.2f [%.2f,%.2f]              %.3f [%.3f,%.3f]" %
              (m, c.mean(), np.percentile(c, 5), np.percentile(c, 95),
               e.mean(), np.percentile(e, 5), np.percentile(e, 95)))
    acc = (np.abs(A - tau_v) >= lam_v)
    print("  %8s       %.2f  (site-calibrated, full)      %.3f" %
          ("full", float(acc.mean()), float((acc * ((A >= tau_v).astype(int) != y)).mean())))
    print("=" * 74)
    print("[s23] done -- paste the four blocks (A)(D)(B)(C) above.")


if __name__ == "__main__":
    main()
