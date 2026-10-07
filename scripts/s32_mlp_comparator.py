"""[a comparator that could win] The paper's black-box baseline is a direct LINEAR abnormal-vs-
normal head on the same frozen features, which is nearly the same object as the bottleneck's
A(x) = 1 - p(Normal); parity there is expected and bounds nothing. This adds a NONLINEAR head
(one hidden layer) on the same frozen embeddings, a black box that is free to use feature
interactions the linear concept layer cannot, and reports

  * abnormality AUROC on the NIH and VinDr test sets for: linear head, MLP head, CBM A(x);
  * a paired, clustered bootstrap interval for the AUROC difference (comparator - CBM), so the
    paper can state a margin instead of 'statistically indistinguishable';
  * the NIH->VinDr selective operating point (acceptance, confident-error) for each.

Cached embeddings, CPU, a few minutes for the MLP fit.

    python -m scripts.s32_mlp_comparator --config configs/vindr.yaml
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from anywhere

from ovcbmr.config import load_config
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.ood.conformal import crc_threshold
from ovcbmr.eval.metrics import auroc, bootstrap_ci


def _load(emb_dir, proc, split):
    """(E [N,D], y abnormal = not No-finding, patient ids) from cached embeddings + index."""
    import numpy as np
    import torch
    d = torch.load(os.path.join(emb_dir, f"{split}_img.pt"), weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    with open(os.path.join(proc, "meta.json")) as f:
        nidx = json.load(f)["vocabulary"].index("Normal")
    keep = [i for i, iid in enumerate(d["ids"]) if iid in items]
    ids = [d["ids"][i] for i in keep]
    y = np.array([0 if int(items[i]["labels_known"][nidx]) == 1 else 1 for i in ids], dtype=int)
    return d["emb"].numpy()[keep], y, [str(items[i].get("patient_id", i)) for i in ids]


def _fit(sc, yc, alpha):
    """tau by balanced accuracy and the CRC stringency, both on the calibration scores."""
    import numpy as np
    P, N = int(yc.sum()), int(len(yc) - yc.sum())
    ts = np.linspace(float(sc.min()), float(sc.max()), 201)
    ba = [0.5 * (((sc >= t) & (yc == 1)).sum() / max(P, 1) + ((sc < t) & (yc == 0)).sum() / max(N, 1))
          for t in ts]
    tau = float(ts[int(np.argmax(ba))])
    loss = ((sc >= tau).astype(int) != yc).astype(float)
    conf = np.abs(sc - tau)
    lambdas = sorted(set(np.quantile(conf, np.linspace(0, 1, 201)).tolist()))
    risks = [float(((conf >= lam) * loss).mean()) for lam in lambdas]
    lam, _ = crc_threshold(risks, lambdas, n=len(conf), alpha=alpha, B=1.0)
    return tau, (lam if lam is not None else lambdas[-1])


def _deploy(sc, y, tau, lam):
    import numpy as np
    acc = np.abs(sc - tau) >= lam
    err = ((sc >= tau).astype(int) != y).astype(float)
    return float(acc.mean()), float((acc * err).mean())


def _diff_ci(sa, sb, y, pat, nb, seed):
    """Clustered paired bootstrap for AUROC(sa) - AUROC(sb) on the same cases."""
    vals = list(zip(sa.tolist(), sb.tolist(), y.tolist()))

    def stat(v):
        yy = [t[2] for t in v]
        if sum(yy) == 0 or sum(yy) == len(yy):
            return 0.0
        return auroc([t[0] for t in v], yy) - auroc([t[1] for t in v], yy)
    return bootstrap_ci(vals, statistic=stat, n_boot=nb, seed=seed, groups=pat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vindr.yaml")
    ap.add_argument("--nboot", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=256)
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config)

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    ch = cfg.cross_hospital
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    nidx = vocab.index("Normal")
    alpha = float(getattr(cfg.conformal, "alpha_risk", 0.10))
    seed = int(getattr(cfg, "seed", 0))

    Etr, ytr, _ = _load(ch.nih_embeddings, ch.nih_processed, "train")
    Ec, yc, _ = _load(ch.nih_embeddings, ch.nih_processed, "calib")
    Et, yt, pt = _load(ch.nih_embeddings, ch.nih_processed, "test")
    Ev, yv, pv = _load(cfg.paths.embeddings, cfg.paths.processed, "test")

    print(f"[s32] fitting linear and MLP({args.hidden}) abnormality heads on {len(ytr)} NIH train embeddings...")
    lin = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Etr, ytr)
    mlp = MLPClassifier(hidden_layer_sizes=(args.hidden,), early_stopping=True, validation_fraction=0.1,
                        max_iter=200, random_state=seed).fit(Etr, ytr)
    preds = {
        "linear head": lambda E: lin.predict_proba(E)[:, 1],
        f"MLP head ({args.hidden})": lambda E: mlp.predict_proba(E)[:, 1],
        "CBM  1-p(Normal)": lambda E: 1.0 - predict_matrix(E, heads, vocab)[:, nidx],
    }
    sc = {k: {"c": f(Ec), "t": f(Et), "v": f(Ev)} for k, f in preds.items()}
    ref = "CBM  1-p(Normal)"

    print(f"\n{'predictor':<20}{'AUROC NIH':>10}{'AUROC VinDr':>12}{'accept':>8}{'conf-err':>10}")
    out = {"alpha_risk": alpha, "n_boot": args.nboot, "methods": {}, "differences_vs_cbm": {}}
    for k, s in sc.items():
        au_n, au_v = auroc(s["t"].tolist(), yt.tolist()), auroc(s["v"].tolist(), yv.tolist())
        tau, lam = _fit(s["c"], yc, alpha)
        acc, ce = _deploy(s["v"], yv, tau, lam)
        print(f"{k:<20}{au_n:>10.3f}{au_v:>12.3f}{acc:>8.2f}{ce:>10.3f}")
        out["methods"][k] = {"auroc_nih": au_n, "auroc_vindr": au_v,
                             "nih_to_vindr_accept": acc, "nih_to_vindr_confident_error": ce}

    print(f"\nAUROC difference vs the bottleneck (comparator - CBM), 95% clustered paired bootstrap, "
          f"{args.nboot} resamples")
    for k in sc:
        if k == ref:
            continue
        dn, lon, hin = _diff_ci(sc[k]["t"], sc[ref]["t"], yt, pt, args.nboot, seed)
        dvv, lov, hiv = _diff_ci(sc[k]["v"], sc[ref]["v"], yv, pv, args.nboot, seed)
        print(f"  {k:<20} NIH {dn:+.3f} [{lon:+.3f},{hin:+.3f}]   VinDr {dvv:+.3f} [{lov:+.3f},{hiv:+.3f}]")
        out["differences_vs_cbm"][k] = {"nih": [dn, lon, hin], "vindr": [dvv, lov, hiv]}

    print("\n[s32] Read: an interval inside a stated margin (say +/-0.01 AUROC) supports 'no loss "
          "beyond the margin'; an MLP interval clearly above zero is the cost of the linear "
          "concept layer and should be reported as such.")
    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, "s32_mlp_comparator.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[s32] wrote {op}")


if __name__ == "__main__":
    main()
