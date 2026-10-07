"""[black-box baseline] Does routing through named concepts cost anything statistically vs a
direct black-box classifier? Train a DIRECT abnormal-vs-normal linear head on the SAME frozen
BiomedCLIP embeddings (no concepts) and run the identical selective-prediction (CRC) machinery,
then compare cross-hospital (NIH->VinDr) discrimination (AUROC), coverage, and accept-&-error
against the CBM-derived abnormality A = 1 - p(Normal). If the two match, the concept bottleneck
buys named-concept interpretability (and the unanchored valve, which has no black-box analogue)
at no statistical cost -- the claim the paper asserts. Cached embeddings, no GPU.

    python -m scripts.s20_blackbox_baseline --config configs/vindr.yaml
"""
from __future__ import annotations
import json
import os

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.ood.conformal import crc_threshold
from ovcbmr.eval.metrics import auroc, bootstrap_ci


def _load(emb_dir, proc, split):
    """(emb [N,D], y abnormal=not No-finding, patient) from cached embeddings + index."""
    import numpy as np
    import torch
    d = torch.load(os.path.join(emb_dir, f"{split}_img.pt"), weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    with open(os.path.join(proc, "meta.json")) as f:
        nidx = json.load(f)["vocabulary"].index("Normal")
    E, y, pat = [], [], []
    for i, iid in enumerate(d["ids"]):
        it = items.get(iid)
        if it is None:
            continue
        E.append(d["emb"][i].numpy())
        y.append(0 if int(it["labels_known"][nidx]) == 1 else 1)
        pat.append(it.get("patient_id", iid))
    return np.array(E), np.array(y, dtype=int), pat


def _fit(sc, yc, alpha):
    import numpy as np
    P, N = int(yc.sum()), int(len(yc) - yc.sum())
    ts = np.linspace(float(sc.min()), float(sc.max()), 201)
    ba = [0.5 * (((sc >= t) & (yc == 1)).sum() / max(P, 1) + ((sc < t) & (yc == 0)).sum() / max(N, 1)) for t in ts]
    tau = float(ts[int(np.argmax(ba))])
    loss = ((sc >= tau).astype(int) != yc).astype(float)
    conf = np.abs(sc - tau)
    lambdas = sorted(set(np.quantile(conf, np.linspace(0, 1, 201)).tolist()))
    risks = [float(((conf >= lam) * loss).mean()) for lam in lambdas]
    lam, _ = crc_threshold(risks, lambdas, n=len(conf), alpha=alpha, B=1.0)
    return tau, (lam if lam is not None else lambdas[-1])


def _deploy(sc, y, pat, tau, lam, n_boot, seed):
    import numpy as np
    acc = (np.abs(sc - tau) >= lam)
    loss = ((sc >= tau).astype(int) != y).astype(float)
    _, lo, hi = bootstrap_ci((acc * loss).tolist(), groups=pat, n_boot=n_boot, seed=seed)
    return float(acc.mean()), float((acc * loss).mean()), [lo, hi]


def main():
    cfg = parse_config_arg()
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    ch = cfg.cross_hospital
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    nidx = vocab.index("Normal")
    alpha = float(getattr(cfg.conformal, "alpha_risk", 0.10))
    n_boot = int(getattr(getattr(cfg, "eval", object()), "bootstrap", 1000))
    seed = int(getattr(cfg, "seed", 0))

    Etr, ytr, _ = _load(ch.nih_embeddings, ch.nih_processed, "train")
    Ec, yc, pc = _load(ch.nih_embeddings, ch.nih_processed, "calib")
    Et, yt, pt = _load(ch.nih_embeddings, ch.nih_processed, "test")
    Ev, yv, pv = _load(cfg.paths.embeddings, cfg.paths.processed, "test")

    # black-box direct abnormal-vs-normal head on the SAME frozen embeddings
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Etr, ytr)
    def bb(E):
        return clf.predict_proba(E)[:, 1]
    # CBM abnormality A = 1 - p(Normal) from the shared concept heads
    def cbm(E):
        return 1.0 - predict_matrix(E, heads, vocab)[:, nidx]

    print(f"[s20] black-box direct head vs CBM abnormality | alpha_risk={alpha} | boot={n_boot}\n")
    print(f"{'predictor':<20}{'AUROC NIH':>10}{'AUROC VinDr':>12}{'NIH->VinDr cover':>17}"
          f"{'accept&err [CI]':>22}{'interpretable':>15}")
    out = {}
    for name, f, tag in (("black-box direct", bb, "heatmap only"), ("CBM  1-p(Normal)", cbm, "named concepts")):
        sc_c, sc_t, sc_v = f(Ec), f(Et), f(Ev)
        au_n, au_v = auroc(sc_t.tolist(), yt.tolist()), auroc(sc_v.tolist(), yv.tolist())
        tau, lam = _fit(sc_c, yc, alpha)
        cov, ae, ci = _deploy(sc_v, yv, pv, tau, lam, n_boot, seed)
        print(f"{name:<20}{au_n:>10.3f}{au_v:>12.3f}{cov:>17.2f}"
              f"{ae:>11.3f} [{ci[0]:.3f},{ci[1]:.3f}]{tag:>15}")
        out[name] = {"auroc_nih": au_n, "auroc_vindr": au_v, "nih_to_vindr_coverage": cov,
                     "accept_error": ae, "accept_error_ci": ci, "interpretability": tag}

    print("\n[s20] The CBM additionally provides: (i) which finding is uncertain (concept attribution),")
    print("      and (ii) the unanchored/open-set safety valve -- neither has a black-box equivalent.")
    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, "s20_blackbox_baseline.json")
    with open(op, "w") as f:
        json.dump({"alpha_risk": alpha, "methods": out}, f, indent=2)
    print(f"[s20] wrote {op}")


if __name__ == "__main__":
    main()
