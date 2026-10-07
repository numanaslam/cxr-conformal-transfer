"""[FN-specific CRC + reverse G2] The referee's point: G2's symmetric 'accepted-and-misclassified'
loss is not the clinically asymmetric quantity. In triage the quantity to bound is the MISSED
abnormal (accepted, predicted normal, truly abnormal). CRC controls it directly: define

    L_FN(x; lambda) = 1[ accepted(lambda) AND predicted-normal AND truly-abnormal ]

which is monotone non-increasing in lambda (a larger lambda shrinks the accepted set), so CRC
applies unchanged. We fit tau (balanced accuracy) and the CRC lambda on the CALIBRATION hospital
to bound E[L_FN] <= alpha, deploy on the TEST hospital, and report acceptance, the accepted-FN
rate (the controlled quantity), and the accepted-case error. We run BOTH deployment directions,
which also supplies the reverse-direction (VinDr->NIH) G2 row the referee found missing, and we
print the symmetric-loss result beside it for contrast. Cached embeddings, CPU, no GPU.

    python -m scripts.s25_fn_crc --config configs/vindr.yaml
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from anywhere

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.ood.conformal import crc_threshold
from ovcbmr.eval.metrics import bootstrap_ci


def _load(emb_dir, proc, split, heads, vocab, nidx):
    """(A = 1 - p(Normal), y abnormal=not No-finding, patient) for a hospital split."""
    import numpy as np
    import torch
    p = os.path.join(emb_dir, f"{split}_img.pt")
    if not os.path.exists(p):
        return None
    d = torch.load(p, weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    Pm = predict_matrix(d["emb"].numpy(), heads, vocab)
    A, y, pat = [], [], []
    for i, iid in enumerate(d["ids"]):
        it = items.get(iid)
        if it is None:
            continue
        A.append(1.0 - float(Pm[i][nidx]))
        y.append(0 if int(it["labels_known"][nidx]) == 1 else 1)
        pat.append(it.get("patient_id", iid))
    return np.array(A), np.array(y, dtype=int), pat


def _fit(Ac, yc, alpha, loss_kind):
    """tau (balanced accuracy) and CRC lambda on calibration, for the chosen loss."""
    import numpy as np
    P, N = int(yc.sum()), int(len(yc) - yc.sum())
    ts = np.linspace(float(Ac.min()), float(Ac.max()), 201)
    ba = [0.5 * (((Ac >= t) & (yc == 1)).sum() / max(P, 1) + ((Ac < t) & (yc == 0)).sum() / max(N, 1)) for t in ts]
    tau = float(ts[int(np.argmax(ba))])
    yhat = (Ac >= tau).astype(int)
    if loss_kind == "fn":
        loss = ((yhat == 0) & (yc == 1)).astype(float)     # missed abnormal
    else:
        loss = (yhat != yc).astype(float)                  # symmetric error
    conf = np.abs(Ac - tau)
    lambdas = sorted(set(np.quantile(conf, np.linspace(0, 1, 201)).tolist()))
    risks = [float(((conf >= lam) * loss).mean()) for lam in lambdas]
    lam, _ = crc_threshold(risks, lambdas, n=len(conf), alpha=alpha, B=1.0)
    return tau, (lam if lam is not None else lambdas[-1])


def _deploy(dp, tau, lam, n_boot, seed):
    import numpy as np
    A, y, pat = dp
    yhat = (A >= tau).astype(int)
    acc = np.abs(A - tau) >= lam
    err = (yhat != y).astype(float)
    fn = ((yhat == 0) & (y == 1)).astype(float)
    _, lo, hi = bootstrap_ci((acc * fn).tolist(), groups=pat, n_boot=n_boot, seed=seed)
    return {"n": int(len(y)), "accept": float(acc.mean()),
            "accepted_fn_rate": float(fn[acc].mean()) if acc.sum() else 0.0,
            "accept_and_fn": float((acc * fn).mean()), "accept_and_fn_ci": [lo, hi],
            "error_given_accept": float(err[acc].mean()) if acc.sum() else 0.0}


def main():
    cfg = parse_config_arg()
    ch = getattr(cfg, "cross_hospital", None)
    if ch is None:
        print("[s25] config needs a cross_hospital block."); return
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    nidx = vocab.index("Normal")
    alpha = float(getattr(cfg.conformal, "alpha_risk", 0.10))
    n_boot = int(getattr(getattr(cfg, "eval", object()), "bootstrap", 1000))
    seed = int(getattr(cfg, "seed", 0))

    nih_cal = _load(ch.nih_embeddings, ch.nih_processed, "calib", heads, vocab, nidx)
    nih_te = _load(ch.nih_embeddings, ch.nih_processed, "test", heads, vocab, nidx)
    vin_cal = _load(cfg.paths.embeddings, cfg.paths.processed, "calib", heads, vocab, nidx)
    vin_te = _load(cfg.paths.embeddings, cfg.paths.processed, "test", heads, vocab, nidx)
    if any(x is None for x in (nih_cal, nih_te, vin_cal, vin_te)):
        print("[s25] missing an embeddings split."); return

    directions = [("NIH->VinDr", nih_cal, vin_te), ("VinDr->NIH", vin_cal, nih_te)]
    out = {"alpha_risk": alpha, "rows": {}}
    print(f"[s25] FN-controlled vs symmetric CRC | alpha={alpha} | boot={n_boot}\n")
    print(f"{'direction':12}{'loss':10}{'tau':>7}{'lam':>7}{'accept':>8}{'acc.FN':>8}"
          f"{'FN<=a?':>8}{'err|acc':>9}")
    for dname, cal, te in directions:
        for loss_kind in ("fn", "symmetric"):
            tau, lam = _fit(cal[0], cal[1], alpha, loss_kind)
            r = _deploy(te, tau, lam, n_boot, seed)
            ok = "YES" if r["accept_and_fn"] <= alpha + 1e-9 else "NO"
            print(f"{dname:12}{loss_kind:10}{tau:7.3f}{lam:7.3f}{r['accept']:8.2f}"
                  f"{r['accepted_fn_rate']:8.3f}{ok:>8}{r['error_given_accept']:9.3f}")
            out["rows"][f"{dname}:{loss_kind}"] = {**r, "tau": tau, "lam": lam}
        print()

    print("[s25] Read: for loss='fn' the accept&FN column is the CRC-controlled quantity and "
          "should sit <= alpha; compare its acceptance against loss='symmetric' to see the "
          "cost of bounding misses directly.")
    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, "s25_fn_crc.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[s25] wrote {op}")


if __name__ == "__main__":
    main()
