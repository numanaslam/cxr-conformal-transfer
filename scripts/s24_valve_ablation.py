"""[valve ablation] Does the product form s(x)=A(x)*(1-max_k a_k) earn itself, or would a
simpler score do the same G1 job? Referee asks for a baseline for the G1 score. We compare
four scores on the SAME NIH-calibrated -> VinDr-deployed setup:

    valve   s(x) = A(x) * (1 - max_{k != Normal} a_k)      [the paper's statistic]
    A       A(x) = 1 - p(Normal)                            [abnormality only]
    gap     g(x) = 1 - max_{k != Normal} a_k                [unanchored gap only]
    entropy H(x) = normalized entropy of the abnormal-concept probability vector

For each score we (1) fit the one-sided split-conformal q_hat on NIH in-vocab known-abnormal
calibration cases at alpha, (2) report the VinDr in-vocab false-deferral rate (should be ~alpha,
lower is not better -- it means fewer known-abnormal cases flagged) and the VinDr OOV-abnormal
flag rate (the open-set job: HIGHER is better), and (3) report the direct separation AUROC of
OOV-abnormal (label 1) vs in-vocab known-abnormal (label 0), which is the discriminative task
the valve was designed for. If the valve's separation AUROC does not beat A and gap, the
product form is not doing work and the paper should say so. Cached embeddings, CPU, no GPU.

    python -m scripts.s24_valve_ablation --config configs/vindr.yaml
"""
from __future__ import annotations
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from anywhere

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.ood.conformal import split_conformal_threshold, empirical_flag_rate
from ovcbmr.eval.metrics import auroc, bootstrap_ci


def _entropy(ps):
    """Normalized Shannon entropy of an abnormal-concept probability vector (each in [0,1],
    read as an independent present-probability, renormalized to a distribution)."""
    import numpy as np
    q = np.asarray(ps, dtype="float64")
    s = q.sum()
    if s <= 0:
        return 0.0
    q = q / s
    q = q[q > 0]
    return float(-(q * np.log(q)).sum() / math.log(len(ps)))


def _scores(emb, heads, vocab, nidx, abn_idx):
    """Return dict of the four per-image scores over an embedding matrix."""
    import numpy as np
    Pm = predict_matrix(emb, heads, vocab)                 # [N, K] present-probabilities
    A = 1.0 - Pm[:, nidx]
    abn = Pm[:, abn_idx]                                   # [N, |abnormal|]
    gap = 1.0 - abn.max(axis=1)
    valve = A * gap
    ent = np.array([_entropy(row) for row in abn])
    return {"valve": valve, "A": A, "gap": gap, "entropy": ent}


def _load(emb_dir, proc, split, heads, vocab, nidx, abn_idx):
    """(scores dict, class per case in {'known','oov','normal'}, patient) for a split."""
    import numpy as np
    import torch
    p = os.path.join(emb_dir, f"{split}_img.pt")
    if not os.path.exists(p):
        return None
    d = torch.load(p, weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    emb, cls, pat = [], [], []
    for i, iid in enumerate(d["ids"]):
        it = items.get(iid)
        if it is None:
            continue
        lk = it["labels_known"]
        normal = int(lk[nidx]) == 1
        any_known_abn = any(int(lk[j]) == 1 for j in abn_idx)
        emb.append(d["emb"][i].numpy())
        cls.append("normal" if normal else ("known" if any_known_abn else "oov"))
        pat.append(it.get("patient_id", iid))
    emb = np.asarray(emb)
    sc = _scores(emb, heads, vocab, nidx, abn_idx)
    return sc, np.array(cls), pat


def main():
    cfg = parse_config_arg()
    ch = getattr(cfg, "cross_hospital", None)
    if ch is None:
        print("[s24] config needs a cross_hospital block."); return
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    nidx = vocab.index("Normal")
    abn_idx = [i for i, c in enumerate(vocab) if c != "Normal"]
    alpha = float(getattr(cfg.conformal, "alpha_flag", 0.05))
    n_boot = int(getattr(getattr(cfg, "eval", object()), "bootstrap", 1000))
    seed = int(getattr(cfg, "seed", 0))

    nih_cal = _load(ch.nih_embeddings, ch.nih_processed, "calib", heads, vocab, nidx, abn_idx)
    vin_te = _load(cfg.paths.embeddings, cfg.paths.processed, "test", heads, vocab, nidx, abn_idx)
    if nih_cal is None or vin_te is None:
        print("[s24] missing an embeddings split."); return

    import numpy as np
    cal_sc, cal_cls, _ = nih_cal
    te_sc, te_cls, te_pat = vin_te
    known = te_cls == "known"
    oov = te_cls == "oov"
    print(f"[s24] NIH calib known-abn n={int((cal_cls=='known').sum())} | "
          f"VinDr test known={int(known.sum())} oov={int(oov.sum())} | alpha={alpha}\n")
    print(f"{'score':8}{'q_hat':>9}{'VinDr ffr(known)':>18}{'OOV flag':>10}{'sep AUROC [CI]':>26}")

    out = {"alpha": alpha, "scores": {}}
    for name in ("valve", "A", "gap", "entropy"):
        cal_known = cal_sc[name][cal_cls == "known"]
        q = split_conformal_threshold(cal_known.tolist(), alpha)     # one-sided upper quantile
        ffr = empirical_flag_rate(te_sc[name][known].tolist(), q)     # known-abn flagged (false deferral)
        oov_flag = empirical_flag_rate(te_sc[name][oov].tolist(), q)  # OOV-abn flagged (the open-set job)
        # separation AUROC: OOV (1) vs in-vocab known (0), among abnormal only
        s = np.concatenate([te_sc[name][oov], te_sc[name][known]])
        y = np.concatenate([np.ones(oov.sum()), np.zeros(known.sum())])
        au = auroc(s.tolist(), y.tolist())
        _, lo, hi = bootstrap_ci(list(zip(s.tolist(), y.tolist())),
                                 statistic=lambda v: auroc([a for a, _ in v], [b for _, b in v]),
                                 n_boot=n_boot, seed=seed)
        print(f"{name:8}{q:9.4f}{ffr:18.3f}{oov_flag:10.3f}{au:12.3f} [{lo:.3f},{hi:.3f}]")
        out["scores"][name] = {"q_hat": q, "vindr_ffr_known": ffr, "oov_flag_rate": oov_flag,
                               "sep_auroc": au, "sep_auroc_ci": [lo, hi]}

    v = out["scores"]["valve"]["sep_auroc"]
    best_simple = max(out["scores"]["A"]["sep_auroc"], out["scores"]["gap"]["sep_auroc"])
    print("\n[s24] VERDICT:")
    if v > best_simple + 0.01:
        print(f"  valve sep-AUROC {v:.3f} beats the best simple score {best_simple:.3f} -> "
              f"the product form earns itself; keep the open-set framing.")
    else:
        print(f"  valve sep-AUROC {v:.3f} does NOT beat the best simple score {best_simple:.3f} -> "
              f"the product form adds little; present s(x) as a difficulty score, not an open-set detector.")

    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, "s24_valve_ablation.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[s24] wrote {op}")


if __name__ == "__main__":
    main()
