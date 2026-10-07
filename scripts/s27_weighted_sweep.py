"""[weighted-conformal sweep] The referee's point: with domain-classifier AUROC 0.99 and 95th-pct
clipping, importance-weighted conformal is forced toward the unweighted solution, so the reported
failure could be an artifact of the clipping rather than of the method. This sweeps the classifier
regularization C and the weight clip quantile, and adds a label-free QUANTILE-ALIGNMENT baseline
(shift the deployment valve-score distribution so a low quantile matches the NIH calibration
distribution, then apply the NIH threshold). All measured as VinDr in-vocab G1 false deferral at
alpha, against the naive (0.085) and group-conditional (0.053) references. Cached embeddings, CPU.

    python -m scripts.s27_weighted_sweep --config configs/vindr.yaml
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from anywhere

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.ood.conformal import (split_conformal_threshold, empirical_flag_rate,
                                  estimate_shift_weights, weighted_split_conformal_threshold)


def _valve_known(emb_dir, proc, split, heads, vocab, nidx, abn_idx, known_only):
    """(valve scores, embeddings) for cases; known_only keeps in-vocab known-abnormal."""
    import numpy as np
    import torch
    d = torch.load(os.path.join(emb_dir, f"{split}_img.pt"), weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    E, S = [], []
    for i, iid in enumerate(d["ids"]):
        it = items.get(iid)
        if it is None:
            continue
        lk = it["labels_known"]
        if known_only and not any(int(lk[j]) == 1 for j in abn_idx):
            continue
        e = d["emb"][i].numpy()
        p = predict_matrix(e[None, :], heads, vocab)[0]
        A = 1.0 - p[nidx]
        gap = 1.0 - max(p[j] for j in abn_idx)
        E.append(e); S.append(A * gap)
    return np.asarray(S), np.asarray(E)


def main():
    cfg = parse_config_arg()
    ch = getattr(cfg, "cross_hospital", None)
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    nidx = vocab.index("Normal")
    abn_idx = [i for i, c in enumerate(vocab) if c != "Normal"]
    alpha = float(getattr(cfg.conformal, "alpha_flag", 0.05))

    import numpy as np
    S_cal, E_cal = _valve_known(ch.nih_embeddings, ch.nih_processed, "calib", heads, vocab, nidx, abn_idx, True)
    S_te, E_te = _valve_known(cfg.paths.embeddings, cfg.paths.processed, "test", heads, vocab, nidx, abn_idx, True)
    q_naive = split_conformal_threshold(S_cal.tolist(), alpha)
    ffr_naive = empirical_flag_rate(S_te.tolist(), q_naive)
    print(f"[s27] naive q={q_naive:.4f} ffr={ffr_naive:.3f} (paper 0.085); target={alpha}, group-cond ref 0.053\n")
    print(f"{'C':>6}{'clip_q':>8}{'ESS%':>7}{'q_weighted':>12}{'VinDr ffr':>11}")

    out = {"alpha": alpha, "naive": {"q": q_naive, "ffr": ffr_naive}, "sweep": []}
    for C in (0.01, 0.1, 1.0):
        for clip in (0.90, 0.95, 0.99, 1.0):
            w = estimate_shift_weights(E_cal, E_te, C=C, clip_quantile=clip)
            w = np.asarray(w[0] if isinstance(w, tuple) else w).ravel()
            ess = (w.sum() ** 2) / (np.square(w).sum() + 1e-12)
            qw = weighted_split_conformal_threshold(S_cal.tolist(), w.tolist(), alpha)
            ffr = empirical_flag_rate(S_te.tolist(), qw)
            print(f"{C:>6}{clip:>8.2f}{100*ess/len(w):>7.0f}{qw:>12.4f}{ffr:>11.3f}")
            out["sweep"].append({"C": C, "clip": clip, "ess_frac": ess / len(w), "q": qw, "ffr": ffr})

    # label-free quantile alignment: align the 10th percentile of deployment scores to NIH calib,
    # then apply the naive NIH threshold to the shifted deployment scores.
    shift = np.quantile(S_cal, 0.10) - np.quantile(S_te, 0.10)
    ffr_align = empirical_flag_rate((S_te + shift).tolist(), q_naive)
    print(f"\n[s27] quantile-alignment (label-free): shift={shift:+.4f} -> VinDr ffr={ffr_align:.3f}")
    out["quantile_alignment"] = {"shift": float(shift), "ffr": ffr_align}
    print("[s27] Read: if ffr stays near 0.085 across the whole sweep, the failure is structural "
          "(near-disjoint support), not a clipping artifact; quantile alignment is the honest "
          "label-free comparator to group-conditional 0.053.")

    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, "s27_weighted_sweep.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[s27] wrote {op}")


if __name__ == "__main__":
    main()
