"""[local Platt refit] The referee notes a tension: the paper says the accepted case returns a
'calibrated abnormality risk', but Table 1 shows calibration does NOT transfer to VinDr (Normal
ECE 0.030 -> 0.152). Group-conditional deployment already needs a labeled local batch, so
refitting the two Platt scalars per concept on that same VinDr calibration split is free. This
script measures whether a local Platt refit restores calibration on the VinDr test split.

For each concept we take the pre-Platt logit z0 = emb . coef + intercept from the shared head,
fit new (a, b) on the VinDr CALIB split (logit vs local label), and report VinDr TEST ECE under
(i) the source NIH Platt already in the head and (ii) the locally refit Platt. AUROC is unchanged
by any monotone Platt map, so only ECE moves. Cached embeddings, CPU, no GPU.

    python -m scripts.s26_local_platt --config configs/vindr.yaml
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from anywhere

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads
from ovcbmr.concept.calibrate import fit_platt
from ovcbmr.eval.metrics import expected_calibration_error, auroc


def _logits_labels(emb_dir, proc, split, heads, vocab):
    """Per-concept pre-Platt logits z0 and binary labels on a split. Returns
    ({concept: [z0]}, {concept: [y]}) over concepts that have a head."""
    import numpy as np
    import torch
    d = torch.load(os.path.join(emb_dir, f"{split}_img.pt"), weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    ids = [iid for iid in d["ids"] if iid in items]
    E = np.asarray([d["emb"][i].numpy() for i, iid in enumerate(d["ids"]) if iid in items])
    lab = {c: [] for c in vocab}
    for iid in ids:
        lk = items[iid]["labels_known"]
        for j, c in enumerate(vocab):
            lab[c].append(int(lk[j]))
    z0 = {}
    for c in vocab:
        h = heads.get(c)
        if h is None:
            continue
        z0[c] = (E @ np.asarray(h["coef"], dtype="float64") + float(h["intercept"]))
    return z0, {c: lab[c] for c in z0}


def _sigmoid(z):
    import numpy as np
    return 1.0 / (1.0 + np.exp(-z))


def main():
    cfg = parse_config_arg()
    ch = getattr(cfg, "cross_hospital", None)
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]

    z_cal, y_cal = _logits_labels(cfg.paths.embeddings, cfg.paths.processed, "calib", heads, vocab)
    z_te, y_te = _logits_labels(cfg.paths.embeddings, cfg.paths.processed, "test", heads, vocab)

    print("[s26] VinDr test ECE: source NIH Platt vs locally refit Platt (AUROC unchanged)\n")
    print(f"{'concept':18}{'AUROC':>7}{'ECE(NIH Platt)':>16}{'ECE(local)':>12}{'improve':>9}")
    rows, e_src, e_loc = {}, [], []
    for c in z_te:
        h = heads[c]
        a0, b0 = h.get("platt", [1.0, 0.0])
        p_src = _sigmoid(a0 * z_te[c] + b0)
        a1, b1 = fit_platt(z_cal[c].tolist(), y_cal[c])      # refit on VinDr calib
        p_loc = _sigmoid(a1 * z_te[c] + b1)
        au = auroc(z_te[c].tolist(), y_te[c])
        ece0 = expected_calibration_error(p_src.tolist(), y_te[c])
        ece1 = expected_calibration_error(p_loc.tolist(), y_te[c])
        print(f"{c:18}{au:7.3f}{ece0:16.3f}{ece1:12.3f}{ece0 - ece1:+9.3f}")
        rows[c] = {"auroc": au, "ece_source": ece0, "ece_local": ece1}
        e_src.append(ece0); e_loc.append(ece1)
    ms, ml = sum(e_src) / len(e_src), sum(e_loc) / len(e_loc)
    print(f"\n[s26] mean ECE {ms:.3f} -> {ml:.3f} after local refit. "
          f"{'Calibration recovers locally; refit Platt at the site and keep the calibrated-risk claim.' if ml < ms - 0.01 else 'Local refit does not help; drop the calibrated-risk claim.'}")
    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, "s26_local_platt.json")
    with open(op, "w") as f:
        json.dump({"mean_ece_source": ms, "mean_ece_local": ml, "per_concept": rows}, f, indent=2)
    print(f"[s26] wrote {op}")


if __name__ == "__main__":
    main()
