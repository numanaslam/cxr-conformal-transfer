"""[build 3] Fit the G1 conformal safety-valve threshold on the calibration split.

For each KNOWN-ABNORMAL calib image: calibrated concept probs p_k = sigmoid(w_k*sim_k + b_k)
(Platt), abnormality A = 1 - p(Normal), anchoring a_k = p_k (Normal excluded), unexplained
s(x) = A*(1 - max_k a_k). Then split-conformal q_hat so that
    P( false UNANCHORED flag on a known finding ) <= conformal.alpha_flag   (finite-sample).

Saves processed/conformal.json. (G2 selective-deferral via CRC is applied in s06, where a
disease-level risk exists.) Runs on the synthetic cache for a dry run:

    python -m scripts.s04_fit_conformal --config configs/synth.yaml
"""
from __future__ import annotations
import json
import os
import sys

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.calibrate import apply_platt, load_bank as load_platt
from ovcbmr.concept.valve import image_unexplained
from ovcbmr.ood.conformal import split_conformal_threshold, empirical_flag_rate


def main():
    cfg = parse_config_arg()
    import torch

    emb_dir, proc = cfg.paths.embeddings, cfg.paths.processed
    for p in ("concept_text.pt", "calib_img.pt"):
        if not os.path.exists(os.path.join(emb_dir, p)):
            print(f"[s04] missing {p} — run s02 first."); sys.exit(1)
    if not os.path.exists(os.path.join(proc, "platt.json")):
        print("[s04] missing platt.json — run s03 first."); sys.exit(1)

    text = torch.load(os.path.join(emb_dir, "concept_text.pt"), weights_only=False)
    calib = torch.load(os.path.join(emb_dir, "calib_img.pt"), weights_only=False)
    platt = load_platt(os.path.join(proc, "platt.json"))
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    with open(os.path.join(proc, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]

    concepts = text["concepts"]
    sims = (calib["emb"] @ text["pos"].t()).tolist()          # [N, K] cosine logits
    ids = calib["ids"]
    normal = "Normal" if "Normal" in concepts else None
    ab_cols = [j for j, c in enumerate(vocab) if c != "Normal"]  # abnormal label columns

    scores = []
    for row, iid in enumerate(ids):
        it = items.get(iid)
        if it is None:
            continue
        labels = it["labels_known"]
        if not any(labels[j] for j in ab_cols):               # keep only known-abnormal cases
            continue
        p = [apply_platt(sims[row][k], *platt.get(c, (1.0, 0.0)))
             for k, c in enumerate(concepts)]
        _, s = image_unexplained(p, concepts, normal_name=(normal or "Normal"))
        scores.append(s)

    if not scores:
        print("[s04] no known-abnormal calib images — check labels/splits."); sys.exit(1)

    alpha = float(getattr(cfg.conformal, "alpha_flag", 0.05))
    q_hat = split_conformal_threshold(scores, alpha)
    finite = q_hat != float("inf")
    rate = empirical_flag_rate(scores, q_hat)
    out = {"q_hat": q_hat if finite else None, "q_hat_finite": finite, "alpha_flag": alpha,
           "empirical_flag_rate": rate, "n_calib_abnormal": len(scores),
           "normal_concept": normal,
           "abnormality_def": "1 - p(Normal)" if normal else "max_k p_k"}
    path = os.path.join(proc, "conformal.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[s04] G1 on {len(scores)} known-abnormal calib images | alpha={alpha}")
    if finite:
        print(f"[s04] q_hat={q_hat:.4f} | empirical false-flag rate={rate:.3f} (<= {alpha})")
    else:
        need = int(1.0 / alpha) - 1
        print(f"[s04] q_hat=inf — calib too small for alpha={alpha} "
              f"(need >= {need} abnormal calib, have {len(scores)}). Expected on synth; "
              "finite on real data.")
    print(f"[s04] wrote {path}")


if __name__ == "__main__":
    main()
