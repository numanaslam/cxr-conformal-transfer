"""[E4] Interpretability on the trained-head CBM (ADR-0005): per-concept reliability, and
abstention ATTRIBUTION — when the model is uncertain, which concept is the reason. NIH cache, no GPU.

    python -m scripts.s10_interpretability --config configs/nih.yaml
"""
from __future__ import annotations
import json
import os
from collections import Counter

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.eval.metrics import auroc, expected_calibration_error


def main():
    cfg = parse_config_arg()
    import numpy as np
    import torch

    emb_dir, proc = cfg.paths.embeddings, cfg.paths.processed
    hp = os.path.join(proc, "concept_heads.json")
    if not os.path.exists(hp):
        print("[s10] concept_heads.json missing — run s09_train_heads first."); return
    heads = load_heads(hp)
    te = torch.load(os.path.join(emb_dir, "test_img.pt"), weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    with open(os.path.join(proc, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]

    ids = te["ids"]
    emb = te["emb"].numpy()
    Y = np.array([[items[i]["labels_known"][j] for j in range(len(vocab))] for i in ids], dtype=int)
    P = predict_matrix(emb, heads, vocab)                    # [N, K] present-probabilities

    # ---- per-concept reliability (interpretable, individually meaningful concepts) ----
    print(f"{'concept':<20} {'AUROC':>7} {'ECE':>6}")
    rel = {}
    for j, c in enumerate(vocab):
        if c not in heads:
            continue
        a = auroc(P[:, j].tolist(), Y[:, j].tolist())
        e = expected_calibration_error(P[:, j].tolist(), Y[:, j].tolist())
        rel[c] = {"auroc": a, "ece": e}
        print(f"{c:<20} {a:>7.3f} {e:>6.3f}")

    # ---- abstention attribution ----
    # per image: driver = the concept closest to its decision boundary (most on-the-fence);
    # image uncertainty = that concept's margin. Abstain on the most-uncertain 20%.
    # attribute to the most-uncertain FINDING (exclude Normal — the aggregate abnormal/normal
    # call dominates otherwise; that's a separate, coarser signal)
    margin = np.abs(P - 0.5)
    if "Normal" in vocab:
        margin[:, vocab.index("Normal")] = 1e9
    driver = margin.argmin(axis=1)
    img_unc = margin.min(axis=1)
    thr = float(np.quantile(img_unc, 0.20))
    abst = img_unc <= thr
    drv = Counter(vocab[int(driver[i])] for i in range(len(ids)) if abst[i])
    print(f"\n[s10] abstention attribution (most-uncertain {int(abst.sum())} images = bottom 20%):")
    print(f"{'driver concept':<20} {'count':>6} {'share':>7}")
    for c, ct in drv.most_common():
        print(f"{c:<20} {ct:>6} {100*ct/max(1,int(abst.sum())):>6.1f}%")

    # ---- qualitative examples (most uncertain) ----
    print("\n[s10] examples (most-uncertain images): id | driver concept (p) | true")
    for i in np.argsort(img_unc)[:6]:
        c = int(driver[i])
        print(f"  {ids[i]} | {vocab[c]} (p={P[i, c]:.2f}) | true={int(Y[i, c])}")

    os.makedirs(cfg.paths.results, exist_ok=True)
    with open(os.path.join(cfg.paths.results, "s10_interpretability.json"), "w") as f:
        json.dump({"per_concept": rel, "abstention_drivers": dict(drv),
                   "n_abstained": int(abst.sum())}, f, indent=2)
    print(f"\n[s10] wrote {os.path.join(cfg.paths.results, 's10_interpretability.json')}")


if __name__ == "__main__":
    main()
