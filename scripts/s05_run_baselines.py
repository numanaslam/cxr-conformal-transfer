"""[build 4] Embedding-space OOD baselines on cached embeddings — the DIAGNOSTIC.

Question: does ANY method separate novel (unseen) findings from known findings, or is the
dataset's entanglement the ceiling? Runs on cached embeddings only (no GPU / no re-encode).
Reports, for each baseline, Axis-A AUROC (any-unseen vs rest) and the honest clean-subset
AUROC (unseen-only vs known-only), alongside the concept valve s(x).

Baselines (all scored so higher = more OOD/unanchored): Mahalanobis, KNN, JointEnergy, MCM,
MSP-margin (protocol §3). NegLabel needs a negative-text bank (omitted here).

    python -m scripts.s05_run_baselines --config configs/nih.yaml
"""
from __future__ import annotations
import json
import os

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.calibrate import apply_platt, load_bank as load_platt
from ovcbmr.concept.valve import image_unexplained
from ovcbmr.ood import baselines as BL
from ovcbmr.eval.metrics import auroc


def main():
    cfg = parse_config_arg()
    import numpy as np
    import torch

    emb_dir, proc = cfg.paths.embeddings, cfg.paths.processed
    text = torch.load(os.path.join(emb_dir, "concept_text.pt"), weights_only=False)
    tr = torch.load(os.path.join(emb_dir, "train_img.pt"), weights_only=False)
    te = torch.load(os.path.join(emb_dir, "test_img.pt"), weights_only=False)
    platt = load_platt(os.path.join(proc, "platt.json"))
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    with open(os.path.join(proc, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]

    concepts = text["concepts"]
    pos = text["pos"].numpy()
    tr_emb = tr["emb"].numpy()
    te_emb = te["emb"].numpy()
    te_sims = te_emb @ pos.T                                   # [N, K] cosine
    te_ids = te["ids"]
    vcol = {c: i for i, c in enumerate(vocab)}
    ab_cols = [j for j, c in enumerate(vocab) if c != "Normal"]

    # labels: any-unseen (Axis A) + clean subset (unseen-only vs known-only)
    Y, clean_y, keep = [], [], []
    for iid in te_ids:
        it = items[iid]
        y = 1 if any(it["labels_unseen"]) else 0
        hk = any(it["labels_known"][j] for j in ab_cols)
        Y.append(y)
        if y and not hk:
            clean_y.append(1); keep.append(True)
        elif (not y) and hk:
            clean_y.append(0); keep.append(True)
        else:
            clean_y.append(0); keep.append(False)

    # per-image Platt logits + probs
    logits, probs = [], []
    for row in range(len(te_ids)):
        lg = [platt.get(c, (1.0, 0.0))[0] * float(te_sims[row][k]) + platt.get(c, (1.0, 0.0))[1]
              for k, c in enumerate(concepts)]
        logits.append(lg)
        probs.append([apply_platt(float(te_sims[row][k]), *platt.get(c, (1.0, 0.0)))
                      for k, c in enumerate(concepts)])

    scores = {}
    scores["valve_s(x)"] = [image_unexplained(p, concepts, "Normal")[1] for p in probs]
    scores["JointEnergy"] = [-BL.joint_energy(lg) for lg in logits]
    scores["MCM"] = [BL.mcm_score(list(te_sims[row])) for row in range(len(te_ids))]
    scores["MSP-margin"] = [BL.msp_margin(lg) for lg in logits]

    # Mahalanobis (per-known-concept means on train)
    masks = {c: np.array([items[iid]["labels_known"][vcol[c]] == 1 for iid in tr["ids"]])
             for c in vocab}
    _, means, prec = BL.fit_mahalanobis(tr_emb, masks)
    scores["Mahalanobis"] = BL.mahalanobis_score(te_emb, means, prec).tolist()

    # KNN (kth-NN distance to a subsampled train bank)
    rng = np.random.default_rng(0)
    bank = tr_emb[rng.choice(tr_emb.shape[0], size=min(20000, tr_emb.shape[0]), replace=False)]
    scores["KNN"] = BL.knn_score_batch(te_emb, bank, k=int(getattr(cfg.ood, "knn_k", 50))).tolist()

    n_clean1 = sum(1 for y, k in zip(clean_y, keep) if k and y)
    n_clean0 = sum(keep) - n_clean1
    print(f"[s05] test={len(te_ids)} | unanchored(any)={sum(Y)} | "
          f"clean subset: {n_clean1} unseen-only vs {n_clean0} known-only")
    print(f"{'baseline':<13} {'AnyAUROC':>9} {'CleanAUROC':>11}")
    print("-" * 35)
    results = {}
    for name, sc in scores.items():
        a_any = auroc(sc, Y)
        cs = [s for s, k in zip(sc, keep) if k]
        cl = [y for y, k in zip(clean_y, keep) if k]
        a_clean = auroc(cs, cl)
        results[name] = {"any": a_any, "clean": a_clean}
        print(f"{name:<13} {a_any:>9.3f} {a_clean:>11.3f}")

    best = max(results, key=lambda n: results[n]["clean"])
    print("-" * 35)
    print(f"[s05] best clean-AUROC: {best} = {results[best]['clean']:.3f} "
          f"(chance=0.5; valve_s(x)={results['valve_s(x)']['clean']:.3f})")
    os.makedirs(cfg.paths.results, exist_ok=True)
    outp = os.path.join(cfg.paths.results, "s05_baselines.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[s05] wrote {outp}")


if __name__ == "__main__":
    main()
