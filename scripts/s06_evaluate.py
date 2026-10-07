"""[build 5, first cut] Evaluate the safety valve on the test split — Axis A: unanchored
detection. Computes s(x) per test image and scores how well it separates images carrying a
held-out (unseen) finding from the rest, plus the valve operating point at the G1 q_hat.

    python -m scripts.s06_evaluate --config configs/nih.yaml

(Full Settings A-D + baseline comparison + DeLong come once s05 is in. This is the headline
preliminary number: does the concept-level unexplained statistic detect novel findings?)
"""
from __future__ import annotations
import json
import os

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.calibrate import apply_platt, load_bank as load_platt
from ovcbmr.concept.valve import image_unexplained
from ovcbmr.eval.metrics import auroc, fpr_at_tpr, bootstrap_ci


def _safe_auroc(pairs):
    a = auroc([s for s, _ in pairs], [y for _, y in pairs])
    return a if a == a else 0.5           # guard nan (degenerate bootstrap sample)


def main():
    cfg = parse_config_arg()
    import torch

    emb_dir, proc = cfg.paths.embeddings, cfg.paths.processed
    text = torch.load(os.path.join(emb_dir, "concept_text.pt"), weights_only=False)
    test = torch.load(os.path.join(emb_dir, "test_img.pt"), weights_only=False)
    platt = load_platt(os.path.join(proc, "platt.json"))
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    with open(os.path.join(proc, "meta.json")) as f:
        meta = json.load(f)
    vocab = meta["vocabulary"]
    conf_path = os.path.join(proc, "conformal.json")
    conf = json.load(open(conf_path)) if os.path.exists(conf_path) else {}
    q_hat = conf.get("q_hat")

    concepts = text["concepts"]
    sims = (test["emb"] @ text["pos"].t()).tolist()
    ids = test["ids"]
    normal = "Normal" if "Normal" in concepts else "Normal"
    ab_vocab = [j for j, c in enumerate(vocab) if c != "Normal"]

    S, Y, PID = [], [], []
    clean = []                              # unseen-only(+1) vs known-abnormal-only(0)
    for row, iid in enumerate(ids):
        it = items.get(iid)
        if it is None:
            continue
        p = [apply_platt(sims[row][k], *platt.get(c, (1.0, 0.0)))
             for k, c in enumerate(concepts)]
        _, s = image_unexplained(p, concepts, normal_name=normal)
        y = 1 if any(it["labels_unseen"]) else 0
        has_known_ab = any(it["labels_known"][j] for j in ab_vocab)
        S.append(s); Y.append(y); PID.append(it.get("patient_id"))
        if y and not has_known_ab:
            clean.append((s, 1))
        elif (not y) and has_known_ab:
            clean.append((s, 0))

    n, npos = len(S), sum(Y)
    print(f"[s06] test images={n} | unanchored (any unseen)={npos} ({100*npos/max(1,n):.1f}%)")

    au = auroc(S, Y)
    fpr = fpr_at_tpr(S, Y, tpr=float(getattr(cfg.eval, "fpr_at_tpr", 0.95)))
    _, lo, hi = bootstrap_ci(list(zip(S, Y)), statistic=_safe_auroc,
                             n_boot=int(getattr(cfg.eval, "bootstrap", 1000)),
                             groups=PID, seed=0)
    print(f"[s06] Axis-A unanchored AUROC = {au:.3f}  (95% CI {lo:.3f}-{hi:.3f}, patient-bootstrap)")
    print(f"[s06] FPR@95TPR = {fpr:.3f}")

    if any(y for _, y in clean) and any(not y for _, y in clean):
        auc_clean = auroc([s for s, _ in clean], [y for _, y in clean])
        n1 = sum(y for _, y in clean)
        print(f"[s06] clean-subset AUROC (unseen-only vs known-only) = {auc_clean:.3f} "
              f"[{n1} vs {len(clean)-n1}] — removes the easy Normal signal")

    result = {"n_test": n, "n_unanchored": npos, "auroc": au, "auroc_ci": [lo, hi], "fpr95": fpr}
    if q_hat is not None:
        flagged = [(s > q_hat) for s in S]
        tp = sum(1 for f, y in zip(flagged, Y) if f and y)
        fp = sum(1 for f, y in zip(flagged, Y) if f and not y)
        nflag = sum(flagged)
        recall = tp / npos if npos else 0.0
        prec = tp / nflag if nflag else 0.0
        fprate = fp / (n - npos) if (n - npos) else 0.0
        print(f"[s06] valve @ q_hat={q_hat:.4f}: flags {nflag}/{n} | recall={recall:.2f} "
              f"precision={prec:.2f} false-flag={fprate:.3f}")
        result.update({"q_hat": q_hat, "recall": recall, "precision": prec, "false_flag": fprate})

    os.makedirs(cfg.paths.results, exist_ok=True)
    outp = os.path.join(cfg.paths.results, "axisA_unanchored.json")
    with open(outp, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[s06] wrote {outp}")


if __name__ == "__main__":
    main()
