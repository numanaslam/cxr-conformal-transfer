"""[ADR-0005] Trained linear concept heads on frozen embeddings (frozen-encoder CBM), then
re-run E2 selective prediction with a trained abnormal-vs-normal predictor to measure the lift
over zero-shot cosine (s08). Saves processed/concept_heads.json. Runs on cached embeddings (no GPU).

    python -m scripts.s09_train_heads --config configs/nih.yaml
"""
from __future__ import annotations
import json
import os

from ovcbmr.config import parse_config_arg
from ovcbmr.eval.metrics import auroc, aurc, expected_calibration_error
from ovcbmr.ood.conformal import crc_threshold


def main():
    cfg = parse_config_arg()
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression

    emb_dir, proc = cfg.paths.embeddings, cfg.paths.processed
    tr = torch.load(os.path.join(emb_dir, "train_img.pt"), weights_only=False)
    ca = torch.load(os.path.join(emb_dir, "calib_img.pt"), weights_only=False)
    te = torch.load(os.path.join(emb_dir, "test_img.pt"), weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    with open(os.path.join(proc, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]

    def emb_of(dp):
        return dp["emb"].numpy()

    def labels(dp):
        return np.array([[items[i]["labels_known"][j] for j in range(len(vocab))]
                         for i in dp["ids"]], dtype=int)

    Xtr, Ytr = emb_of(tr), labels(tr)
    Xca, Yca = emb_of(ca), labels(ca)
    Xte, Yte = emb_of(te), labels(te)

    # ---- per-concept trained heads (frozen-encoder CBM), Platt-calibrated on calib ----
    heads, aurocs, eces = {}, [], []
    print(f"{'concept':<20} {'pos_train':>9} {'AUROC':>7} {'ECE':>6}")
    for j, c in enumerate(vocab):
        ytr = Ytr[:, j]
        if ytr.sum() < 10 or ytr.sum() == len(ytr):
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced").fit(Xtr, ytr)
        raw_ca, raw_te = clf.decision_function(Xca), clf.decision_function(Xte)
        yca = Yca[:, j]
        if len(np.unique(yca)) < 2:                          # degenerate calib -> identity
            pa, pb = 1.0, 0.0
        else:
            pl = LogisticRegression(max_iter=1000).fit(raw_ca.reshape(-1, 1), yca)
            pa, pb = float(pl.coef_[0][0]), float(pl.intercept_[0])
        p = 1.0 / (1.0 + np.exp(-(pa * raw_te + pb)))         # Platt-calibrated present-prob
        a = auroc(p.tolist(), Yte[:, j].tolist())
        e = expected_calibration_error(p.tolist(), Yte[:, j].tolist())
        heads[c] = {"coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "platt": [pa, pb]}
        aurocs.append(a); eces.append(e)
        print(f"{c:<20} {int(ytr.sum()):>9} {a:>7.3f} {e:>6.3f}")
    mean_auroc = sum(aurocs) / max(1, len(aurocs))
    print(f"[s09] mean per-concept AUROC = {mean_auroc:.3f} | mean ECE = {sum(eces)/max(1,len(eces)):.3f}  "
          f"(K={len(aurocs)}; heads Platt-calibrated)")
    with open(os.path.join(proc, "concept_heads.json"), "w") as f:
        json.dump(heads, f)

    # ---- E2 with a trained abnormal-vs-normal head ----
    ab_cols = [j for j, c in enumerate(vocab) if c != "Normal"]

    def abn(Y):
        return (Y[:, ab_cols].sum(axis=1) > 0).astype(int)

    ytr_ab, yca_ab, yte_ab = abn(Ytr), abn(Yca), abn(Yte)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced").fit(Xtr, ytr_ab)
    pca, pte = clf.predict_proba(Xca)[:, 1], clf.predict_proba(Xte)[:, 1]
    base = auroc(pte.tolist(), yte_ab.tolist())

    P, N = int(yca_ab.sum()), int(len(yca_ab) - yca_ab.sum())
    ts = np.linspace(0.0, 1.0, 201)
    ba = [0.5 * (((pca >= t) & (yca_ab == 1)).sum() / max(P, 1)
                 + ((pca < t) & (yca_ab == 0)).sum() / max(N, 1)) for t in ts]
    tau = float(ts[int(np.argmax(ba))])

    def lc(p, y):
        yh = (p >= tau).astype(int)
        return (yh != y).astype(float), np.abs(p - tau)

    loss_c, conf_c = lc(pca, yca_ab)
    loss_t, conf_t = lc(pte, yte_ab)
    full = float(loss_t.mean())
    au = aurc(loss_t.tolist(), conf_t.tolist())

    alpha = float(getattr(cfg.conformal, "alpha_risk", 0.10))
    lams = sorted(set(np.quantile(conf_c, np.linspace(0, 1, 201)).tolist()))
    risks = [float(((conf_c >= l) * loss_c).mean()) for l in lams]
    lam, _ = crc_threshold(risks, lams, n=len(conf_c), alpha=alpha, B=1.0)
    if lam is None:
        lam = lams[-1]
    acc = conf_t >= lam
    cov = float(acc.mean())
    aer = float((acc * loss_t).mean())
    sel = float(loss_t[acc].mean()) if acc.sum() > 0 else 0.0

    print(f"\n[s09] abnormal-vs-normal TRAINED head: AUROC={base:.3f}  (zero-shot s08 was 0.693) | "
          f"full error={full:.3f}  (s08 was 0.334) | AURC={au:.4f}")
    print(f"[s09] CRC @ alpha={alpha}: coverage={cov:.2f}  (s08 was 0.33) | "
          f"accept&error={aer:.3f} (<= {alpha}? {'YES' if aer <= alpha + 1e-9 else 'NO'}) | "
          f"selective error={sel:.3f} vs full {full:.3f}")

    os.makedirs(cfg.paths.results, exist_ok=True)
    with open(os.path.join(cfg.paths.results, "s09_trained.json"), "w") as f:
        json.dump({"mean_concept_auroc": mean_auroc, "n_concepts": len(aurocs),
                   "abnormal_auroc": base, "tau": tau, "full_error": full, "aurc": au,
                   "alpha": alpha, "coverage": cov, "accept_error_rate": aer,
                   "selective_error": sel}, f, indent=2)
    print("[s09] wrote concept_heads.json + s09_trained.json")


if __name__ == "__main__":
    main()
