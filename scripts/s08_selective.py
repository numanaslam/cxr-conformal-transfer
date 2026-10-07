"""[E2] Selective prediction on the abnormal-vs-normal triage task (NIH cache, no GPU).

Predictor: abnormality A(x) = 1 - p(Normal) (Platt-calibrated), thresholded at tau* (chosen on
calib for balanced accuracy). Confidence = |A - tau*| (distance from the decision boundary).
Abstaining on low-confidence cases improves accuracy on the accepted cases (risk-coverage /
AURC). Conformal Risk Control (G2) picks an abstention threshold lambda that provably bounds the
rate of CONFIDENT MISTAKES on the whole stream: E[accept AND error] <= alpha (finite-sample).

    python -m scripts.s08_selective --config configs/nih.yaml
"""
from __future__ import annotations
import json
import os

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.calibrate import apply_platt, load_bank as load_platt
from ovcbmr.eval.metrics import auroc, risk_coverage_curve, aurc
from ovcbmr.ood.conformal import crc_threshold


def main():
    cfg = parse_config_arg()
    import numpy as np
    import torch

    emb_dir, proc = cfg.paths.embeddings, cfg.paths.processed
    text = torch.load(os.path.join(emb_dir, "concept_text.pt"), weights_only=False)
    calib = torch.load(os.path.join(emb_dir, "calib_img.pt"), weights_only=False)
    test = torch.load(os.path.join(emb_dir, "test_img.pt"), weights_only=False)
    platt = load_platt(os.path.join(proc, "platt.json"))
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    with open(os.path.join(proc, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]

    concepts = text["concepts"]
    pos = text["pos"].numpy()
    if "Normal" not in concepts:
        print("[s08] no 'Normal' concept — abnormal/normal task needs it."); return
    nidx = concepts.index("Normal")
    wN, bN = platt.get("Normal", (1.0, 0.0))
    ab_cols = [j for j, c in enumerate(vocab) if c != "Normal"]

    def abnormality_and_label(dp):
        emb = dp["emb"].numpy()
        simN = emb @ pos[nidx]                                   # cosine to Normal, [M]
        A = np.array([1.0 - apply_platt(float(s), wN, bN) for s in simN])
        y = np.array([1 if any(items[i]["labels_known"][j] for j in ab_cols) else 0
                      for i in dp["ids"]], dtype=int)
        return A, y

    Ac, yc = abnormality_and_label(calib)
    At, yt = abnormality_and_label(test)

    # operating point tau* on calib (balanced accuracy)
    P, N = int(yc.sum()), int(len(yc) - yc.sum())
    ts = np.linspace(float(Ac.min()), float(Ac.max()), 201)
    ba = [0.5 * (((Ac >= t) & (yc == 1)).sum() / max(P, 1) + ((Ac < t) & (yc == 0)).sum() / max(N, 1))
          for t in ts]
    tau = float(ts[int(np.argmax(ba))])

    def loss_conf(A, y):
        yhat = (A >= tau).astype(int)
        return (yhat != y).astype(float), np.abs(A - tau)

    loss_c, conf_c = loss_conf(Ac, yc)
    loss_t, conf_t = loss_conf(At, yt)

    base_auroc = auroc(At.tolist(), yt.tolist())
    full_err = float(loss_t.mean())
    au = aurc(loss_t.tolist(), conf_t.tolist())
    print(f"[s08] abnormal-vs-normal | tau*={tau:.3f} | test AUROC={base_auroc:.3f} | "
          f"full error (no abstention)={full_err:.3f} | AURC={au:.4f} (lower=better)")

    # CRC (G2): control E[accept AND error] <= alpha_risk, choosing the smallest lambda (max coverage)
    alpha = float(getattr(cfg.conformal, "alpha_risk", 0.10))
    lambdas = sorted(set(np.quantile(conf_c, np.linspace(0, 1, 201)).tolist()))
    risks = [float(((conf_c >= lam) * loss_c).mean()) for lam in lambdas]   # non-increasing in lam
    lam_hat, _ = crc_threshold(risks, lambdas, n=len(conf_c), alpha=alpha, B=1.0)
    if lam_hat is None:
        lam_hat = lambdas[-1]

    acc = conf_t >= lam_hat
    coverage = float(acc.mean())
    accept_err_rate = float((acc * loss_t).mean())                # E[accept & error] on test (the guaranteed quantity)
    sel_err = float(loss_t[acc].mean()) if acc.sum() > 0 else 0.0  # E[error | accept]
    print(f"[s08] CRC @ alpha={alpha}: lambda={lam_hat:.4f} | coverage={coverage:.2f} | "
          f"accept&error rate={accept_err_rate:.3f} (<= {alpha}? {'YES' if accept_err_rate <= alpha + 1e-9 else 'NO'})")
    print(f"[s08] selective error on accepted={sel_err:.3f}  vs  full error={full_err:.3f}  "
          f"(abstained {100*(1-coverage):.0f}% of cases)")

    os.makedirs(cfg.paths.results, exist_ok=True)
    rc = risk_coverage_curve(loss_t.tolist(), conf_t.tolist())
    out = {"task": "abnormal_vs_normal", "tau": tau, "auroc": base_auroc, "full_error": full_err,
           "aurc": au, "alpha_risk": alpha, "lambda": lam_hat, "coverage": coverage,
           "accept_error_rate": accept_err_rate, "selective_error": sel_err,
           "risk_coverage": [{"coverage": c, "risk": r} for c, r in rc]}
    with open(os.path.join(cfg.paths.results, "s08_selective.json"), "w") as f:
        json.dump(out, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cs = [c for c, _ in rc]; rs = [r for _, r in rc]
        plt.figure(figsize=(5, 4))
        plt.plot(cs, rs, "-", label="selective risk (error on accepted)")
        plt.axhline(full_err, ls="--", color="gray", label=f"full error {full_err:.2f}")
        plt.scatter([coverage], [sel_err], color="red", zorder=5, label=f"CRC op-point (alpha={alpha})")
        plt.xlabel("coverage"); plt.ylabel("risk (error on accepted)")
        plt.title("Selective prediction: abnormal vs normal (NIH)")
        plt.legend(); plt.tight_layout()
        figp = os.path.join(cfg.paths.results, "selective_risk_coverage.png")
        plt.savefig(figp, dpi=150)
        print(f"[s08] figure -> {figp}")
    except Exception as e:                                        # noqa: BLE001
        print(f"[s08] plot skipped ({e}); curve is in s08_selective.json")


if __name__ == "__main__":
    main()
