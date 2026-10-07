"""[G2 cross-hospital] Does the selective-prediction (patient-safety) guarantee survive
deployment at a new hospital? G1 (false-abstention) over-coverage just means we defer MORE
known-abnormal cases than promised -- safe but inefficient. The clinically important test
is G2: among ACCEPTED (auto-cleared) cases, is the error / missed-abnormal rate still
bounded when we calibrate on NIH and deploy on VinDr?

Predictor: the shared NIH-trained heads (predict_matrix); abnormality A = 1 - p(Normal).
Task: abnormal(ANY finding, i.e. not No-finding)-vs-normal. The decision threshold tau*
(balanced accuracy) and the CRC threshold lambda are fit on the CALIBRATION hospital and
applied to the TEST hospital. Reports coverage, accept&error (the CRC-bounded quantity, with
a patient-clustered CI), error|accept, and the accepted false-negative (missed-abnormal)
rate. Cached embeddings, no GPU.

    python -m scripts.s16_crc_cross_hospital --config configs/vindr.yaml
"""
from __future__ import annotations
import json
import os

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.ood.conformal import crc_threshold
from ovcbmr.eval.metrics import bootstrap_ci


def _load(emb_dir, proc, split, heads, vocab, nidx):
    """(A, y, patient) for a hospital/split. A = 1 - p(Normal); y = 1 if abnormal (any finding)."""
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
        y.append(0 if int(it["labels_known"][nidx]) == 1 else 1)   # abnormal = NOT No-finding
        pat.append(it.get("patient_id", iid))
    return np.array(A), np.array(y, dtype=int), pat


def _fit(Ac, yc, alpha):
    """tau* (balanced accuracy) and CRC lambda_hat on the calibration hospital."""
    import numpy as np
    P, N = int(yc.sum()), int(len(yc) - yc.sum())
    ts = np.linspace(float(Ac.min()), float(Ac.max()), 201)
    ba = [0.5 * (((Ac >= t) & (yc == 1)).sum() / max(P, 1) + ((Ac < t) & (yc == 0)).sum() / max(N, 1))
          for t in ts]
    tau = float(ts[int(np.argmax(ba))])
    loss_c = ((Ac >= tau).astype(int) != yc).astype(float)
    conf_c = np.abs(Ac - tau)
    lambdas = sorted(set(np.quantile(conf_c, np.linspace(0, 1, 201)).tolist()))
    risks = [float(((conf_c >= lam) * loss_c).mean()) for lam in lambdas]
    lam_hat, _ = crc_threshold(risks, lambdas, n=len(conf_c), alpha=alpha, B=1.0)
    return tau, (lam_hat if lam_hat is not None else lambdas[-1])


def _deploy(dp, tau, lam, n_boot, seed):
    import numpy as np
    A, y, pat = dp
    yhat = (A >= tau).astype(int)
    loss = (yhat != y).astype(float)
    acc = (np.abs(A - tau) >= lam)
    miss = ((yhat == 0) & (y == 1)).astype(float)                 # missed abnormal (false negative)
    _, lo, hi = bootstrap_ci((acc * loss).tolist(), groups=pat, n_boot=n_boot, seed=seed)
    return {"n": int(len(y)), "n_abnormal": int(y.sum()), "coverage": float(acc.mean()),
            "accept_error": float((acc * loss).mean()), "accept_error_ci": [lo, hi],
            "error_given_accept": float(loss[acc].mean()) if acc.sum() else 0.0,
            "accepted_fn_rate": float(miss[acc].mean()) if acc.sum() else 0.0}


def main():
    cfg = parse_config_arg()
    ch = getattr(cfg, "cross_hospital", None)
    if ch is None:
        print("[s16] config needs a cross_hospital block."); return
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
        print("[s16] missing an embeddings split."); return

    tau_n, lam_n = _fit(nih_cal[0], nih_cal[1], alpha)             # NIH-calibrated
    tau_v, lam_v = _fit(vin_cal[0], vin_cal[1], alpha)             # VinDr-calibrated
    rows = [("NIH test  (cal NIH)", nih_te, tau_n, lam_n),         # source baseline (should hold)
            ("VinDr test (cal NIH)", vin_te, tau_n, lam_n),        # NAIVE cross-hospital  <-- key row
            ("VinDr test (cal VinDr)", vin_te, tau_v, lam_v)]      # group-conditional

    print(f"[s16] G2 selective prediction (abnormal-vs-normal) | alpha_risk={alpha} | boot={n_boot}")
    print(f"       tau*/lambda: NIH={tau_n:.3f}/{lam_n:.3f}  VinDr={tau_v:.3f}/{lam_v:.3f}\n")
    print(f"{'deployment':<24}{'cover':>6}{'accept&error [CI]':>26}{'err|acc':>9}{'acc.FN':>8}  <=a")
    out = {"alpha_risk": alpha, "tau": {"nih": tau_n, "vindr": tau_v},
           "lambda": {"nih": lam_n, "vindr": lam_v}, "rows": {}}
    for name, dp, tau, lam in rows:
        r = _deploy(dp, tau, lam, n_boot, seed)
        lo, hi = r["accept_error_ci"]
        ok = "YES" if r["accept_error"] <= alpha + 1e-9 else "NO"
        print(f"{name:<24}{r['coverage']:>6.2f}{r['accept_error']:>11.3f} [{lo:.3f},{hi:.3f}]"
              f"{r['error_given_accept']:>9.3f}{r['accepted_fn_rate']:>8.3f}  {ok}")
        out["rows"][name] = r

    key = out["rows"]["VinDr test (cal NIH)"]
    klo, khi = key["accept_error_ci"]
    print("\n[s16] VERDICT (naive NIH->VinDr G2):")
    if key["accept_error"] > alpha + 1e-9:
        print(f"  accept&error {key['accept_error']:.3f} [{klo:.3f},{khi:.3f}] > {alpha} -> the safety "
              f"guarantee is VIOLATED cross-hospital -> shift is a PATIENT-SAFETY issue (strong headline).")
    else:
        below = khi <= alpha + 1e-9
        print(f"  accept&error {key['accept_error']:.3f} [{klo:.3f},{khi:.3f}] <= {alpha}"
              f"{' (CI entirely below alpha)' if below else ' (CI touches alpha)'} -> G2 HOLDS; "
              f"cross-hospital shift hurts abstention EFFICIENCY (G1), not accepted-case safety.")
        print("  -> reframe: 'shift primarily affects abstention efficiency, not patient-safety risk.'")

    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, "s16_crc_cross_hospital.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[s16] wrote {op}")


if __name__ == "__main__":
    main()
