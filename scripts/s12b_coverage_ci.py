"""[coverage CI] Patient-clustered percentile-bootstrap confidence intervals on the
cross-hospital false-flag rates from s12. Same point estimates as s12 (identical
computation), plus CIs so the headline coverage numbers carry uncertainty. The
bootstrap resamples PATIENTS (cluster bootstrap) at the deployed threshold q_hat, i.e.
the sampling variability of the measured coverage for a fixed calibrated predictor.
Runs on cached embeddings (no GPU).

    python -m scripts.s12b_coverage_ci --config configs/vindr.yaml
"""
from __future__ import annotations
import json
import os

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.concept.valve import image_unexplained
from ovcbmr.ood.conformal import split_conformal_threshold, empirical_flag_rate
from ovcbmr.eval.metrics import bootstrap_ci


def _abnormal(emb_dir, proc, heads, vocab, split):
    """(score, patient_id) for ABNORMAL (not No-finding) images of one hospital/split,
    scored by the shared NIH-trained heads."""
    import torch
    p = os.path.join(emb_dir, f"{split}_img.pt")
    if not os.path.exists(p):
        return []
    d = torch.load(p, weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    Pm = predict_matrix(d["emb"].numpy(), heads, vocab)
    nidx = vocab.index("Normal") if "Normal" in vocab else -1
    out = []
    for i, iid in enumerate(d["ids"]):
        it = items.get(iid)
        if it is None:
            continue
        if nidx >= 0 and it["labels_known"][nidx] == 1:            # skip No-finding
            continue
        s = image_unexplained(Pm[i].tolist(), vocab, "Normal")[1]
        out.append((s, it.get("patient_id", iid)))
    return out


def main():
    cfg = parse_config_arg()
    ch = getattr(cfg, "cross_hospital", None)
    if ch is None:
        print("[s12b] config needs a cross_hospital: {nih_processed, nih_embeddings} block."); return
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    alpha = float(getattr(cfg.conformal, "alpha_flag", 0.05))
    n_boot = int(getattr(getattr(cfg, "eval", object()), "bootstrap", 1000))
    seed = int(getattr(cfg, "seed", 0))

    nih_cal = _abnormal(ch.nih_embeddings, ch.nih_processed, heads, vocab, "calib")
    nih_te = _abnormal(ch.nih_embeddings, ch.nih_processed, heads, vocab, "test")
    vin_cal = _abnormal(cfg.paths.embeddings, cfg.paths.processed, heads, vocab, "calib")
    vin_te = _abnormal(cfg.paths.embeddings, cfg.paths.processed, heads, vocab, "test")
    if min(len(nih_cal), len(nih_te), len(vin_cal), len(vin_te)) < 30:
        print("[s12b] not enough abnormal images in one pool (check VinDr manifest/encoding)."); return

    def _flag(sp, q):
        return empirical_flag_rate([s for s, _ in sp], q)

    def _ci(sp, q):
        vals = [1.0 if s > q else 0.0 for s, _ in sp]
        pats = [p for _, p in sp]
        _, lo, hi = bootstrap_ci(vals, groups=pats, n_boot=n_boot, seed=seed)
        return lo, hi

    def cell(cal_same, cal_other, test_sp):
        q_naive = split_conformal_threshold([s for s, _ in cal_other], alpha)   # calibrate on source
        q_grp = split_conformal_threshold([s for s, _ in cal_same], alpha)      # calibrate on deploy site
        return {"naive": {"rate": _flag(test_sp, q_naive), "ci": _ci(test_sp, q_naive)},
                "groupcond": {"rate": _flag(test_sp, q_grp), "ci": _ci(test_sp, q_grp)},
                "n_test": len(test_sp), "n_patients": len({p for _, p in test_sp})}

    dv = cell(vin_cal, nih_cal, vin_te)   # deploy on VinDr (calib NIH)
    dn = cell(nih_cal, vin_cal, nih_te)   # deploy on NIH (calib VinDr)

    def fmt(c):
        r, (lo, hi) = c["rate"], c["ci"]
        return f"{r:.3f} [{lo:.3f},{hi:.3f}]"

    print(f"[s12b] alpha={alpha} | bootstrap={n_boot} (patient-clustered) | seed={seed}\n")
    print(f"{'deploy on':<8}{'test/pat':>13}{'naive (cross)':>24}{'group-conditional':>24}   target")
    for name, c in (("VinDr", dv), ("NIH", dn)):
        print(f"{name:<8}{c['n_test']:>6}/{c['n_patients']:<6}{fmt(c['naive']):>24}{fmt(c['groupcond']):>24}   {alpha}")

    os.makedirs(cfg.paths.results, exist_ok=True)
    out = os.path.join(cfg.paths.results, "s12b_coverage_ci.json")
    with open(out, "w") as f:
        json.dump({"alpha": alpha, "n_boot": n_boot, "seed": seed,
                   "deploy_vindr": dv, "deploy_nih": dn}, f, indent=2)
    print(f"\n[s12b] wrote {out}")


if __name__ == "__main__":
    main()
