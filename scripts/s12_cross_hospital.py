"""[cross-hospital pilot] Does cross-hospital shift break naive conformal? NIH (US) vs VinDr
(Vietnam), leave-one-hospital-out, scoring BOTH hospitals with the shared NIH-trained heads.
This is the cheap experiment that decides whether to pursue the full CXR-LT acquisition.
Runs on cached embeddings (no GPU).

    python -m scripts.s12_cross_hospital --config configs/vindr.yaml
"""
from __future__ import annotations
import json
import os

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.concept.valve import image_unexplained
from ovcbmr.ood.conformal import split_conformal_threshold, empirical_flag_rate


def _abnormal_scores(emb_dir, proc, heads, vocab, split):
    """Valve scores s(x) for ABNORMAL (not No-finding) images of one hospital/split, using the
    shared NIH-trained heads."""
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
        if nidx >= 0 and it["labels_known"][nidx] == 1:        # skip No-finding (normal)
            continue
        out.append(image_unexplained(Pm[i].tolist(), vocab, "Normal")[1])
    return out


def main():
    cfg = parse_config_arg()
    ch = getattr(cfg, "cross_hospital", None)
    if ch is None:
        print("[s12] config needs a cross_hospital: {nih_processed, nih_embeddings} block."); return
    hpath = os.path.join(ch.nih_processed, "concept_heads.json")
    if not os.path.exists(hpath):
        print(f"[s12] {hpath} missing — run s09_train_heads on NIH first."); return
    heads = load_heads(hpath)
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]

    nih_cal = _abnormal_scores(ch.nih_embeddings, ch.nih_processed, heads, vocab, "calib")
    nih_te = _abnormal_scores(ch.nih_embeddings, ch.nih_processed, heads, vocab, "test")
    vin_cal = _abnormal_scores(cfg.paths.embeddings, cfg.paths.processed, heads, vocab, "calib")
    vin_te = _abnormal_scores(cfg.paths.embeddings, cfg.paths.processed, heads, vocab, "test")
    alpha = float(getattr(cfg.conformal, "alpha_flag", 0.05))
    print(f"[s12] abnormal counts | NIH cal/test={len(nih_cal)}/{len(nih_te)} "
          f"VinDr cal/test={len(vin_cal)}/{len(vin_te)} | alpha={alpha}")
    if min(len(nih_cal), len(nih_te), len(vin_cal), len(vin_te)) < 30:
        print("[s12] not enough abnormal images in one pool (check VinDr manifest/encoding)."); return

    def deploy(cal_same, cal_other, test):
        return (empirical_flag_rate(test, split_conformal_threshold(cal_other, alpha)),  # naive cross
                empirical_flag_rate(test, split_conformal_threshold(cal_same, alpha)))    # group-cond

    fn_v, fg_v = deploy(vin_cal, nih_cal, vin_te)   # calibrate on NIH -> deploy on VinDr
    fn_n, fg_n = deploy(nih_cal, vin_cal, nih_te)   # calibrate on VinDr -> deploy on NIH
    print(f"\n{'deploy on':<10} {'naive(cross-hospital)':>22} {'group-conditional':>18}  (target {alpha})")
    print(f"{'VinDr':<10} {fn_v:>22.3f} {fg_v:>18.3f}")
    print(f"{'NIH':<10} {fn_n:>22.3f} {fg_n:>18.3f}")

    violated = (fn_v > alpha + 0.02) or (fn_n > alpha + 0.02)
    print("\n[s12] VERDICT:")
    if violated:
        print("  naive cross-hospital VIOLATES alpha -> the safety story is REAL.")
        print("  -> pursue the full CXR-LT/PadChest acquisition for the polished benchmark. FULL PAPER.")
    else:
        print("  naive cross-hospital stays <= alpha -> shift benign even across hospitals.")
        print("  -> reframe to the honest workshop/short paper; skip the CXR-LT slog.")
    print(f"  group-conditional restores exact coverage: NIH {fg_n:.3f}, VinDr {fg_v:.3f} (~{alpha})")

    os.makedirs(cfg.paths.results, exist_ok=True)
    with open(os.path.join(cfg.paths.results, "s12_cross_hospital.json"), "w") as f:
        json.dump({"alpha": alpha,
                   "n": {"nih_cal": len(nih_cal), "nih_test": len(nih_te),
                         "vindr_cal": len(vin_cal), "vindr_test": len(vin_te)},
                   "deploy_vindr": {"naive_cross": fn_v, "groupcond": fg_v},
                   "deploy_nih": {"naive_cross": fn_n, "groupcond": fg_n},
                   "naive_violates_alpha": violated}, f, indent=2)
    print(f"[s12] wrote {os.path.join(cfg.paths.results, 's12_cross_hospital.json')}")


if __name__ == "__main__":
    main()
