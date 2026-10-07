"""[sample-efficiency] How many locally-labelled calibration cases does site-conditional
conformal need to restore the guarantee at a new hospital? Subsample the VinDr (deployment)
in-vocabulary known-abnormal calibration pool at increasing sizes m; for each m, over K
random subsets, compute q_hat and the VinDr-test false-abstention rate. The MEAN tracks the
target for any m (conformal is marginally valid), but the SPREAD shrinks with m -- so the
smallest m whose spread is tight around alpha is the practical calibration budget. Reports
mean / std / [p2.5,p97.5] / fraction-of-subsets-near-target, and a figure. Cached, no GPU.

    python -m scripts.s18_sample_efficiency --config configs/vindr.yaml
"""
from __future__ import annotations
import json
import os
import random

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.concept.valve import image_unexplained
from ovcbmr.ood.conformal import split_conformal_threshold, empirical_flag_rate


def _invocab_scores(emb_dir, proc, split, heads, vocab, ab_cols):
    import torch
    d = torch.load(os.path.join(emb_dir, f"{split}_img.pt"), weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    Pm = predict_matrix(d["emb"].numpy(), heads, vocab)
    out = []
    for i, iid in enumerate(d["ids"]):
        it = items.get(iid)
        if it is None or not any(int(it["labels_known"][j]) == 1 for j in ab_cols):
            continue
        out.append(image_unexplained(Pm[i].tolist(), vocab, "Normal")[1])
    return out


def main():
    cfg = parse_config_arg()
    import numpy as np
    ch = cfg.cross_hospital
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    ab_cols = [j for j, c in enumerate(vocab) if c != "Normal"]
    alpha = float(getattr(cfg.conformal, "alpha_flag", 0.05))
    seed = int(getattr(cfg, "seed", 0))
    K = 200

    cal = _invocab_scores(cfg.paths.embeddings, cfg.paths.processed, "calib", heads, vocab, ab_cols)
    test = _invocab_scores(cfg.paths.embeddings, cfg.paths.processed, "test", heads, vocab, ab_cols)
    n_cal = len(cal)
    print(f"[s18] VinDr in-vocab known-abnormal: calib={n_cal}, test={len(test)} | alpha={alpha} | K={K}")
    if n_cal < 25:
        print("[s18] calibration pool too small."); return

    grid = [m for m in (25, 50, 100, 200, 400, 800, 1500) if m < n_cal] + [n_cal]
    rng = random.Random(seed)
    rows = []
    print(f"\n{'m (calib)':>10}{'mean ffr':>10}{'std':>8}{'p2.5':>8}{'p97.5':>8}{'%<=a+.01':>10}")
    for m in grid:
        ffrs = [empirical_flag_rate(test, split_conformal_threshold(
                    rng.sample(cal, m) if m < n_cal else cal, alpha)) for _ in range(K)]
        a = np.array(ffrs)
        p2, p97, within = float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), float((a <= alpha + 0.01).mean())
        print(f"{m:>10}{a.mean():>10.3f}{a.std():>8.3f}{p2:>8.3f}{p97:>8.3f}{within:>10.2f}")
        rows.append({"m": m, "mean_ffr": float(a.mean()), "std": float(a.std()),
                     "p2.5": p2, "p97.5": p97, "frac_within_alpha": within})

    # smallest m whose 95% band is within [0, alpha+0.02] and mean within 0.01 of alpha
    ok = [r for r in rows if r["p97.5"] <= alpha + 0.02 and abs(r["mean_ffr"] - alpha) <= 0.01]
    rec = min((r["m"] for r in ok), default=None)
    print(f"\n[s18] smallest calibration budget with a tight guarantee (95% band <= a+0.02, "
          f"mean within .01 of a): {rec if rec is not None else 'not reached within pool'} in-vocab cases")

    os.makedirs(cfg.paths.results, exist_ok=True)
    with open(os.path.join(cfg.paths.results, "s18_sample_efficiency.json"), "w") as f:
        json.dump({"alpha": alpha, "K": K, "n_cal": n_cal, "n_test": len(test),
                   "recommended_m": rec, "rows": rows}, f, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ms = [r["m"] for r in rows]
        mean = [r["mean_ffr"] for r in rows]
        lo = [r["p2.5"] for r in rows]; hi = [r["p97.5"] for r in rows]
        plt.figure(figsize=(5.4, 4))
        plt.fill_between(ms, lo, hi, alpha=0.2, color="#27ae60", label="95% band over subsets")
        plt.plot(ms, mean, "o-", color="#1e7a34", label="mean false-abstention")
        plt.axhline(alpha, ls="--", color="red", lw=1, label=f"target α={alpha}")
        plt.xscale("log"); plt.xlabel("local calibration cases (in-vocab known-abnormal)")
        plt.ylabel("VinDr false-abstention rate"); plt.ylim(bottom=0)
        plt.title("Sample-efficiency of site-conditional calibration (NIH→VinDr)")
        plt.legend(fontsize=8); plt.tight_layout()
        figp = os.path.join(cfg.paths.results, "fig_sample_efficiency.png")
        plt.savefig(figp, dpi=150); print(f"[s18] figure -> {figp}")
    except Exception as e:                                        # noqa: BLE001
        print(f"[s18] plot skipped ({e}); curve is in s18_sample_efficiency.json")


if __name__ == "__main__":
    main()
