"""[E1 - headline] Coverage-under-shift demo, on cached embeddings (no GPU, no new data).

NIH has a real internal covariate shift: PA (standing outpatients) vs AP (portable/ICU, sicker).
We calibrate the G1 valve threshold on PA known-abnormal images, then measure the empirical
false-flag rate on held-out PA (in-distribution) and AP (shifted), for NAIVE split-conformal vs
SHIFT-ROBUST weighted conformal (domain-classifier likelihood ratios). Sweeps alpha and saves the
coverage-vs-alpha curve + figure.

Story we're testing: naive conformal MIScovers under shift (its q_hat no longer controls the
false-flag rate on AP); shift-robust weighting tracks the target alpha.

    python -m scripts.s07_shift_demo --config configs/nih.yaml
"""
from __future__ import annotations
import json
import os

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.concept.valve import image_unexplained
from ovcbmr.ood.conformal import (split_conformal_threshold, weighted_split_conformal_threshold,
                                   estimate_shift_weights, empirical_flag_rate,
                                   effective_sample_size)


def main():
    cfg = parse_config_arg()
    import numpy as np
    import torch

    emb_dir, proc = cfg.paths.embeddings, cfg.paths.processed
    hp = os.path.join(proc, "concept_heads.json")
    if not os.path.exists(hp):
        print("[s07] concept_heads.json missing — run s09_train_heads first."); return
    heads = load_heads(hp)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    with open(os.path.join(proc, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]

    # gather all cached embeddings across splits into one array + id->row
    embs, ids = [], []
    for split in ("train", "calib", "val", "test"):
        p = os.path.join(emb_dir, f"{split}_img.pt")
        if os.path.exists(p):
            d = torch.load(p, weights_only=False)
            embs.append(d["emb"].numpy())
            ids.extend(d["ids"])
    emb = np.concatenate(embs, axis=0)
    idpos = {iid: i for i, iid in enumerate(ids)}

    # per-image unexplained statistic s(x) from the trained-head predictor (ADR-0005) — the
    # default predictor; the earlier zero-shot version inflated the naive shift-failure.
    Pmat = predict_matrix(emb, heads, vocab)                    # [M, K] calibrated present-probs
    ab_cols = [j for j, c in enumerate(vocab) if c != "Normal"]

    def sx(row):
        return image_unexplained(Pmat[row].tolist(), vocab, "Normal")[1]

    # bucket KNOWN-ABNORMAL images by (split, projection)
    buckets = {}
    for iid, it in items.items():
        if iid not in idpos:
            continue
        if not any(it["labels_known"][j] for j in ab_cols):    # known-abnormal only
            continue
        proj = str(it.get("projection", "")).upper()
        if proj in ("PA", "AP"):
            buckets.setdefault((it.get("split"), proj), []).append(iid)

    cal_ids = buckets.get(("train", "PA"), [])                 # calibrate G1 on PA
    ref_ids = buckets.get(("train", "AP"), [])                 # AP reference for shift weights
    pa_eval_ids = buckets.get(("test", "PA"), [])              # in-distribution eval
    ap_eval_ids = buckets.get(("test", "AP"), [])              # shifted eval
    for nm, L in [("cal PA(train)", cal_ids), ("ref AP(train)", ref_ids),
                  ("eval PA(test)", pa_eval_ids), ("eval AP(test)", ap_eval_ids)]:
        print(f"[s07] {nm:<14}: {len(L)} known-abnormal images")
    if min(len(cal_ids), len(ref_ids), len(ap_eval_ids), len(pa_eval_ids)) < 50:
        print("[s07] not enough PA/AP known-abnormal images for the shift demo.")
        return

    def scores_of(idlist):
        return [sx(idpos[i]) for i in idlist]

    def embs_of(idlist):
        return emb[[idpos[i] for i in idlist]]

    cal_s = scores_of(cal_ids)
    pa_eval_s = scores_of(pa_eval_ids)
    ap_eval_s = scores_of(ap_eval_ids)
    group_cal_s = scores_of(ref_ids)          # AP calibration set for group-conditional (Mondrian)

    # shift weights for the calibration points: P_AP / P_PA
    w = estimate_shift_weights(embs_of(cal_ids), embs_of(ref_ids)).tolist()
    ess = effective_sample_size(w)
    print(f"[s07] shift weights: min={min(w):.3f} med={sorted(w)[len(w)//2]:.3f} max={max(w):.1f} "
          f"| ESS={ess:.0f}/{len(w)} ({100*ess/max(1,len(w)):.1f}%) "
          f"{'[OK]' if ess/max(1,len(w)) > 0.05 else '[LOW -> weighted quantile unreliable]'}")

    alphas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    rows = []
    for a in alphas:
        qn = split_conformal_threshold(cal_s, a)
        qr = weighted_split_conformal_threshold(cal_s, w, a)
        qg = split_conformal_threshold(group_cal_s, a)         # Mondrian: calibrate on the AP group
        rows.append({
            "alpha": a,
            "naive_ffr_pa": empirical_flag_rate(pa_eval_s, qn),
            "robust_ffr_pa": empirical_flag_rate(pa_eval_s, qr),
            "naive_ffr_ap": empirical_flag_rate(ap_eval_s, qn),
            "robust_ffr_ap": empirical_flag_rate(ap_eval_s, qr),
            "group_ffr_ap": empirical_flag_rate(ap_eval_s, qg),
        })

    r5 = next(r for r in rows if abs(r["alpha"] - 0.05) < 1e-9)
    print(f"\n[s07] alpha=0.05  in-dist PA : naive_ffr={r5['naive_ffr_pa']:.3f}  robust_ffr={r5['robust_ffr_pa']:.3f}")
    print(f"[s07] alpha=0.05  SHIFT   AP : naive_ffr={r5['naive_ffr_ap']:.3f}  robust_ffr={r5['robust_ffr_ap']:.3f}   (target 0.050)")
    print(f"[s07] alpha=0.05  group-conditional AP (per-site calibration): ffr={r5['group_ffr_ap']:.3f}   (target 0.050)")
    dev_n = abs(r5["naive_ffr_ap"] - 0.05)
    dev_r = abs(r5["robust_ffr_ap"] - 0.05)
    verdict = ("shift-robust is closer to target on AP (weighting helps)"
               if dev_r < dev_n - 1e-6 else
               "naive already near target on AP (PA->AP shift is mild here; CXR-LT cross-hospital is stronger)")
    print(f"[s07] |ffr - alpha| on AP: naive={dev_n:.3f} vs robust={dev_r:.3f}  ->  {verdict}")
    track = sum(abs(r["robust_ffr_ap"] - r["alpha"]) for r in rows) / len(rows)
    trackg = sum(abs(r["group_ffr_ap"] - r["alpha"]) for r in rows) / len(rows)
    print(f"[s07] mean|ffr-alpha| over sweep (AP): naive={sum(abs(r['naive_ffr_ap']-r['alpha']) for r in rows)/len(rows):.3f}  "
          f"robust={track:.3f}  group-conditional={trackg:.3f}")
    print(f"[s07] group-conditional {'gives ~EXACT per-site coverage' if trackg < 0.03 else 'off target'}; "
          f"importance-weighting {'restores coverage' if track < 0.03 else 'only partially corrects (residual = conditional shift)'}")

    os.makedirs(cfg.paths.results, exist_ok=True)
    out = {"curve": rows, "n": {"cal_pa": len(cal_ids), "ref_ap": len(ref_ids),
                                "eval_pa": len(pa_eval_ids), "eval_ap": len(ap_eval_ids)}}
    with open(os.path.join(cfg.paths.results, "s07_shift.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[s07] wrote {os.path.join(cfg.paths.results, 's07_shift.json')}")

    # headline figure: coverage-vs-alpha on the AP shift
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["alpha"] for r in rows]
        plt.figure(figsize=(5, 4))
        plt.plot(xs, xs, "k--", label="target (y=x)")
        plt.plot(xs, [r["naive_ffr_ap"] for r in rows], "o-", label="naive pooled (AP shift)")
        plt.plot(xs, [r["robust_ffr_ap"] for r in rows], "s-", label="importance-weighted (AP)")
        plt.plot(xs, [r["group_ffr_ap"] for r in rows], "^-", label="group-conditional (AP)")
        plt.xlabel("target alpha"); plt.ylabel("empirical false-flag rate (AP)")
        plt.title("G1 coverage under PA->AP shift"); plt.legend(); plt.tight_layout()
        figp = os.path.join(cfg.paths.results, "shift_coverage.png")
        plt.savefig(figp, dpi=150)
        print(f"[s07] figure -> {figp}")
    except Exception as e:                                     # noqa: BLE001
        print(f"[s07] plot skipped ({e}); curve is in s07_shift.json")


if __name__ == "__main__":
    main()
