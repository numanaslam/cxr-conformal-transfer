"""[E5-proxy] Multi-GROUP coverage on NIH observed groups (view / sex / age) — the honest
stand-in for the true multi-HOSPITAL E5 (which needs CXR-LT/PadChest).

For each grouping variable, LEAVE-ONE-GROUP-OUT: naive calibrates on the OTHER groups (i.e.
deploy on a group you didn't calibrate on, reproducing the E1 shift), group-conditional
calibrates on the held-out group. Shows group-conditional gives ~exact coverage for EVERY group
where naive misses under the shift, using trained-head valve scores.
NIH cache + Data_Entry_2017.csv, no GPU.

    python -m scripts.s11_multigroup --config configs/nih.yaml
"""
from __future__ import annotations
import csv
import json
import os

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.concept.valve import image_unexplained
from ovcbmr.ood.conformal import split_conformal_threshold, empirical_flag_rate


def _age_band(s):
    digits = "".join(ch for ch in str(s) if ch.isdigit())
    if not digits:
        return None
    a = int(digits)
    if a > 120:                                  # guard bad values
        return None
    return "<40" if a < 40 else "40-59" if a < 60 else "60-79" if a < 80 else "80+"


def main():
    cfg = parse_config_arg()
    import torch

    emb_dir, proc = cfg.paths.embeddings, cfg.paths.processed
    hp = os.path.join(proc, "concept_heads.json")
    if not os.path.exists(hp):
        print("[s11] concept_heads.json missing — run s09_train_heads first."); return
    heads = load_heads(hp)
    ca = torch.load(os.path.join(emb_dir, "calib_img.pt"), weights_only=False)
    te = torch.load(os.path.join(emb_dir, "test_img.pt"), weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    with open(os.path.join(proc, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]

    de_path = os.path.join(os.path.dirname(cfg.data.manifest), "Data_Entry_2017.csv")
    if not os.path.exists(de_path):
        print(f"[s11] {de_path} missing — needed for group variables."); return
    groups = {}
    with open(de_path, newline="") as f:
        for r in csv.DictReader(f):
            band = _age_band(r.get("Patient Age", ""))
            groups[r["Image Index"]] = {"view": str(r.get("View Position", "")).upper(),
                                        "sex": str(r.get("Patient Gender", "")).upper(),
                                        "age_band": band}

    ab_cols = [j for j, c in enumerate(vocab) if c != "Normal"]

    def collect(dp):
        emb = dp["emb"].numpy()
        Pm = predict_matrix(emb, heads, vocab)
        recs = []
        for i, iid in enumerate(dp["ids"]):
            it = items.get(iid)
            if it is None or iid not in groups:
                continue
            if not any(it["labels_known"][j] for j in ab_cols):     # known-abnormal only
                continue
            _, s = image_unexplained(Pm[i].tolist(), vocab, "Normal")
            recs.append((s, groups[iid]))
        return recs

    calib_recs, test_recs = collect(ca), collect(te)
    alpha = float(getattr(cfg.conformal, "alpha_flag", 0.05))
    print(f"[s11] known-abnormal: calib={len(calib_recs)} test={len(test_recs)} | alpha={alpha}")
    print("[s11] LEAVE-ONE-GROUP-OUT: naive calibrates on the OTHER groups (deploy on unseen "
          "group g); group-conditional calibrates on g. Reproduces the E1 shift per group.")

    summary = {}
    for gv in ("view", "sex", "age_band"):
        gset = sorted({g[gv] for _, g in test_recs if g.get(gv)})
        print(f"\n[s11] grouping = {gv}")
        print(f"{'group':<10} {'n_test':>7} {'naive(LOGO)':>12} {'groupcond':>10}  (target {alpha})")
        dev_n, dev_g, rows = [], [], []
        for g in gset:
            test_g = [s for s, gd in test_recs if gd.get(gv) == g]
            cal_g = [s for s, gd in calib_recs if gd.get(gv) == g]
            cal_notg = [s for s, gd in calib_recs if gd.get(gv) and gd.get(gv) != g]
            if len(test_g) < 30 or len(cal_g) < 30 or len(cal_notg) < 30:
                continue
            fn = empirical_flag_rate(test_g, split_conformal_threshold(cal_notg, alpha))  # deploy on unseen g
            fg = empirical_flag_rate(test_g, split_conformal_threshold(cal_g, alpha))     # per-group calib
            print(f"{g:<10} {len(test_g):>7} {fn:>12.3f} {fg:>10.3f}")
            dev_n.append(abs(fn - alpha)); dev_g.append(abs(fg - alpha))
            rows.append({"group": g, "n_test": len(test_g), "naive_logo_ffr": fn, "groupcond_ffr": fg})
        if dev_n:
            print(f"  max|ffr-alpha|: naive(LOGO)={max(dev_n):.3f}  group-conditional={max(dev_g):.3f}  "
                  f"-> group-conditional {'tighter for every group' if max(dev_g) < max(dev_n) else 'not uniformly better (small groups?)'}")
            summary[gv] = {"rows": rows, "max_dev_naive_logo": max(dev_n), "max_dev_groupcond": max(dev_g)}

    os.makedirs(cfg.paths.results, exist_ok=True)
    with open(os.path.join(cfg.paths.results, "s11_multigroup.json"), "w") as f:
        json.dump({"alpha": alpha, "groupings": summary,
                   "note": "NIH observed-group proxy for multi-hospital E5 (needs CXR-LT)"}, f, indent=2)
    print(f"\n[s11] wrote {os.path.join(cfg.paths.results, 's11_multigroup.json')}")
    print("[s11] NOTE: NIH observed-group proxy; true multi-HOSPITAL E5 needs CXR-LT/PadChest.")


if __name__ == "__main__":
    main()
