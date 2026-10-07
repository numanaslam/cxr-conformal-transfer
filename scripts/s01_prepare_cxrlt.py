"""[build 1] Prepare CXR-LT 2026 into patient-level splits with a conformal calib split.

Reads the challenge label CSV, keeps frontal projections, assigns splits by PATIENT (no
leakage), reserves a `calib` split for conformal, and writes:
    processed/index.json   items with split + per-concept known labels + unseen flags
    processed/meta.json    active vocabulary, unseen (Task-2) columns, split summary

Run:
    python -m scripts.s01_prepare_cxrlt --config configs/default.yaml
"""
from __future__ import annotations
import json
import os
import sys

from ovcbmr.config import parse_config_arg
from ovcbmr.utils.seed import set_seed
from ovcbmr.io.datasets import read_manifest, filter_frontal, build_items, label_vector
from ovcbmr.io import splits as S
from ovcbmr.concept import bank as B


def main():
    cfg = parse_config_arg()
    set_seed(cfg.seed)
    d = cfg.data

    if not os.path.exists(d.manifest):
        print(f"[s01] manifest not found: {d.manifest}")
        print("      Download CXR-LT 2026 (SERVER_SETUP.md §6) and set data.manifest / data.image_root.")
        sys.exit(1)

    records = read_manifest(d.manifest)
    print(f"[s01] read {len(records)} rows from {d.manifest}")
    if getattr(d, "frontal_only", True):
        records = filter_frontal(records, getattr(d, "projection_col", "projection"),
                                 tuple(getattr(d, "projections_keep", ["PA", "AP"])))
        print(f"[s01] frontal-only -> {len(records)} rows")

    # concept columns present in the manifest
    bank = B.load_bank(getattr(cfg.concepts, "bank", "default"))
    fields = set(records[0].keys()) if records else set()
    vocab = [c for c in B.active_vocabulary(bank) if c in fields]
    unseen = [c for c in B.concept_names(bank, "unseen") if c in fields]
    missing = [c for c in B.active_vocabulary(bank) if c not in fields]
    if missing:
        print(f"[s01] WARNING: {len(missing)} known concepts absent from manifest columns "
              f"(adjust the bank to the official class list): {missing[:6]}...")
    print(f"[s01] active vocabulary |V|={len(vocab)} | unseen (Task-2)={len(unseen)}")

    # ---- splits: honor official split if present, else assign; always ensure a calib split
    split_col = getattr(d, "split_col", "split")
    has_official = split_col in fields and any(r.get(split_col) for r in records)
    if has_official:
        for r in records:
            r["split"] = (r.get(split_col) or "train").strip().lower()
        if getattr(cfg.splits, "calib_from_train", True):
            frac = getattr(cfg.splits, "fractions", None)
            cf = getattr(frac, "calib", 0.1) if frac is not None else 0.1
            moved = S.carve_calibration(records, calib_frac=cf / 0.7, seed=cfg.seed)
            print(f"[s01] honored official split; carved calib from train ({len(moved)} patients)")
    else:
        frac = getattr(cfg.splits, "fractions", None)
        fractions = ({"train": frac.train, "calib": frac.calib, "val": frac.val, "test": frac.test}
                     if frac is not None else
                     {"train": 0.70, "calib": 0.10, "val": 0.10, "test": 0.10})
        S.assign_patient_splits(records, fractions, seed=cfg.seed,
                                patient_key=getattr(d, "patient_col", "patient_id"))
        print(f"[s01] assigned patient-level splits {fractions}")

    S.verify_no_leakage(records, patient_key=getattr(d, "patient_col", "patient_id"))
    print("[s01] no-leakage check: PASS")

    # ---- items + labels
    items = build_items(records, cfg)
    for it, r in zip(items, records):
        it["labels_known"] = label_vector(r, vocab)
        it["labels_unseen"] = label_vector(r, unseen)

    os.makedirs(cfg.paths.processed, exist_ok=True)
    with open(os.path.join(cfg.paths.processed, "index.json"), "w") as f:
        json.dump(items, f)
    summary = S.split_summary(records, patient_key=getattr(d, "patient_col", "patient_id"))
    unanchored = S.unanchored_test_ids(records, unseen,
                                       id_key=getattr(d, "id_col", "image_id"))
    meta = {"vocabulary": vocab, "unseen": unseen, "splits": summary,
            "n_unanchored_test": len(unanchored)}
    with open(os.path.join(cfg.paths.processed, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[s01] splits: {summary}")
    print(f"[s01] unanchored (Task-2 positives in test): {len(unanchored)} images")
    print(f"[s01] wrote index.json + meta.json under {cfg.paths.processed}")


if __name__ == "__main__":
    main()
