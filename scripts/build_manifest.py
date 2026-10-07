"""Assemble the unified pipeline manifest from the CXR-LT 2026 release + source metadata.

TARGET SCHEMA  ->  data/raw/cxrlt2026/labels.csv
    image_id, patient_id, projection, split, <30 known label cols>, <6 unseen label cols>

The known/unseen column names must match data/concepts/cxrlt2026_concepts.yaml
(check with scripts/s00_check_data.py). Source columns to join:
    NIH ChestX-ray14  : Data_Entry_2017.csv  -> 'Image Index','Patient ID','View Position'
    PadChest          : PADCHEST_..._labels.csv -> 'ImageID','PatientID','Projection','Labels'
    CXR-LT 2026       : release label/split CSV -> per-image 30+6 class flags + official split

This is a SKELETON: the exact CXR-LT 2026 file format ships with the release. Fill the two
TODO blocks (source readers + the label-name mapping) and this writes the manifest.

    python -m scripts.build_manifest --raw data/raw
"""
from __future__ import annotations
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ovcbmr.concept.bank import load_bank, active_vocabulary, concept_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.raw, "cxrlt2026", "labels.csv")

    bank = load_bank("default")
    known = active_vocabulary(bank)
    unseen = concept_names(bank, "unseen")
    header = ["image_id", "patient_id", "projection", "split"] + known + unseen

    # TODO(1): read the CXR-LT 2026 release label/split CSV (per-image 30+6 flags + split).
    # TODO(2): join NIH Data_Entry_2017 + PadChest labels for patient_id + projection, and
    #          map their finding names onto `known`/`unseen` (via the release's label map).
    rows = []  # list[dict] with keys == header

    if not rows:
        print("[build_manifest] SKELETON — no rows assembled yet.")
        print(f"[build_manifest] target: {out}")
        print(f"[build_manifest] header ({len(header)} cols): "
              f"image_id, patient_id, projection, split, +{len(known)} known, +{len(unseen)} unseen")
        print("[build_manifest] fill TODO(1)/TODO(2) with the CXR-LT 2026 release format, then re-run.")
        return

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"[build_manifest] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
