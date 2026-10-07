"""Build a VinDr-CXR manifest for the cross-hospital pilot (site=vindr), mapping VinDr classes
onto the NIH concept vocabulary so the NIH-trained heads apply directly. Aggregates the
box-level competition train.csv (image_id, class_name) to per-image labels.

    python scripts/build_manifest_vindr.py --csv data/raw/vindr/train.csv \
        --image-root data/raw/vindr/train --out data/raw/vindr/labels.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ovcbmr.concept.bank import load_bank, active_vocabulary, concept_names

# VinDr (Kaggle vinbigdata) class_name -> our concept name(s). Unmapped VinDr findings still
# make the image abnormal (Normal=0); the shift experiment only needs abnormal-vs-No-finding.
VINDR2OURS = {
    "Atelectasis": ["Atelectasis"], "Cardiomegaly": ["Cardiomegaly"],
    "Pleural effusion": ["Pleural Effusion"], "Infiltration": ["Infiltration"],
    "Nodule/Mass": ["Nodule", "Mass"], "Consolidation": ["Consolidation"],
    "Pleural thickening": ["Pleural Thickening"], "No finding": ["Normal"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="VinDr train.csv (columns: image_id, class_name)")
    ap.add_argument("--image-root", default="data/raw/vindr/train")
    ap.add_argument("--out", default="data/raw/vindr/labels.csv")
    ap.add_argument("--ext", default=".png")
    ap.add_argument("--bank", default="data/concepts/nih_concepts.yaml",
                    help="use the SAME bank as NIH so the NIH-trained heads apply")
    args = ap.parse_args()

    bank = load_bank(args.bank)
    known, unseen = active_vocabulary(bank), concept_names(bank, "unseen")
    allc = known + unseen

    per_img = defaultdict(set)
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            per_img[r["image_id"]].add(str(r.get("class_name", "")).strip())
    print(f"[vindr] {len(per_img)} unique images in {args.csv}")

    present = set(os.listdir(args.image_root)) if os.path.isdir(args.image_root) else None
    header = ["image_id", "patient_id", "projection", "split", "site"] + allc
    rows, skipped = [], 0
    for img, classes in per_img.items():
        fname = img + args.ext
        if present is not None and fname not in present:
            skipped += 1
            continue
        mapped = set()
        for cn in classes:
            mapped.update(VINDR2OURS.get(cn, []))
        only_nf = (classes == {"No finding"})
        rec = {"image_id": fname, "patient_id": img, "projection": "PA",  # VinDr frontal; view n/a
               "split": "", "site": "vindr"}
        for c in allc:
            rec[c] = 1 if (c in mapped or (c == "Normal" and only_nf)) else 0
        rows.append(rec)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    ab = sum(1 for r in rows if r["Normal"] == 0)
    print(f"[vindr] wrote {len(rows)} images ({skipped} skipped — no PNG under {args.image_root}) -> {args.out}")
    print(f"[vindr] abnormal (not No-finding): {ab} ({100*ab/max(1,len(rows)):.1f}%)")
    print(f"[vindr] mapped-known positives: {{ {', '.join(f'{c}:{sum(r[c] for r in rows)}' for c in known)} }}")
    print("[vindr] next: s01 -> s02 with configs/vindr.yaml, then s12_cross_hospital")


if __name__ == "__main__":
    main()
