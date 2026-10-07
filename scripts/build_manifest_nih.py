"""Build an NIH ChestX-ray14-only manifest (+ matching concept bank) to run the REAL
pipeline on real chest X-rays before PadChest/CXR-LT arrive.

Maps NIH's 14 findings (+ No Finding) onto our concept names, holds out a configurable set
as 'unseen' (Task-2) for a PRELIMINARY open-set experiment, and writes:
    data/raw/nih/labels.csv            manifest: image_id, patient_id, projection, split, labels
    data/concepts/nih_concepts.yaml    known/unseen bank matching the manifest columns

    python scripts/build_manifest_nih.py --csv data/raw/nih/Data_Entry_2017.csv \
        --image-root data/raw/nih/images
Then run s01..s04 with --config configs/nih.yaml.

Caveat: NIH open-set is approximate — held-out findings co-occur with kept ones on the same
image, so a 'novel' finding may be partly anchored by a correlated known concept. The real
open-world test is CXR-LT 2026 Task 2; this is a dev-loop + first-real-numbers run.
"""
from __future__ import annotations
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ovcbmr.concept.bank import DEFAULT_BANK

# NIH finding label -> our concept-bank name
NIH2BANK = {
    "Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Effusion": "Pleural Effusion",
    "Infiltration": "Infiltration", "Mass": "Mass", "Nodule": "Nodule", "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax", "Consolidation": "Consolidation", "Edema": "Pulmonary Edema",
    "Emphysema": "Emphysema", "Fibrosis": "Fibrosis", "Pleural_Thickening": "Pleural Thickening",
    "Hernia": "Hernia", "No Finding": "Normal",
}
DEFAULT_UNSEEN = ["Pneumothorax", "Hernia", "Pulmonary Edema", "Emphysema", "Fibrosis", "Pneumonia"]


def _display(name):
    for grp in ("known", "unseen"):
        for c in DEFAULT_BANK[grp]:
            if c["name"] == name:
                return c.get("display", name.lower())
    return name.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="NIH Data_Entry_2017.csv")
    ap.add_argument("--out", default="data/raw/nih/labels.csv")
    ap.add_argument("--bank-out", default="data/concepts/nih_concepts.yaml")
    ap.add_argument("--image-root", default="data/raw/nih/images")
    ap.add_argument("--unseen", default=",".join(DEFAULT_UNSEEN),
                    help="comma-separated concept names held out as Task-2 unseen")
    ap.add_argument("--max", type=int, default=0, help="cap images (random sample) for a fast loop; 0=all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_concepts = list(dict.fromkeys(NIH2BANK.values()))     # unique, order-preserving
    unseen = [s.strip() for s in args.unseen.split(",") if s.strip() and s.strip() in all_concepts]
    known = [c for c in all_concepts if c not in unseen]

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"[nih] read {len(rows)} rows from {args.csv}")

    if args.image_root and os.path.isdir(args.image_root):
        present = set(os.listdir(args.image_root))
        rows = [r for r in rows if r["Image Index"] in present]
        print(f"[nih] with image file present: {len(rows)}")
    if args.max and len(rows) > args.max:
        rows = random.Random(args.seed).sample(rows, args.max)
        print(f"[nih] random-sampled {len(rows)} for a fast loop")

    header = ["image_id", "patient_id", "projection", "split"] + known + unseen
    out_rows = []
    for r in rows:
        findings = [x.strip() for x in r.get("Finding Labels", "").split("|") if x.strip()]
        mapped = {NIH2BANK[f] for f in findings if f in NIH2BANK}
        rec = {"image_id": r["Image Index"], "patient_id": r.get("Patient ID", ""),
               "projection": r.get("View Position", ""), "split": ""}
        for c in known + unseen:
            rec[c] = 1 if c in mapped else 0
        out_rows.append(rec)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(out_rows)

    os.makedirs(os.path.dirname(args.bank_out) or ".", exist_ok=True)
    with open(args.bank_out, "w") as f:
        f.write("# Auto-generated NIH concept bank (build_manifest_nih.py). Change --unseen to re-split.\n")
        f.write("known:\n")
        for c in known:
            f.write(f"  - {{name: {c!r}, display: {_display(c)!r}}}\n")
        f.write("unseen:\n")
        for c in unseen:
            f.write(f"  - {{name: {c!r}, display: {_display(c)!r}}}\n")

    pos = {c: sum(rr[c] for rr in out_rows) for c in known + unseen}
    projs = {}
    for rr in out_rows:
        projs[rr["projection"]] = projs.get(rr["projection"], 0) + 1
    print(f"[nih] wrote {len(out_rows)} rows -> {args.out}")
    print(f"[nih] known V ({len(known)}): {known}")
    print(f"[nih] unseen ({len(unseen)}): {unseen}")
    print(f"[nih] projections: {projs}")
    print(f"[nih] positives/concept: {pos}")
    print(f"[nih] wrote bank -> {args.bank_out}")
    print("[nih] next: s01 -> s02 -> s03 -> s04 with --config configs/nih.yaml")


if __name__ == "__main__":
    main()
