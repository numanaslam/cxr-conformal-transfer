"""[build 0] Pre-flight data check — run right after download, BEFORE the env is even built.

Stdlib only (csv/os) so it works before `pip install`. Validates that the manifest has the
columns the pipeline needs, reports the projection distribution (frontal filter), checks how
many concept columns are present, and confirms a sample of image files exist.

    python scripts/s00_check_data.py \
        --manifest data/raw/cxrlt2026/labels.csv \
        --image-root data/raw/cxrlt2026/images
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ovcbmr.concept.bank import load_bank, active_vocabulary, concept_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/raw/cxrlt2026/labels.csv")
    ap.add_argument("--image-root", default="data/raw/cxrlt2026/images")
    ap.add_argument("--id-col", default="image_id")
    ap.add_argument("--patient-col", default="patient_id")
    ap.add_argument("--projection-col", default="projection")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--sample", type=int, default=50, help="how many image files to probe")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        print(f"FAIL: manifest not found: {args.manifest}")
        print("      run scripts/download_data.sh + scripts/build_manifest.py first.")
        sys.exit(1)

    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    print(f"[s00] rows={len(rows)}  columns={len(cols)}")

    warn = 0
    required = [args.id_col, args.patient_col, args.projection_col]
    missing_req = [c for c in required if c not in cols]
    if missing_req:
        print(f"FAIL: required columns missing: {missing_req}")
        sys.exit(1)
    print(f"[s00] required columns present: {required}")
    if args.split_col not in cols:
        print(f"[s00] WARN: no '{args.split_col}' column — s01 will generate patient-level splits")
        warn += 1

    # projection distribution (frontal filter sanity)
    proj = Counter(str(r.get(args.projection_col, "")).upper() for r in rows)
    frontal = proj.get("PA", 0) + proj.get("AP", 0)
    print(f"[s00] projections: {dict(proj)}")
    print(f"[s00] frontal (PA/AP) = {frontal}/{len(rows)} ({100*frontal/max(1,len(rows)):.1f}%)")

    # concept columns present
    bank = load_bank("default")
    known = active_vocabulary(bank)
    unseen = concept_names(bank, "unseen")
    k_present = [c for c in known if c in cols]
    u_present = [c for c in unseen if c in cols]
    print(f"[s00] known concept columns:  {len(k_present)}/{len(known)} present")
    print(f"[s00] unseen concept columns: {len(u_present)}/{len(unseen)} present")
    if len(k_present) < len(known) or len(u_present) < len(unseen):
        print("      NOTE: edit data/concepts/cxrlt2026_concepts.yaml to the official CXR-LT class names")
        warn += 1

    # patient count
    n_patients = len({r.get(args.patient_col) for r in rows})
    print(f"[s00] unique patients: {n_patients}")

    # probe a sample of image files
    root = args.image_root
    probe = rows[: args.sample]
    found = 0
    for r in probe:
        iid = r.get(args.id_col, "")
        p = iid if os.path.isabs(iid) else os.path.join(root, iid)
        if os.path.exists(p):
            found += 1
    print(f"[s00] image files found: {found}/{len(probe)} sampled under {root}")
    if probe and found == 0:
        print("      WARN: no sampled images resolved — check --image-root / id-to-path mapping")
        warn += 1

    print(f"[s00] {'PASS' if warn == 0 else f'PASS with {warn} warning(s)'} — "
          "ready for s01_prepare_cxrlt" if warn == 0 else
          f"[s00] review {warn} warning(s) above before s01_prepare_cxrlt")


if __name__ == "__main__":
    main()
