"""Generate a tiny SYNTHETIC CXR-LT-style dataset for an end-to-end dry run of s01->s03.

No real data / no PHI: solid-gray PNGs + RANDOM labels using the default concept bank's column
names. Proves the whole data -> embeddings -> calibration path on your GPU before the real
download. NOT for any result (labels are random).

    python scripts/make_synthetic_cxrlt.py --n 128 --out data/raw/cxrlt2026_synth
    python -m scripts.s01_prepare_cxrlt      --config configs/synth.yaml
    python -m scripts.s02_extract_embeddings --config configs/synth.yaml
    python -m scripts.s03_calibrate_concepts --config configs/synth.yaml
"""
from __future__ import annotations
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ovcbmr.concept.bank import load_bank, active_vocabulary, concept_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="number of synthetic images")
    ap.add_argument("--out", default="data/raw/cxrlt2026_synth")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from PIL import Image
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    bank = load_bank("default")
    known = active_vocabulary(bank)
    unseen = concept_names(bank, "unseen")
    header = ["image_id", "patient_id", "projection", "split"] + known + unseen

    projs = ["PA", "AP", "PA", "AP", "L"]          # ~20% lateral -> dropped by frontal filter
    rows = []
    for i in range(args.n):
        iid = f"img_{i:04d}.png"
        shade = 100 + rng.randint(0, 60)           # slight per-image variation
        Image.new("RGB", (64, 64), (shade, shade, shade)).save(os.path.join(img_dir, iid))
        rec = {"image_id": iid, "patient_id": f"pat_{i // 3:04d}",   # ~3 images/patient
               "projection": projs[i % len(projs)], "split": ""}      # empty -> s01 generates splits
        for c in known:
            rec[c] = 1 if rng.random() < 0.30 else 0
        for c in unseen:
            rec[c] = 1 if rng.random() < 0.10 else 0
        rows.append(rec)

    with open(os.path.join(args.out, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"[synth] wrote {args.n} images + labels.csv under {args.out}")
    print(f"[synth] columns: id/patient/projection/split + {len(known)} known + {len(unseen)} unseen")
    print("[synth] next: s01 -> s02 -> s03 with --config configs/synth.yaml")


if __name__ == "__main__":
    main()
