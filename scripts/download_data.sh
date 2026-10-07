#!/usr/bin/env bash
# Download the CXR-LT 2026 sources. This edition is multi-center: images come from
# NIH ChestX-ray14 (public) + PadChest (registration required); the CXR-LT 2026 release
# provides the label + split CSV (credentialed access).
#
# This script DOES NOT bypass any data-use agreement. PadChest and the CXR-LT release
# require you to register and accept their terms first; the commands below only run the
# downloads you are authorized for. Run from the repo root.
#
#   export KAGGLE_USERNAME=... KAGGLE_KEY=...        # for the NIH Kaggle mirror (option A)
#   export PHYSIONET_USER=...                        # for CXR-LT on PhysioNet
#   bash scripts/download_data.sh
set -euo pipefail

RAW="${RAW:-data/raw}"
mkdir -p "$RAW/nih" "$RAW/padchest" "$RAW/cxrlt2026/images"

echo "==============================================================================="
echo " 1) NIH ChestX-ray14  (public, ~42 GB, 112,120 images, labels: Data_Entry_2017)"
echo "==============================================================================="
# Option A — Kaggle CLI (needs ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY):
#   pip install kaggle
#   kaggle datasets download -d nih-chest-xrays/data -p "$RAW/nih" --unzip
#
# Option B — official NIH Clinical Center (public Box, 12 tarballs images_001..012.tar.gz):
#   Open https://nihcc.app.box.com/v/ChestXray-NIHCC and use the provided
#   `batch_download_zips.py` (direct Box links), plus Data_Entry_2017_v2020.csv.
#   Then: for f in "$RAW"/nih/images_*.tar.gz; do tar -xzf "$f" -C "$RAW/nih/images"; done
echo "   -> see options A/B in this script; images land under $RAW/nih/images"

echo "==============================================================================="
echo " 2) PadChest  (BIMCV, registration REQUIRED, ~1 TB full / resized PNG set smaller)"
echo "==============================================================================="
# Register and accept the agreement:  https://bimcv.cipf.es/bimcv-projects/padchest/
# You will receive links to the image ZIP parts + the labels CSV:
#   PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
# Download them into "$RAW/padchest" and unzip the image parts into "$RAW/padchest/images".
echo "   -> register, then place PadChest PNGs under $RAW/padchest/images + the labels CSV"

echo "==============================================================================="
echo " 3) CXR-LT 2026 labels + official splits  (PhysioNet, credentialed access)"
echo "==============================================================================="
# Accept the DUA on the CXR-LT 2026 PhysioNet project page, then download with wget.
# NOTE: confirm the exact project slug + version from the challenge / PhysioNet page and
# set CXRLT_SLUG below (the 2024 edition was 'cxr-lt-iccv-workshop-cvamd').
CXRLT_SLUG="${CXRLT_SLUG:-REPLACE_WITH_CXR_LT_2026_SLUG}"
CXRLT_VERSION="${CXRLT_VERSION:-1.0.0}"
if [ "$CXRLT_SLUG" != "REPLACE_WITH_CXR_LT_2026_SLUG" ] && [ -n "${PHYSIONET_USER:-}" ]; then
  wget -r -N -c -np --user "$PHYSIONET_USER" --ask-password \
       "https://physionet.org/files/${CXRLT_SLUG}/${CXRLT_VERSION}/" \
       -P "$RAW/cxrlt2026"
else
  echo "   -> set CXRLT_SLUG + PHYSIONET_USER, then re-run to fetch the label/split CSVs"
fi

echo "==============================================================================="
echo " 4) Assemble the pipeline manifest  ->  $RAW/cxrlt2026/labels.csv"
echo "==============================================================================="
# The pipeline expects one CSV with columns:
#   image_id, patient_id, projection, split, <30 known label cols>, <6 unseen label cols>
# Join the CXR-LT 2026 label/split file with the NIH + PadChest metadata and symlink the
# referenced frames under $RAW/cxrlt2026/images. Skeleton + target schema:
#   python -m scripts.build_manifest --raw "$RAW"
echo "   -> python -m scripts.build_manifest --raw \"$RAW\"   (adapt to the release format)"

echo "-------------------------------------------------------------------------------"
echo "After assembly, validate before running the pipeline:"
echo "   python scripts/s00_check_data.py --manifest $RAW/cxrlt2026/labels.csv --image-root $RAW/cxrlt2026/images"
echo "-------------------------------------------------------------------------------"
