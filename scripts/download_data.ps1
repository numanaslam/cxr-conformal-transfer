<#
Download the CXR-LT 2026 sources on Windows (PowerShell). Multi-center: images from
NIH ChestX-ray14 (public) + PadChest (registration); labels/splits from CXR-LT 2026 (PhysioNet).

This does NOT bypass any data-use agreement — PadChest and CXR-LT require you to register
and accept terms first. Run from the repo root.

  # optional env for the automated bits:
  $env:CXRLT_SLUG   = "REPLACE_WITH_CXR_LT_2026_SLUG"   # confirm on the challenge/PhysioNet page
  $env:CXRLT_VERSION= "1.0.0"
  $env:PHYSIONET_USER = "your_physionet_username"
  powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1
#>
$ErrorActionPreference = "Stop"
$RAW = if ($env:RAW) { $env:RAW } else { "data\raw" }
New-Item -ItemType Directory -Force -Path "$RAW\nih","$RAW\padchest","$RAW\cxrlt2026\images" | Out-Null

Write-Host "=== 1) NIH ChestX-ray14 (public, ~42 GB) ==="
# Option A - Kaggle CLI (put kaggle.json in %USERPROFILE%\.kaggle):
#   pip install kaggle
#   kaggle datasets download -d nih-chest-xrays/data -p "$RAW\nih" --unzip
# Option B - official NIH Box (12 tarballs images_001..012.tar.gz + Data_Entry_2017_v2020.csv):
#   https://nihcc.app.box.com/v/ChestXray-NIHCC   (use their batch_download_zips.py)
# Option C (recommended) - parallel, resumable downloader for the 12 tarballs at once:
#   python scripts\pdownload.py --list scripts\nih_urls.txt --out "$RAW\nih" --workers 4 --extract
Write-Host "   -> Option C (parallel): python scripts\pdownload.py --list scripts\nih_urls.txt --out $RAW\nih --workers 4 --extract"

Write-Host "=== 2) PadChest (BIMCV, registration REQUIRED) ==="
# Register + accept: https://bimcv.cipf.es/bimcv-projects/padchest/
# Download the image ZIP parts + PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
# into $RAW\padchest , then Expand-Archive the parts into $RAW\padchest\images.
Write-Host "   -> register, then place PadChest PNGs under $RAW\padchest\images + the labels CSV"

Write-Host "=== 3) CXR-LT 2026 labels + splits (PhysioNet, credentialed) ==="
# Get the EXACT PhysioNet project link + data-access steps from the challenge site:
#   https://cxr-lt.github.io/CXR-LT-2026/     (also CodaLab: competitions/18601)
# CXR-LT is the PhysioNet 'cxr-lt-iccv-workshop-cvamd' family (confirm the 2026 version, ~2.0.0).
# Register on PhysioNet, accept the DUA, then recursive-download (install wget once on Windows):
#   conda install -y -c conda-forge wget
# The label files give: dicom_id + 30 seen + 6 unseen class columns (+ split, patient/site metadata).
# IMAGES: a curated PadChest subset + NIH — PhysioNet may host the curated images directly; if not,
# get PadChest from BIMCV (step 2) and NIH from step 1, then map by dicom_id.
$slug = if ($env:CXRLT_SLUG) { $env:CXRLT_SLUG } else { "cxr-lt-iccv-workshop-cvamd" }
$ver  = if ($env:CXRLT_VERSION) { $env:CXRLT_VERSION } else { "2.0.0" }
if ($slug -ne "REPLACE_WITH_CXR_LT_2026_SLUG" -and $env:PHYSIONET_USER) {
  if (Get-Command wget -ErrorAction SilentlyContinue) {
    wget -r -N -c -np --user $env:PHYSIONET_USER --ask-password `
         "https://physionet.org/files/$slug/$ver/" -P "$RAW\cxrlt2026"
  } else {
    Write-Host "   wget not found. Run:  conda install -y -c conda-forge wget   then re-run."
  }
} else {
  Write-Host "   -> set `$env:CXRLT_SLUG + `$env:PHYSIONET_USER, then re-run to fetch the label/split CSVs"
}

Write-Host "=== 4) Assemble the pipeline manifest -> $RAW\cxrlt2026\labels.csv ==="
# Expected columns: image_id, patient_id, projection, split, <30 known>, <6 unseen>
Write-Host "   -> python -m scripts.build_manifest --raw `"$RAW`"   (adapt to the release format)"

Write-Host "-------------------------------------------------------------------------------"
Write-Host "Validate before the pipeline:"
Write-Host "   python scripts\s00_check_data.py --manifest $RAW\cxrlt2026\labels.csv --image-root $RAW\cxrlt2026\images"
