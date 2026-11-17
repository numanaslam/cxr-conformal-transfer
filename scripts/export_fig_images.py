"""[figure helper] Export a few real test-split radiographs per site for Figure 1 (run on the GPU machine).

For each site, takes the first N test images (sorted by id, so the choice is reproducible) whose
label is Normal, falling back to any test image, converts them to 8-bit greyscale at native resolution
(or --size), and writes them to fig_assets/<site>_<k>.png plus fig_assets.zip.

    python scripts/export_fig_images.py --sites nih=configs/nih.yaml vindr=configs/vindr.yaml ^
        iuxray=configs/iuxray_nihtax.yaml --n 4
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="fig_assets")
    ap.add_argument("--size", type=int, default=0, help="shorter side in pixels; 0 keeps the native resolution")
    ap.add_argument("--iu-raw", default=None,
                    help="Indiana originals (images_normalized); used instead of the 256-pixel working copy")
    a = ap.parse_args()
    from PIL import Image
    from ovcbmr.config import load_config
    from ovcbmr.io.datasets import read_manifest
    from ovcbmr.io.splits import _truthy

    os.makedirs(a.out, exist_ok=True)
    written = []
    for spec in a.sites:
        name, cfg_path = spec.split("=", 1)
        cfg = load_config(cfg_path)
        with open(os.path.join(cfg.paths.processed, "index.json")) as f:
            items = [it for it in json.load(f) if it.get("split") == "test"]
        rows = {r[getattr(cfg.data, "id_col", "image_id")]: r for r in read_manifest(cfg.data.manifest)}
        normal = [it for it in items if _truthy(rows.get(it["id"], {}).get("Normal"))]
        pool = sorted(normal or items, key=lambda it: it["id"])
        k = 0
        for it in pool:
            if k == a.n:
                break
            path = it["path"]
            if name == "iuxray" and a.iu_raw:
                stem = os.path.splitext(it["id"])[0]
                for cand in (stem + ".dcm.png", "CXR" + stem + ".png", stem + ".png"):
                    if os.path.exists(os.path.join(a.iu_raw, cand)):
                        path = os.path.join(a.iu_raw, cand)
                        break
            try:
                im = Image.open(path).convert("L")
            except Exception as e:                       # noqa: BLE001
                print(f"[fig] skip {it['path']}: {e}")
                continue
            if a.size:
                w, h = im.size
                s = a.size / min(w, h)
                im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
            dst = os.path.join(a.out, f"{name}_{k}.png")
            im.save(dst)
            written.append(dst)
            print(f"[fig] {name}: {path} -> {dst} {im.size} ({'Normal' if normal else 'any label'})")
            k += 1
    zp = a.out.rstrip("/\\") + ".zip"
    with zipfile.ZipFile(zp, "w") as z:
        for p in written:
            z.write(p, os.path.basename(p))
    print(f"[fig] wrote {len(written)} images -> {zp}")


if __name__ == "__main__":
    main()
