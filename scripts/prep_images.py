"""Pre-resize source images ONCE so s02 stops decoding full-res PNGs on the fly.

NIH frames are 1024x1024; decoding+resizing them per epoch is what starves the GPU (the
ViT-B/16 forward is fast). Resizing to 256 on disk makes s02 GPU-bound instead of decode-bound,
and speeds every future re-encode (ablations, other backbones). Aspect ratio is preserved
(shorter side -> --size), so BiomedCLIP's Resize+CenterCrop is unchanged; 256 leaves crop margin.

    python scripts/prep_images.py --src data/raw/nih/images --dst data/raw/nih/images_256 --workers 12
Then set  data.image_root: data/raw/nih/images_256  in the config and re-run s02.
"""
from __future__ import annotations
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def _resize_one(src, dst, size):
    from PIL import Image
    try:
        im = Image.open(src).convert("RGB")
        w, h = im.size
        scale = size / min(w, h)
        if scale < 1.0:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
        im.save(dst)
        return True
    except Exception as e:                       # noqa: BLE001 — report and continue
        print(f"  FAIL {os.path.basename(src)}: {e}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ext", default=".png")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = [f for f in os.listdir(args.src) if f.lower().endswith(args.ext.lower())]
    todo = [f for f in files if not os.path.exists(os.path.join(args.dst, f))]
    print(f"[prep] {len(files)} images | {len(files)-len(todo)} already resized | doing {len(todo)} "
          f"-> {args.dst} (size={args.size}, workers={args.workers})")

    t0, done, ok = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_resize_one, os.path.join(args.src, f), os.path.join(args.dst, f),
                          args.size): f for f in todo}
        for fut in as_completed(futs):
            ok += 1 if fut.result() else 0
            done += 1
            if done % 5000 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"[prep] {done}/{len(todo)}  ({rate:.0f} img/s)", flush=True)
    dt = max(time.time() - t0, 1e-9)
    print(f"[prep] done: {ok}/{len(todo)} resized in {dt:.0f}s ({len(todo)/dt:.0f} img/s)")
    print(f"[prep] now set  data.image_root: {args.dst}  and re-run s02")


if __name__ == "__main__":
    main()
