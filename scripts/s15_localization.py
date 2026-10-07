"""[localization] Quantitative interpretability: does concept-conditioned occlusion
saliency land on the radiologist's box? Pointing game on VinDr's VinBigData boxes.

For each VinDr TEST image a radiologist annotated with one of our 8 concepts, we
occlusion-map THAT concept (re-encode a G*G grid; saliency = drop in the concept's
calibrated probability) and check whether the saliency peak cell falls inside the box.
Report hit-rate vs the random-point null (mean box-area fraction) with a per-image
bootstrap CI, overall and per concept. GPU (re-encodes each kept image ~G*G times).

Boxes are normalized by ORIGINAL image dimensions, so you must supply them: either
width/height columns in --boxes, or a --dims CSV (image_id, width|dim1, height|dim0).
The awsaf49 512-px images are square-resized, so original-dim normalization maps boxes
correctly onto the 224 grid. A warning fires if normalized coords exceed 1 (wrong dims).

    python -m scripts.s15_localization --config configs/vindr.yaml \
        --boxes data/raw/vindr/train.csv --dims data/raw/vindr/train_meta.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from collections import defaultdict

from ovcbmr.config import load_config

# VinDr (VinBigData) class_name -> our concept(s); mirrors build_manifest_vindr.
VINDR2OURS = {
    "Atelectasis": ["Atelectasis"], "Cardiomegaly": ["Cardiomegaly"],
    "Pleural effusion": ["Pleural Effusion"], "Infiltration": ["Infiltration"],
    "Nodule/Mass": ["Nodule", "Mass"], "Consolidation": ["Consolidation"],
    "Pleural thickening": ["Pleural Thickening"],
}
_W_KEYS = ("width", "original_width", "dim1", "w", "cols", "Width")
_H_KEYS = ("height", "original_height", "dim0", "h", "rows", "Height")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _pick(row, keys):
    for k in keys:
        v = _num(row.get(k)) if k in row else None
        if v:
            return v
    return None


def _base(name):
    return os.path.splitext(str(name))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vindr.yaml")
    ap.add_argument("--boxes", default="data/raw/vindr/train.csv",
                    help="VinBigData box CSV: image_id,class_name,x_min,y_min,x_max,y_max")
    ap.add_argument("--dims", default="",
                    help="optional CSV image_id,width,height (original dims); else uses "
                         "width/height columns in --boxes")
    ap.add_argument("--grid", type=int, default=14)              # 224/14 = 16px cells
    ap.add_argument("--max-images", type=int, default=300)
    ap.add_argument("--nboot", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)

    import numpy as np
    import torch
    from PIL import Image
    from ovcbmr.concept.heads import load_heads, predict_matrix
    from ovcbmr.eval.metrics import bootstrap_ci
    from ovcbmr.models.encoder import load_biomedclip

    ch = cfg.cross_hospital
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    n_boot = args.nboot or int(getattr(getattr(cfg, "eval", object()), "bootstrap", 1000))
    seed = int(getattr(cfg, "seed", 0))
    G = args.grid

    # --- original dims per image (for box normalization) ---
    dims = {}
    if args.dims and os.path.exists(args.dims):
        with open(args.dims, newline="") as f:
            for r in csv.DictReader(f):
                W, H = _pick(r, _W_KEYS), _pick(r, _H_KEYS)
                if W and H:
                    dims[_base(r.get("image_id", ""))] = (W, H)

    # --- boxes per image, mapped to our concepts ---
    boxes = defaultdict(lambda: defaultdict(list))              # base_id -> concept -> [(x0,y0,x1,y1)]
    with open(args.boxes, newline="") as f:
        for r in csv.DictReader(f):
            targets = VINDR2OURS.get(str(r.get("class_name", "")).strip())
            if not targets:
                continue
            x0, y0, x1, y1 = (_num(r.get("x_min")), _num(r.get("y_min")),
                              _num(r.get("x_max")), _num(r.get("y_max")))
            if None in (x0, y0, x1, y1):
                continue
            bid = _base(r.get("image_id", ""))
            if bid not in dims:                                # dims from box CSV if present
                W, H = _pick(r, _W_KEYS), _pick(r, _H_KEYS)
                if W and H:
                    dims[bid] = (W, H)
            for c in targets:
                boxes[bid][c].append((x0, y0, x1, y1))

    # --- restrict to VinDr TEST images that have boxes + dims ---
    d = torch.load(os.path.join(cfg.paths.embeddings, "test_img.pt"), weights_only=False)
    test_ids = sorted(iid for iid in d["ids"] if _base(iid) in boxes and _base(iid) in dims)
    n_avail = len(test_ids)
    if args.max_images and n_avail > args.max_images:
        test_ids = test_ids[:args.max_images]
    print(f"[s15] VinDr test images with usable box+dims: {n_avail} "
          f"(using {len(test_ids)}; grid {G}x{G}, boot {n_boot})")
    if not test_ids:
        print("[s15] none. Check --boxes columns (image_id/class_name/x_min..y_max) and --dims "
              "(image_id + width|dim1, height|dim0); ids must match VinDr test embeddings.")
        return

    device = cfg.compute.device
    model, preprocess, _ = load_biomedclip(device, getattr(cfg.model, "hf_id"))

    def encode(batch):
        with torch.inference_mode():
            z = torch.nn.functional.normalize(model.encode_image(batch.to(device)), dim=-1)
        return z.float().cpu().numpy()

    nidx = vocab.index("Normal")

    def saliency(base_t, cidx):
        """(concept-conditioned, concept-agnostic) G x G occlusion maps from the SAME forward
        passes: drop in p(concept) and drop in abnormality A = 1 - p(Normal). The agnostic map
        is a fair baseline -- same method, but not concept-specific -- so concept-conditioned
        minus agnostic isolates what the concept SPECIFICITY adds."""
        H = base_t.shape[1]; cell = H // G
        variants = [base_t.clone()]
        for gy in range(G):
            for gx in range(G):
                v = base_t.clone()
                v[:, gy*cell:(gy+1)*cell, gx*cell:(gx+1)*cell] = 0.0
                variants.append(v)
        P = predict_matrix(encode(torch.stack(variants)), heads, vocab)
        pc, pa = P[:, cidx], 1.0 - P[:, nidx]
        sal_c = np.clip(pc[0] - np.asarray(pc[1:]).reshape(G, G), 0.0, None)
        sal_a = np.clip(pa[0] - np.asarray(pa[1:]).reshape(G, G), 0.0, None)
        return sal_c, sal_a

    def covered_cells(bxs, W, H):
        cov = set()
        for x0, y0, x1, y1 in bxs:
            gx0 = int(np.clip(min(x0, x1) / W * G, 0, G - 1))
            gx1 = int(np.clip(max(x0, x1) / W * G - 1e-9, 0, G - 1))
            gy0 = int(np.clip(min(y0, y1) / H * G, 0, G - 1))
            gy1 = int(np.clip(max(y0, y1) / H * G - 1e-9, 0, G - 1))
            for gy in range(gy0, gy1 + 1):
                for gx in range(gx0, gx1 + 1):
                    cov.add((gy, gx))
        return cov

    img_root = cfg.data.image_root
    items, maxnorm = [], 0.0            # (hit_concept, hit_agnostic, null_frac, concept, image_id)
    for iid in test_ids:
        bid = _base(iid); W, H = dims[bid]
        try:
            base_t = preprocess(Image.open(os.path.join(img_root, iid)).convert("RGB"))
        except Exception:                                      # noqa: BLE001
            continue
        for c, bxs in boxes[bid].items():
            if c not in vocab:
                continue
            cov = covered_cells(bxs, W, H)
            if not cov:
                continue
            maxnorm = max([maxnorm] + [x1 / W for _, _, x1, _ in bxs] + [y1 / H for _, _, _, y1 in bxs])
            sal_c, sal_a = saliency(base_t, vocab.index(c))
            gyc, gxc = np.unravel_index(int(np.argmax(sal_c)), sal_c.shape)
            gya, gxa = np.unravel_index(int(np.argmax(sal_a)), sal_a.shape)
            items.append((1.0 if (int(gyc), int(gxc)) in cov else 0.0,
                          1.0 if (int(gya), int(gxa)) in cov else 0.0,
                          len(cov) / (G * G), c, bid))

    if not items:
        print("[s15] no (image, concept) pairs scored."); return
    if maxnorm > 1.5:
        print(f"[s15] WARNING: normalized box coords reach {maxnorm:.2f} (>1) — --dims are probably "
              "not the ORIGINAL image dimensions; hit-rate is unreliable until fixed.")

    imgs = [t[4] for t in items]
    null_rate = sum(t[2] for t in items) / len(items)
    hit_c, loc, hic = bootstrap_ci([t[0] for t in items], groups=imgs, n_boot=n_boot, seed=seed)
    hit_a, loa, hia = bootstrap_ci([t[1] for t in items], groups=imgs, n_boot=n_boot, seed=seed)
    by_c = defaultdict(list)
    for t in items:
        by_c[t[3]].append(t)

    print(f"\n[s15] pointing game over {len(items)} (image,concept) pairs, {len(set(imgs))} images")
    print(f"  concept-conditioned : {hit_c:.3f} [{loc:.3f},{hic:.3f}]   (occlusion on p(concept))")
    print(f"  concept-agnostic    : {hit_a:.3f} [{loa:.3f},{hia:.3f}]   (occlusion on abnormality A=1-p(Normal))")
    print(f"  random-point null   : {null_rate:.3f}")
    print(f"  specificity lift (conditioned - agnostic): {hit_c - hit_a:+.3f}")
    print(f"\n    {'concept':<18}{'hit':>7}{'95% CI':>17}{'null':>7}{'agnos':>7}{'n':>6}  sig")
    per = {}
    for c in sorted(by_c, key=lambda k: -(sum(t[0] for t in by_c[k]) / len(by_c[k]))):
        v = by_c[c]
        hc = sum(t[0] for t in v) / len(v); ha = sum(t[1] for t in v) / len(v); nl = sum(t[2] for t in v) / len(v)
        _, clo, chi = bootstrap_ci([t[0] for t in v], groups=[t[4] for t in v], n_boot=n_boot, seed=seed)
        sig = "*" if clo > nl else " "                          # 95% CI lower bound above its own null
        print(f"    {c:<18}{hc:>7.3f}  [{clo:.3f},{chi:.3f}]{nl:>7.3f}{ha:>7.3f}{len(v):>6}  {sig}")
        per[c] = {"hit_rate": hc, "ci": [clo, chi], "null": nl, "agnostic_hit": ha,
                  "n": len(v), "clears_null": bool(clo > nl)}
    n_sig = sum(1 for c in per if per[c]["clears_null"])
    print(f"\n[s15] {n_sig}/{len(per)} concepts individually clear their own null (95% CI lower bound > null).")
    print(f"       Bonferroni: with {len(per)} concepts, ~99.4% CIs give family-wise control; treat "
          "single-concept significance cautiously at small n.")

    os.makedirs(cfg.paths.results, exist_ok=True)
    out = os.path.join(cfg.paths.results, "s15_localization.json")
    with open(out, "w") as f:
        json.dump({"grid": G, "n_pairs": len(items), "n_images": len(set(imgs)),
                   "n_available_images": n_avail, "n_boot": n_boot,
                   "concept_conditioned": {"hit_rate": hit_c, "ci": [loc, hic]},
                   "concept_agnostic": {"hit_rate": hit_a, "ci": [loa, hia]},
                   "random_null": null_rate, "specificity_lift": hit_c - hit_a,
                   "per_concept": per}, f, indent=2)
    print(f"\n[s15] wrote {out}")


if __name__ == "__main__":
    main()
