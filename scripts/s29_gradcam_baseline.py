"""[localization baseline] Grad-CAM++ on a trained black-box detector, as the competitor the
concept-conditioned occlusion is really up against. The referee's point (correct): comparing our
occlusion map only to a concept-agnostic occlusion of the same frozen model is a weak baseline;
the interpretability claim is against a post-hoc heatmap on a trained classifier. This runs
Grad-CAM++ on a DenseNet-121 trained on NIH ChestX-ray14 (torchxrayvision's published weights,
so no training here) and scores it with the SAME pointing game, grid, boxes, and dims as
s15_localization, so the numbers drop straight into Table 3 for a like-for-like comparison.

Needs: pip install torchxrayvision scikit-image  (weights download once, needs internet).
GPU recommended; one forward + one backward per (image, concept).

    python -m scripts.s29_gradcam_baseline --config configs/vindr.yaml \
        --boxes data/raw/vindr/vinbigdata/train.csv --max-images 1500

Use the SAME --max-images you used for s15 so the two cohorts match (the paper's run is 1500
images / 3050 pairs).
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from anywhere

from ovcbmr.config import load_config

# VinDr (VinBigData) class_name -> our concept(s); mirrors s15 / build_manifest_vindr.
VINDR2OURS = {
    "Atelectasis": ["Atelectasis"], "Cardiomegaly": ["Cardiomegaly"],
    "Pleural effusion": ["Pleural Effusion"], "Infiltration": ["Infiltration"],
    "Nodule/Mass": ["Nodule", "Mass"], "Consolidation": ["Consolidation"],
    "Pleural thickening": ["Pleural Thickening"],
}
# our concept -> torchxrayvision NIH pathology name (xrv uses 'Effusion' and 'Pleural_Thickening').
CONCEPT2XRV = {
    "Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Consolidation": "Consolidation",
    "Pleural Effusion": "Effusion", "Infiltration": "Infiltration", "Mass": "Mass",
    "Nodule": "Nodule", "Pleural Thickening": "Pleural_Thickening",
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


def _load_boxes_dims(boxes_csv, dims_csv):
    """base_id -> concept -> [box]; base_id -> (W, H). Identical policy to s15."""
    dims = {}
    if dims_csv and os.path.exists(dims_csv):
        with open(dims_csv, newline="") as f:
            for r in csv.DictReader(f):
                W, H = _pick(r, _W_KEYS), _pick(r, _H_KEYS)
                if W and H:
                    dims[_base(r.get("image_id", ""))] = (W, H)
    boxes = defaultdict(lambda: defaultdict(list))
    with open(boxes_csv, newline="") as f:
        for r in csv.DictReader(f):
            targets = VINDR2OURS.get(str(r.get("class_name", "")).strip())
            if not targets:
                continue
            x0, y0, x1, y1 = (_num(r.get("x_min")), _num(r.get("y_min")),
                              _num(r.get("x_max")), _num(r.get("y_max")))
            if None in (x0, y0, x1, y1):
                continue
            bid = _base(r.get("image_id", ""))
            if bid not in dims:
                W, H = _pick(r, _W_KEYS), _pick(r, _H_KEYS)
                if W and H:
                    dims[bid] = (W, H)
            for c in targets:
                boxes[bid][c].append((x0, y0, x1, y1))
    return boxes, dims


def _covered_cells(bxs, W, H, G):
    import numpy as np
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vindr.yaml")
    ap.add_argument("--boxes", default="data/raw/vindr/vinbigdata/train.csv")
    ap.add_argument("--dims", default="",
                    help="optional CSV image_id,width,height; else width/height columns in --boxes")
    ap.add_argument("--grid", type=int, default=14)
    ap.add_argument("--max-images", type=int, default=1500)
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--method", choices=["gradcam++", "gradcam"], default="gradcam++")
    ap.add_argument("--weights", default="densenet121-res224-nih")
    args = ap.parse_args()
    cfg = load_config(args.config)

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from ovcbmr.eval.metrics import bootstrap_ci
    try:
        import torchxrayvision as xrv
    except ImportError:
        print("[s29] needs torchxrayvision: pip install torchxrayvision scikit-image"); return

    device = getattr(cfg.compute, "device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = xrv.models.DenseNet(weights=args.weights).to(device).eval()
    paths = list(model.pathologies)
    G = args.grid
    seed = int(getattr(cfg, "seed", 0))

    # our concept -> xrv output channel index (skip any concept the model lacks)
    cidx = {}
    for concept, xname in CONCEPT2XRV.items():
        if xname in paths:
            cidx[concept] = paths.index(xname)
        elif xname.replace("_", " ") in paths:
            cidx[concept] = paths.index(xname.replace("_", " "))
    print(f"[s29] DenseNet-121 ({args.weights}); concepts mapped: {sorted(cidx)}")

    boxes, dims = _load_boxes_dims(args.boxes, args.dims)
    d = torch.load(os.path.join(cfg.paths.embeddings, "test_img.pt"), weights_only=False)
    test_ids = sorted(iid for iid in d["ids"] if _base(iid) in boxes and _base(iid) in dims)
    n_avail = len(test_ids)
    if args.max_images and n_avail > args.max_images:
        test_ids = test_ids[:args.max_images]
    print(f"[s29] VinDr test images with box+dims: {n_avail} (using {len(test_ids)}), method {args.method}")
    if not test_ids:
        print("[s29] none scored -- check --boxes/--dims (same files as s15)."); return

    def load_img(path):
        im = np.array(Image.open(path).convert("L")).astype(np.float32)
        im = xrv.datasets.normalize(im, 255)                      # -> [-1024, 1024], xrv convention
        t = torch.from_numpy(im)[None, None]                      # [1,1,H,W]
        t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        return t.to(device)

    def cam(img, k):
        """Grad-CAM(++) map [gh, gw] for output channel k, from the last dense block."""
        img = img.clone().requires_grad_(True)
        A = model.features(img)                                   # [1, C, 7, 7], the conv activations
        A.retain_grad()
        pooled = F.adaptive_avg_pool2d(F.relu(A, inplace=False), 1).flatten(1)
        score = model.classifier(pooled)[0, k]                    # pre-sigmoid class logit
        model.zero_grad(set_to_none=True)
        grads = torch.autograd.grad(score, A, retain_graph=False)[0][0]   # [C, 7, 7]
        Ak = A.detach()[0]                                        # [C, 7, 7]
        if args.method == "gradcam++":
            g2, g3 = grads ** 2, grads ** 3
            denom = 2 * g2 + (Ak * g3).sum(dim=(1, 2), keepdim=True)
            alpha = torch.where(denom != 0, g2 / denom, torch.zeros_like(denom))
            w = (alpha * F.relu(grads)).sum(dim=(1, 2))           # [C]
        else:
            w = grads.mean(dim=(1, 2))                            # plain Grad-CAM
        m = F.relu((w[:, None, None] * Ak).sum(dim=0))            # [7, 7]
        return m

    items, maxnorm = [], 0.0
    img_root = cfg.data.image_root
    for iid in test_ids:
        bid = _base(iid); W, H = dims[bid]
        p = os.path.join(img_root, iid)
        if not os.path.exists(p):
            continue
        img = load_img(p)
        for c, bxs in boxes[bid].items():
            if c not in cidx:
                continue
            cov = _covered_cells(bxs, W, H, G)
            if not cov:
                continue
            maxnorm = max([maxnorm] + [x1 / W for _, _, x1, _ in bxs] + [y1 / H for _, _, _, y1 in bxs])
            m = cam(img, cidx[c])
            mg = F.interpolate(m[None, None], size=(G, G), mode="bilinear", align_corners=False)[0, 0]
            gy, gx = np.unravel_index(int(torch.argmax(mg).item()), (G, G))
            items.append((1.0 if (int(gy), int(gx)) in cov else 0.0, len(cov) / (G * G), c, bid))

    if not items:
        print("[s29] no (image, concept) pairs scored."); return
    if maxnorm > 1.5:
        print(f"[s29] WARNING: normalized box coords reach {maxnorm:.2f} (>1) -- wrong --dims.")

    imgs = [t[3] for t in items]
    null_rate = sum(t[1] for t in items) / len(items)
    hit, lo, hi = bootstrap_ci([t[0] for t in items], groups=imgs, n_boot=args.nboot, seed=seed)
    by_c = defaultdict(list)
    for t in items:
        by_c[t[2]].append(t)

    # side-by-side with our occlusion result if present
    ours = {}
    s15p = os.path.join(cfg.paths.results, "s15_localization.json")
    if os.path.exists(s15p):
        j = json.load(open(s15p))
        ours = {k: v["hit_rate"] for k, v in j.get("per_concept", {}).items()}
        ours["_overall"] = j.get("concept_conditioned", {}).get("hit_rate")

    print(f"\n[s29] Grad-CAM++ pointing game over {len(items)} pairs, {len(set(imgs))} images")
    print(f"  Grad-CAM++ (DenseNet-NIH) : {hit:.3f} [{lo:.3f},{hi:.3f}]")
    if ours.get("_overall") is not None:
        print(f"  ours (concept occlusion)  : {ours['_overall']:.3f}")
    print(f"  random-point null         : {null_rate:.3f}")
    print(f"\n    {'concept':<18}{'GradCAM++':>10}{'ours':>8}{'null':>7}{'n':>6}")
    per = {}
    for c in sorted(by_c, key=lambda k: -(sum(t[0] for t in by_c[k]) / len(by_c[k]))):
        v = by_c[c]
        hc = sum(t[0] for t in v) / len(v); nl = sum(t[1] for t in v) / len(v)
        _, clo, chi = bootstrap_ci([t[0] for t in v], groups=[t[3] for t in v], n_boot=args.nboot, seed=seed)
        og = ours.get(c)
        print(f"    {c:<18}{hc:>10.3f}{(f'{og:.3f}' if og is not None else '  -- '):>8}{nl:>7.3f}{len(v):>6}")
        per[c] = {"gradcampp_hit": hc, "ci": [clo, chi], "ours_occlusion_hit": og, "null": nl, "n": len(v)}

    win_ours = sum(1 for c in per if per[c]["ours_occlusion_hit"] is not None
                   and per[c]["ours_occlusion_hit"] >= per[c]["gradcampp_hit"])
    print(f"\n[s29] concept-occlusion >= Grad-CAM++ on {win_ours}/{len(per)} concepts. "
          "Report both; this is the honest black-box localization baseline.")
    os.makedirs(cfg.paths.results, exist_ok=True)
    out = os.path.join(cfg.paths.results, "s29_gradcam_baseline.json")
    with open(out, "w") as f:
        json.dump({"method": args.method, "weights": args.weights, "grid": G,
                   "n_pairs": len(items), "n_images": len(set(imgs)), "n_boot": args.nboot,
                   "overall_hit": hit, "overall_ci": [lo, hi], "null": null_rate,
                   "ours_overall": ours.get("_overall"), "per_concept": per}, f, indent=2)
    print(f"[s29] wrote {out}")


if __name__ == "__main__":
    main()
