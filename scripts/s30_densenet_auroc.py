"""[fine-tuned baseline] How much discrimination does the frozen encoder cost versus a CNN
trained end-to-end on the same source data? A reviewer's fair question. We take a DenseNet-121
trained on NIH ChestX-ray14 (torchxrayvision's published weights -- the canonical CheXNet-style
model, no training here), run it on the VinDr test images, and report its per-concept AUROC next
to our frozen-BiomedCLIP linear-probe CBM (from s19_vindr_concepts.json / the paper's Table 1).
This gives the trained-CNN-vs-frozen-CBM comparison directly on the deployment site, at no extra
cost in labels or GPU beyond one forward pass per image. Needs torchxrayvision (as s29).

    python -m scripts.s30_densenet_auroc --config configs/vindr.yaml
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from anywhere

from ovcbmr.config import load_config
from ovcbmr.eval.metrics import auroc

CONCEPT2XRV = {
    "Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Consolidation": "Consolidation",
    "Pleural Effusion": "Effusion", "Infiltration": "Infiltration", "Mass": "Mass",
    "Nodule": "Nodule", "Pleural Thickening": "Pleural_Thickening",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vindr.yaml")
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    try:
        import torchxrayvision as xrv
    except ImportError:
        print("[s30] needs torchxrayvision: pip install torchxrayvision scikit-image"); return

    ch = cfg.cross_hospital
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    with open(os.path.join(cfg.paths.processed, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}

    device = getattr(cfg.compute, "device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = xrv.models.DenseNet(weights="densenet121-res224-nih").to(device).eval()
    paths = list(model.pathologies)
    cidx = {c: paths.index(x) for c, x in CONCEPT2XRV.items()
            if x in paths or x.replace("_", " ") in paths}

    d = torch.load(os.path.join(cfg.paths.embeddings, "test_img.pt"), weights_only=False)
    ids = [iid for iid in d["ids"] if iid in items]
    img_root = cfg.data.image_root

    def load_img(path):
        im = np.array(Image.open(path).convert("L")).astype(np.float32)
        im = xrv.datasets.normalize(im, 255)
        t = torch.from_numpy(im)[None, None]
        return F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False).to(device)

    # DenseNet probabilities per concept + ground-truth labels
    probs = {c: [] for c in cidx}
    ys = {c: [] for c in cidx}
    pats = []
    n = 0
    with torch.inference_mode():
        for iid in ids:
            p = os.path.join(img_root, iid)
            if not os.path.exists(p):
                continue
            out = model(load_img(p))[0].float().cpu().numpy()   # [n_pathologies] in [0,1]
            lk = items[iid]["labels_known"]
            for c, k in cidx.items():
                probs[c].append(float(out[k]))
                ys[c].append(int(lk[vocab.index(c)]))
            pats.append(items[iid].get("patient_id", iid))
            n += 1
            if n % 500 == 0:
                print(f"[s30] {n} images...")
    print(f"[s30] scored {n} VinDr test images with DenseNet-121 (NIH-trained)\n")

    # our CBM VinDr AUROC (from s19 if present)
    ours = {}
    s19 = os.path.join(cfg.paths.results, "s19_vindr_concepts.json")
    if os.path.exists(s19):
        j = json.load(open(s19))
        pc = j.get("per_concept", j)
        for c in cidx:
            v = pc.get(c) if isinstance(pc, dict) else None
            if isinstance(v, dict):
                ours[c] = v.get("vindr_auroc") or v.get("auroc") or v.get("auc")

    print(f"    {'concept':<18}{'DenseNet(NIH)':>14}{'ours CBM':>10}{'n+':>7}")
    out_rows, dn_list, our_list, dn_all = {}, [], [], []
    for c in sorted(cidx):
        y = np.array(ys[c]); s = np.array(probs[c])
        if y.sum() == 0 or y.sum() == len(y):
            au = float("nan")
        else:
            au = auroc(s.tolist(), y.tolist())
        og = ours.get(c)
        if not np.isnan(au):
            dn_all.append(au)
        if og is not None and not np.isnan(au):
            dn_list.append(au); our_list.append(og)
        print(f"    {c:<18}{au:>14.3f}{(f'{og:.3f}' if og is not None else '   -- '):>10}{int(y.sum()):>7}")
        out_rows[c] = {"densenet_auroc": au, "ours_cbm_auroc": og, "n_pos": int(y.sum())}

    if dn_list:
        md, mo = float(np.mean(dn_list)), float(np.mean(our_list))
        print(f"\n[s30] mean over matched concepts: DenseNet {md:.3f} vs frozen CBM {mo:.3f} "
              f"(gap {md - mo:+.3f}).")
        print("[s30] Read: a positive gap is the discrimination the frozen encoder gives up to a "
              "trained CNN; report it so the framework is not read as competitive on accuracy.")
    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, "s30_densenet_auroc.json")
    with open(op, "w") as f:
        json.dump({"n_images": n, "per_concept": out_rows,
                   "mean_densenet": (float(np.mean(dn_all)) if dn_all else None),
                   "mean_ours": (float(np.mean(our_list)) if our_list else None)}, f, indent=2)
    print(f"[s30] wrote {op}")


if __name__ == "__main__":
    main()
