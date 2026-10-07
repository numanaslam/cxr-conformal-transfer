"""[figures] Concept-specific saliency for the interpretability figure. For deferred VinDr
cases, overlay an OCCLUSION saliency map for the *uncertain concept* — where the evidence for
that concept lives. Concept-specific (uses our trained head p_k), not black-box GradCAM: we
occlude image regions and measure the drop in the concept probability. Needs GPU (re-encodes a
handful of images ~G*G times each).

    python -m scripts.s14_concept_saliency --config configs/vindr.yaml
"""
from __future__ import annotations
import json
import os

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.concept.valve import image_unexplained
from ovcbmr.ood.conformal import split_conformal_threshold


def _abnormal(emb_dir, proc, split, heads, vocab, nidx):
    import torch
    d = torch.load(os.path.join(emb_dir, f"{split}_img.pt"), weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    Pm = predict_matrix(d["emb"].numpy(), heads, vocab)
    out = []
    for i, iid in enumerate(d["ids"]):
        it = items.get(iid)
        if it is None or it["labels_known"][nidx] == 1:
            continue
        out.append((image_unexplained(Pm[i].tolist(), vocab, "Normal")[1], iid, Pm[i]))
    return out


def main():
    cfg = parse_config_arg()
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.transforms import Normalize
    from PIL import Image
    from ovcbmr.models.encoder import load_biomedclip

    ch = cfg.cross_hospital
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    nidx = vocab.index("Normal")
    ab_cols = [j for j, c in enumerate(vocab) if c != "Normal"]
    alpha = float(getattr(cfg.conformal, "alpha_flag", 0.05))

    nih_cal = [r[0] for r in _abnormal(ch.nih_embeddings, ch.nih_processed, "calib", heads, vocab, nidx)]
    vin_te = _abnormal(cfg.paths.embeddings, cfg.paths.processed, "test", heads, vocab, nidx)
    q = split_conformal_threshold(nih_cal, alpha)
    flagged = sorted([r for r in vin_te if q != float("inf") and r[0] > q], key=lambda r: -r[0])
    # pick 4 deferred cases with DISTINCT uncertain concepts (dedupe by driving concept)
    seen_c, seen_id, picks = set(), set(), []
    for r in flagged:
        cj = min(ab_cols, key=lambda j: abs(float(r[2][j]) - 0.5))
        if cj not in seen_c:
            seen_c.add(cj); seen_id.add(r[1]); picks.append(r)
        if len(picks) == 4:
            break
    for r in flagged:                                           # top up if <4 distinct concepts
        if len(picks) >= 4:
            break
        if r[1] not in seen_id:
            seen_id.add(r[1]); picks.append(r)
    if not picks:
        picks = vin_te[:4]

    device = cfg.compute.device
    model, preprocess, _ = load_biomedclip(device, getattr(cfg.model, "hf_id"))
    mean = std = None
    for t in getattr(preprocess, "transforms", []):
        if isinstance(t, Normalize):
            mean, std = list(t.mean), list(t.std)
    if mean is None:
        mean = [0.48145466, 0.4578275, 0.40821073]; std = [0.26862954, 0.26130258, 0.27577711]
    mean_t, std_t = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)

    def encode(batch):
        with torch.inference_mode():
            z = torch.nn.functional.normalize(model.encode_image(batch.to(device)), dim=-1)
        return z.float().cpu().numpy()

    G = 14                                                      # 224/14 = 16 px cells (exact)
    cmap = plt.get_cmap("jet")
    img_root = cfg.data.image_root
    fig, axes = plt.subplots(2, 2, figsize=(9, 10.4))
    for ax, (s, iid, pv) in zip(axes.ravel(), picks):
        cstar = min(ab_cols, key=lambda j: abs(float(pv[j]) - 0.5))
        cname = vocab[cstar]
        try:
            base_t = preprocess(Image.open(os.path.join(img_root, iid)).convert("RGB"))
        except Exception:                                       # noqa: BLE001
            ax.text(0.5, 0.5, "(image not found)", ha="center", va="center"); ax.axis("off"); continue
        H = base_t.shape[1]; cell = H // G
        variants = [base_t.clone()]
        for gy in range(G):
            for gx in range(G):
                v = base_t.clone()
                v[:, gy*cell:(gy+1)*cell, gx*cell:(gx+1)*cell] = 0.0   # occlude (baseline = mean)
                variants.append(v)
        P = predict_matrix(encode(torch.stack(variants)), heads, vocab)[:, cstar]
        sal = np.clip(P[0] - np.asarray(P[1:]).reshape(G, G), 0.0, None)   # drop in p_k when hidden
        if sal.max() > 1e-9:
            sal = sal / sal.max()                                # per-image normalize to [0,1]
        sal_up = np.asarray(Image.fromarray((sal * 255).astype("uint8")).resize((H, H), Image.BILINEAR),
                            dtype="float32") / 255.0             # smooth upsample
        disp = (base_t * std_t + mean_t).clamp(0, 1).mean(0).numpy()      # de-normalized grayscale
        ax.imshow(disp, cmap="gray")
        rgba = cmap(sal_up)
        rgba[..., 3] = np.clip((sal_up - 0.25) / 0.75, 0.0, 1.0) * 0.7    # per-pixel alpha: low sal -> clear
        ax.imshow(rgba)
        ax.set_title(f"Deferred — uncertain: {cname} (p={float(pv[cstar]):.2f})", fontsize=11)
        ax.axis("off")
    fig.suptitle("Concept-specific occlusion saliency on deferred VinDr cases\n"
                 "(warm = regions whose removal most lowers the uncertain concept's probability)",
                 fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, wspace=0.06, hspace=0.14)
    out = os.path.join(cfg.paths.results, "fig3b_concept_saliency.png")
    os.makedirs(cfg.paths.results, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"[s14] wrote {out} | q={q:.3f} | deferred pool={len(flagged)}")


if __name__ == "__main__":
    main()
