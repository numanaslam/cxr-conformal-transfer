"""[figure asset] Export a real, coherent example for the method figure (Fig. 1).

Picks one ACCEPTED, in-vocabulary abnormal VinDr test case and exports three things
for the SAME patient, so the method figure's input radiograph, its concept-probability
bar panel, and its localization thumbnail all show real data from one image:
  * fig_ex_cxr.png       -- the clean radiograph (no overlay)
  * fig_ex_concepts.json -- the REAL calibrated concept probabilities p_k, plus A, m, s(x)
  * fig_ex_saliency.png  -- concept-conditioned occlusion saliency for the top concept

    python -m scripts.make_fig_example --config configs/vindr.yaml
    # optional overrides:  --image-id <id>   --concept <Name>   --outdir docs/assets

Needs the VinDr cached embeddings, the NIH-trained heads, and VinDr images on disk.
Selecting the case and computing p_k needs no GPU; only the saliency overlay re-encodes
the image (GPU). Use --no-saliency to skip that and stay CPU-only.
"""
from __future__ import annotations
import argparse
import json
import os

from ovcbmr.config import load_config
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.concept.valve import image_unexplained
from ovcbmr.ood.conformal import split_conformal_threshold


def _load(emb_dir, split):
    import torch
    d = torch.load(os.path.join(emb_dir, f"{split}_img.pt"), weights_only=False)
    return d["emb"].numpy(), d["ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vindr.yaml")
    ap.add_argument("--image-id", default="", help="force a specific VinDr test image id")
    ap.add_argument("--concept", default="", help="force the concept for the saliency overlay")
    ap.add_argument("--alpha", type=float, default=0.05, help="G1 target (valve accept threshold)")
    ap.add_argument("--grid", type=int, default=14)
    ap.add_argument("--outdir", default="")
    ap.add_argument("--no-saliency", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    import numpy as np

    ch = cfg.cross_hospital
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    nidx = vocab.index("Normal")
    ab_cols = [j for j, c in enumerate(vocab) if c != "Normal"]

    with open(os.path.join(cfg.paths.processed, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}

    def score(P_row):
        return image_unexplained(P_row.tolist(), vocab, "Normal")[1]

    def is_invocab_abnormal(iid):
        it = items.get(iid)
        return bool(it) and any(int(it["labels_known"][j]) == 1 for j in ab_cols)

    # --- site-conditional G1 threshold on VinDr calib in-vocab known-abnormal ---
    ce, cids = _load(cfg.paths.embeddings, "calib")
    Pc = predict_matrix(ce, heads, vocab)
    cal_scores = [score(Pc[i]) for i, iid in enumerate(cids) if is_invocab_abnormal(iid)]
    q_site = split_conformal_threshold(cal_scores, args.alpha)

    # --- score all VinDr test images ---
    te, tids = _load(cfg.paths.embeddings, "test")
    Pt = predict_matrix(te, heads, vocab)

    def present(P_row):
        return [(vocab[j], float(P_row[j])) for j in ab_cols if P_row[j] >= 0.5]

    cand = []
    for i, iid in enumerate(tids):
        if not is_invocab_abnormal(iid):
            continue
        s = score(Pt[i])
        cand.append((iid, i, s, s <= q_site, present(Pt[i])))

    def rank_key(c):
        _, _, _, acc, pres = c
        top = max((p for _, p in pres), default=0.0)
        return (int(acc), 1 if 1 <= len(pres) <= 2 else 0, top)

    cand.sort(key=rank_key, reverse=True)
    if not cand:
        print("[fig] no in-vocabulary abnormal VinDr test cases found; check config/paths."); return

    print(f"[fig] q_site (site-conditional G1 accept threshold, alpha={args.alpha}) = {q_site:.3f}")
    print("[fig] top candidates  (id | s(x) | accepted | present concepts >=0.5):")
    for iid, i, s, acc, pres in cand[:8]:
        ps = ", ".join(f"{c} {p:.2f}" for c, p in pres) or "(none>=0.5)"
        print(f"    {iid:<30} s={s:.3f}  acc={int(acc)}  [{ps}]")

    chosen = None
    if args.image_id:
        for c in cand:
            if c[0] == args.image_id or os.path.splitext(os.path.basename(c[0]))[0] == args.image_id:
                chosen = c; break
        if chosen is None:
            print(f"[fig] --image-id {args.image_id} not among candidates; falling back to top.")
    chosen = chosen or cand[0]
    iid, i, s, acc, pres = chosen
    P_row = Pt[i]
    A = 1.0 - float(P_row[nidx])
    m = max((float(P_row[j]) for j in ab_cols), default=0.0)
    pk = {vocab[j]: round(float(P_row[j]), 3) for j in range(len(vocab))}
    top_concept = args.concept or (max(pres, key=lambda t: t[1])[0] if pres
                                   else vocab[ab_cols[int(np.argmax([P_row[j] for j in ab_cols]))]])

    outdir = args.outdir or os.path.join(cfg.paths.results, "fig_example")
    os.makedirs(outdir, exist_ok=True)
    meta = {"image_id": iid, "s": round(s, 4), "q_site": round(q_site, 4),
            "accepted": bool(acc), "A": round(A, 4), "max_anchor": round(m, 4),
            "present_concepts": pres, "top_concept": top_concept, "p_k": pk}
    with open(os.path.join(outdir, "fig_ex_concepts.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[fig] CHOSEN  {iid}")
    print(f"      s(x)={s:.3f}  (<= q_site {q_site:.3f}  => accepted={int(acc)});  top concept = {top_concept}")
    print(f"      A = 1 - p(Normal) = {A:.3f};  max anchor m = {m:.3f}")
    print("[fig] REAL calibrated concept probabilities p_k (paste these into the figure bars):")
    for c in sorted(vocab, key=lambda c: -pk[c]):
        print(f"      {c:<18} {pk[c]:.3f}")

    # --- clean radiograph (no overlay) ---
    from PIL import Image
    raw = Image.open(os.path.join(cfg.data.image_root, iid)).convert("L")
    raw.convert("RGB").save(os.path.join(outdir, "fig_ex_cxr.png"))
    print(f"\n[fig] wrote {outdir}/fig_ex_cxr.png  and  fig_ex_concepts.json")

    if args.no_saliency:
        return

    # --- concept-conditioned occlusion saliency overlay for the SAME image (GPU re-encode) ---
    try:
        import torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ovcbmr.models.encoder import load_biomedclip

        device = cfg.compute.device
        model, preprocess, _ = load_biomedclip(device, getattr(cfg.model, "hf_id"))
        cidx = vocab.index(top_concept)
        G = args.grid
        base_t = preprocess(raw.convert("RGB"))
        Himg = base_t.shape[1]
        cell = Himg // G
        variants = [base_t.clone()]
        for gy in range(G):
            for gx in range(G):
                v = base_t.clone()
                v[:, gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell] = 0.0
                variants.append(v)
        with torch.inference_mode():
            z = torch.nn.functional.normalize(model.encode_image(torch.stack(variants).to(device)), dim=-1)
        P = predict_matrix(z.float().cpu().numpy(), heads, vocab)
        sal = np.clip(P[0, cidx] - np.asarray(P[1:, cidx]).reshape(G, G), 0.0, None)
        # Display cleanup only (the s15 pointing-game hit-rate uses the raw map, not this):
        # zero the outer ring (occluding a border cell is a known artifact), 3x3 box-smooth,
        # normalize, and keep only the salient region so the radiograph stays visible under a
        # focused heatmap rather than a dense full-grid wash.
        sal[0, :] = sal[:, 0] = sal[:, -1] = 0.0
        sp = np.pad(sal, 1, mode="edge")
        sal = sum(sp[i:i + G, j:j + G] for i in range(3) for j in range(3)) / 9.0
        sal = sal / (sal.max() + 1e-8)
        sal_disp = np.where(sal >= 0.40, sal, np.nan)

        disp = raw.resize((Himg, Himg))
        fig, axp = plt.subplots(figsize=(3, 3))
        axp.imshow(disp, cmap="gray")
        axp.imshow(sal_disp, cmap="turbo", alpha=0.6, interpolation="bilinear",
                   extent=(0, Himg, Himg, 0), vmin=0.0, vmax=1.0)
        axp.axis("off")
        fig.savefig(os.path.join(outdir, "fig_ex_saliency.png"), dpi=200,
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"[fig] wrote {outdir}/fig_ex_saliency.png  (concept-conditioned occlusion, {top_concept})")
    except Exception as e:  # noqa: BLE001
        print(f"[fig] saliency overlay skipped ({type(e).__name__}: {e}); CXR + p_k still exported.")


if __name__ == "__main__":
    main()
