"""[build 2 check] Verify frozen BiomedCLIP runs end-to-end on this GPU — NO dataset needed.

Loads the encoder, embeds the concept bank (text tower), pushes a synthetic image through the
image tower, and computes cosine logits — the exact path s02 uses. Confirms embed_dim and that
the image/text towers produce aligned, L2-normalized embeddings. Downloads ~0.4 GB of weights
on first run.

    python -m scripts.s02b_check_encoder --config configs/default.yaml
"""
from __future__ import annotations

from ovcbmr.config import parse_config_arg
from ovcbmr.concept import bank as B


def main():
    cfg = parse_config_arg()
    import torch
    from PIL import Image
    from ovcbmr.models.encoder import (
        load_biomedclip, encode_concept_bank, cosine_logits)

    device = cfg.compute.device
    model, preprocess, tokenizer = load_biomedclip(device, getattr(cfg.model, "hf_id"))
    print(f"[s02b] BiomedCLIP loaded on {device}")

    # --- text tower: concept bank ---
    bank = B.load_bank(getattr(cfg.concepts, "bank", "default"))
    prompts = B.all_prompts(bank, "known")
    entries = encode_concept_bank(model, tokenizer, prompts, device)
    names = list(prompts.keys())
    pos = torch.stack([entries[n]["pos"] for n in names])                 # [K, D]
    D = int(pos.shape[1])
    print(f"[s02b] concept text embeddings: {tuple(pos.shape)}  (K={len(names)}, D={D})")
    assert abs(float(pos[0].norm()) - 1.0) < 1e-3, "text embeddings must be L2-normalized"

    # --- image tower: synthetic image (no files needed) ---
    size = int(getattr(cfg.data, "image_size", 224))
    img = Image.new("RGB", (size, size), (128, 128, 128))
    px = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        z = torch.nn.functional.normalize(model.encode_image(px), dim=-1).float().cpu()  # [1, D]
    print(f"[s02b] image embedding: {tuple(z.shape)}")
    assert z.shape[1] == D, f"image/text dim mismatch: {z.shape[1]} vs {D}"

    # --- cosine path (what concept scoring uses) ---
    logits = cosine_logits(z, pos)                                        # [1, K]
    top = int(logits.argmax(dim=1)[0])
    print(f"[s02b] cosine logits: {tuple(logits.shape)} | argmax for gray image = "
          f"'{names[top]}' ({float(logits[0, top]):.3f})   [sanity: path runs]")

    cfg_dim = int(getattr(cfg.model, "embed_dim", D))
    if D != cfg_dim:
        print(f"[s02b] NOTE: set model.embed_dim: {D} in configs/default.yaml (currently {cfg_dim})")
    print("[s02b] ENCODER OK — s02 will run once the manifest + images are in place")


if __name__ == "__main__":
    main()
