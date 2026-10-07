"""[build 2] Cache frozen BiomedCLIP embeddings for every split + the concept bank.

Writes (under cfg.paths.embeddings):
    <split>_img.pt        {"emb": FloatTensor[N,D], "ids": [str]}
    concept_text.pt       {"concepts": [name], "pos": [K,D], "neg": [K,D]}

Requires torch + open_clip + the images (SERVER_SETUP.md). Run after s01.

    python -m scripts.s02_extract_embeddings --config configs/default.yaml
"""
from __future__ import annotations
import json
import os
import sys

from ovcbmr.config import parse_config_arg
from ovcbmr.utils.seed import set_seed
from ovcbmr.io.datasets import make_image_loader
from ovcbmr.concept import bank as B


def main():
    cfg = parse_config_arg()
    set_seed(cfg.seed)
    import torch  # heavy path
    from ovcbmr.models.encoder import (
        load_biomedclip, encode_images, encode_concept_bank)

    idx_path = os.path.join(cfg.paths.processed, "index.json")
    if not os.path.exists(idx_path):
        print("[s02] processed/index.json missing — run s01_prepare_cxrlt first.")
        sys.exit(1)
    with open(idx_path) as f:
        items = json.load(f)

    os.makedirs(cfg.paths.embeddings, exist_ok=True)
    device = cfg.compute.device
    model, preprocess, tokenizer = load_biomedclip(device, getattr(cfg.model, "hf_id"))
    print(f"[s02] BiomedCLIP loaded on {device}")

    # concept-bank text embeddings (prompt-ensemble means)
    bank = B.load_bank(getattr(cfg.concepts, "bank", "default"))
    prompts = B.all_prompts(bank, "known")
    entries = encode_concept_bank(model, tokenizer, prompts, device)
    names = list(prompts.keys())
    pos = torch.stack([entries[n]["pos"] for n in names])
    neg = torch.stack([entries[n]["neg"] for n in names if "neg" in entries[n]]) \
        if getattr(cfg.concepts, "negatives", True) else None
    torch.save({"concepts": names, "pos": pos, "neg": neg},
               os.path.join(cfg.paths.embeddings, "concept_text.pt"))
    print(f"[s02] cached concept_text.pt (K={len(names)})")

    # per-split image embeddings
    by_split = {}
    for it in items:
        by_split.setdefault(it.get("split", "train"), []).append(it)
    bs = int(getattr(cfg.compute, "batch_size", 256))
    nw = int(getattr(cfg.data, "num_workers", 4))
    pin = bool(getattr(cfg.compute, "pin_memory", True))
    pf = int(getattr(cfg.data, "prefetch_factor", 2))
    for split, split_items in by_split.items():
        loader = make_image_loader(split_items, preprocess, batch_size=bs, num_workers=nw,
                                   pin_memory=pin, prefetch_factor=pf)
        emb, ids = encode_images(model, loader, device, amp=getattr(cfg.compute, "amp", True))
        torch.save({"emb": emb, "ids": ids},
                   os.path.join(cfg.paths.embeddings, f"{split}_img.pt"))
        print(f"[s02] cached {split}_img.pt  [{tuple(emb.shape)}]")

    print(f"[s02] done -> {cfg.paths.embeddings}")


if __name__ == "__main__":
    main()
