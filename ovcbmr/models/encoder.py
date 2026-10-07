"""Frozen medical vision-language encoder (BiomedCLIP) via open_clip.

All heavy imports are lazy so the rest of the package imports without torch/open_clip.
The encoder is FROZEN (protocol §0): we only ever run forward passes and cache embeddings.
"""
from __future__ import annotations


def load_biomedclip(device="cuda", hf_id="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
    """Return (model, preprocess, tokenizer), model in eval() on `device`, grads off."""
    import torch
    import open_clip
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True          # autotune convs for the fixed 224 input
        torch.backends.cuda.matmul.allow_tf32 = True    # TF32 matmul (Ampere+) — free throughput
        torch.backends.cudnn.allow_tf32 = True
    model, preprocess = open_clip.create_model_from_pretrained(hf_id)
    tokenizer = open_clip.get_tokenizer(hf_id)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, tokenizer


def encode_images(model, loader, device="cuda", amp=True):
    """Run the frozen image tower over a DataLoader; return (embeddings, ids).

    embeddings : L2-normalized float tensor [N, D]; ids : list[str] aligned to rows.
    inference_mode + autocast (fp16) + non_blocking H2D + a throughput print so you can see
    whether the GPU is fed (img/s) vs. starving on data loading.
    """
    import time
    import torch
    feats, ids, n = [], [], 0
    use_amp = amp and device == "cuda"
    t0 = time.time()
    with torch.inference_mode():
        for pixels, batch_ids in loader:
            pixels = pixels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                z = model.encode_image(pixels)
            z = torch.nn.functional.normalize(z, dim=-1)
            feats.append(z.float().cpu())
            ids.extend(batch_ids)
            n += len(batch_ids)
    dt = max(time.time() - t0, 1e-9)
    print(f"[encode] {n} imgs in {dt:.0f}s = {n / dt:.0f} img/s")
    return torch.cat(feats, dim=0), ids


def encode_texts(model, tokenizer, prompts, device="cuda"):
    """Encode a list of prompts -> L2-normalized text embeddings [len(prompts), D]."""
    import torch
    with torch.no_grad():
        toks = tokenizer(prompts).to(device)
        t = model.encode_text(toks)
        t = torch.nn.functional.normalize(t, dim=-1)
    return t.float().cpu()


def encode_concept_bank(model, tokenizer, prompts_by_concept, device="cuda"):
    """{concept: {"pos":[...],"neg":[...]}} -> {concept: {"pos": emb_mean, "neg": emb_mean}}.

    Prompt-ensemble means (protocol §5): average the normalized prompt embeddings, renormalize.
    """
    import torch
    out = {}
    for concept, groups in prompts_by_concept.items():
        entry = {}
        for key, prompts in groups.items():
            if not prompts:
                continue
            emb = encode_texts(model, tokenizer, prompts, device)
            mean = torch.nn.functional.normalize(emb.mean(dim=0, keepdim=True), dim=-1)
            entry[key] = mean.squeeze(0)
        out[concept] = entry
    return out


def cosine_logits(image_emb, concept_pos_emb):
    """Cosine similarities [N, K] between image embeddings and per-concept positive embeddings.
    (Embeddings are already L2-normalized, so this is a matmul.)
    """
    import torch
    return image_emb @ concept_pos_emb.t()
