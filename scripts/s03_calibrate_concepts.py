"""[build 2] Per-concept Platt calibration on the conformal-independent calib split.

Uses cached calib image embeddings + concept text embeddings -> cosine sims per concept,
then fits (w_k, b_k) against the calib known-concept labels. Writes:
    processed/platt.json   {concept: [w, b]}

    python -m scripts.s03_calibrate_concepts --config configs/default.yaml
"""
from __future__ import annotations
import json
import os
import sys

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.calibrate import fit_bank, save_bank


def main():
    cfg = parse_config_arg()
    import torch  # heavy path

    emb_dir = cfg.paths.embeddings
    calib_path = os.path.join(emb_dir, "calib_img.pt")
    text_path = os.path.join(emb_dir, "concept_text.pt")
    idx_path = os.path.join(cfg.paths.processed, "index.json")
    meta_path = os.path.join(cfg.paths.processed, "meta.json")
    for p in (calib_path, text_path, idx_path, meta_path):
        if not os.path.exists(p):
            print(f"[s03] missing {p} — run s01 then s02 first.")
            sys.exit(1)

    calib = torch.load(calib_path, weights_only=False)   # our own trusted local cache
    text = torch.load(text_path, weights_only=False)
    with open(idx_path) as f:
        items = {it["id"]: it for it in json.load(f)}
    with open(meta_path) as f:
        vocab = json.load(f)["vocabulary"]

    concepts = text["concepts"]
    col = {c: i for i, c in enumerate(vocab)}                 # label column per concept
    sims = calib["emb"] @ text["pos"].t()                     # [N, K] cosine logits
    ids = calib["ids"]

    sims_by_c, labels_by_c = {}, {}
    for ki, c in enumerate(concepts):
        if c not in col:
            continue
        s_list, y_list = [], []
        for row, iid in enumerate(ids):
            it = items.get(iid)
            if it is None:
                continue
            s_list.append(float(sims[row, ki]))
            y_list.append(int(it["labels_known"][col[c]]))
        sims_by_c[c], labels_by_c[c] = s_list, y_list

    params = fit_bank(sims_by_c, labels_by_c, max_iter=int(getattr(cfg.calibrate, "max_iter", 200)))
    out = os.path.join(cfg.paths.processed, "platt.json")
    save_bank(params, out)
    print(f"[s03] fit per-concept Platt for {len(params)} concepts -> {out}")


if __name__ == "__main__":
    main()
