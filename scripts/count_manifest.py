"""[manifest] Post-filter counts that ACTUALLY entered the experiments.

Prints, per split, the number of images that were embedded and used (from the cached
`{split}_img.pt` id lists) plus unique-patient counts from the processed index when a
patient field is present. These trace to *our pipeline* (after frontal filtering +
patient-level splits), unlike the dataset-paper headline sizes.

    python -m scripts.count_manifest --config configs/nih.yaml
    python -m scripts.count_manifest --config configs/vindr.yaml
"""
from __future__ import annotations
import json
import os

from ovcbmr.config import parse_config_arg

_PID_KEYS = ("patient_id", "patient", "PatientID", "pid")


def _patient_of(item):
    for k in _PID_KEYS:
        if isinstance(item, dict) and item.get(k) is not None:
            return item[k]
    return None


def main():
    cfg = parse_config_arg()
    import torch

    emb, proc = cfg.paths.embeddings, cfg.paths.processed
    print(f"[manifest] dataset={cfg.data.dataset}  frontal_only={getattr(cfg.data, 'frontal_only', None)}")
    print(f"           embeddings={emb}")

    total = 0
    per_split_ids = {}
    for s in ("train", "calib", "val", "test"):
        p = os.path.join(emb, f"{s}_img.pt")
        if not os.path.exists(p):
            continue
        ids = torch.load(p, weights_only=False)["ids"]
        per_split_ids[s] = list(ids)
        total += len(ids)
        print(f"  {s:6s}: {len(ids):>7d} images")
    print(f"  {'TOTAL':6s}: {total:>7d} images (post-filter, embedded)")

    idx = os.path.join(proc, "index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            items = json.load(f)
        by_id = {it.get("id"): it for it in items if isinstance(it, dict)}
        pats_all = {_patient_of(it) for it in items}
        pats_all.discard(None)
        print(f"  processed index: {len(items)} items"
              + (f", {len(pats_all)} unique patients" if pats_all else " (no patient field)"))
        # per-split unique patients (join embedding ids -> index patient)
        if pats_all:
            for s, ids in per_split_ids.items():
                ps = {_patient_of(by_id[i]) for i in ids if i in by_id}
                ps.discard(None)
                print(f"    {s:6s}: {len(ps):>6d} unique patients")
    else:
        print(f"  (no {idx}; patient counts unavailable)")


if __name__ == "__main__":
    main()
