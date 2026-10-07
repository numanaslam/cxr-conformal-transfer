"""CXR-LT 2026 manifest handling + frozen-encoder image loading.

Manifest is read with the stdlib `csv` module (no pandas dependency) so `read_manifest`
and `filter_frontal` are testable anywhere. Image decoding + preprocessing lazily import
torch/PIL and use the encoder's own preprocess transform (BiomedCLIP).

Expected manifest columns (adjust names in configs/default.yaml -> data:):
    image_id, patient_id, projection, split, <30 known label cols>, <6 unseen label cols>
"""
from __future__ import annotations
import csv
import os


def read_manifest(csv_path):
    """Read the CXR-LT label CSV into a list of dict rows (stdlib)."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def filter_frontal(records, projection_col="projection", keep=("PA", "AP")):
    """Keep frontal projections only (projection confounder, protocol §4.1)."""
    keep = {k.upper() for k in keep}
    out = [r for r in records if str(r.get(projection_col, "")).upper() in keep]
    return out


def label_vector(record, concept_cols):
    """Binary label vector (list[int]) for the given ordered label columns."""
    from .splits import _truthy
    return [1 if _truthy(record.get(c)) else 0 for c in concept_cols]


def build_items(records, cfg):
    """Turn manifest rows into lightweight item dicts consumed downstream."""
    d = cfg.data
    id_col = getattr(d, "id_col", "image_id")
    pat_col = getattr(d, "patient_col", "patient_id")
    proj_col = getattr(d, "projection_col", "projection")
    root = getattr(d, "image_root", "")
    items = []
    for r in records:
        iid = r[id_col]
        items.append({
            "id": iid,
            "path": os.path.join(root, iid) if root and not os.path.isabs(iid) else iid,
            "patient_id": r.get(pat_col),
            "projection": r.get(proj_col),
            "split": r.get(getattr(d, "split_col", "split")),
        })
    return items


# --------------------------------------------------------------------------- #
# Heavy path (lazy torch/PIL): image dataset + loader for frozen encoding.
# Dataset + collate are MODULE-LEVEL so they pickle for Windows 'spawn' DataLoader
# workers — a class/closure defined *inside* a function is not picklable on Windows.
# --------------------------------------------------------------------------- #
class _CXRDataset:
    """Indexable image dataset; duck-types torch Dataset (DataLoader only needs
    __len__/__getitem__). PIL is imported lazily so this module stays torch-free at import."""

    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        it = self.items[i]
        img = Image.open(it["path"]).convert("RGB")
        return self.preprocess(img), it["id"]


def _collate_pixels(batch):
    import torch
    xs = torch.stack([b[0] for b in batch], dim=0)
    ids = [b[1] for b in batch]
    return xs, ids


def make_image_loader(items, preprocess, batch_size=256, num_workers=4, shuffle=False,
                      pin_memory=True, prefetch_factor=2):
    """DataLoader over item dicts, applying the encoder's `preprocess` to each image.

    Yields (pixel_tensor, ids). Throughput levers (the GPU starves on PNG decode, not the
    forward pass): more workers (8-12), pin_memory (faster H2D), and prefetch to keep the
    queue full. On Windows set cfg.data.num_workers: 0 only if spawn workers hang.
    Host-RAM buffer ~= num_workers * prefetch_factor * batch_size images — lower these if RAM
    is tight; lower batch_size if VRAM OOMs.
    """
    from torch.utils.data import DataLoader
    kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
              collate_fn=_collate_pixels, pin_memory=pin_memory)
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = prefetch_factor
    return DataLoader(_CXRDataset(items, preprocess), **kw)
