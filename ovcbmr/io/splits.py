"""Patient-level dataset splits with **no patient leakage** + a dedicated conformal
calibration split. Pure standard library (no pandas/torch) so the split logic is
unit-testable anywhere.

Design (protocol §4.1):
  * Splitting is by PATIENT, never by image — every image of a patient lands in one split.
  * Assignment is a deterministic hash of (seed, patient_id), so it is reproducible and
    independent of input row order.
  * A `calib` split is reserved for split-conformal calibration (G1/G2).
  * If the CXR-LT release ships an official split, honor it and only *carve* `calib` out of
    the official train (`calib_from_train`), leaving val/test untouched.
"""
from __future__ import annotations
import hashlib
from collections import Counter, defaultdict


def _unit_hash(seed: int, key: str) -> float:
    """Deterministic pseudo-random value in [0, 1) for a patient key."""
    h = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def assign_patient_splits(records, fractions, seed=42,
                          patient_key="patient_id", split_key="split"):
    """Assign each record a split by whole-patient hashing.

    records    : list[dict], each with `patient_key`.
    fractions  : dict like {"train":0.7,"calib":0.1,"val":0.1,"test":0.1} (order preserved).
    Returns    : dict patient_id -> split (records are also annotated in place).
    """
    names = list(fractions.keys())
    total = float(sum(fractions.values()))
    # cumulative boundaries in [0,1]
    bounds, acc = [], 0.0
    for n in names:
        acc += fractions[n] / total
        bounds.append(acc)
    bounds[-1] = 1.0  # guard against fp drift

    patients = sorted({str(r[patient_key]) for r in records})
    patient_split = {}
    for p in patients:
        u = _unit_hash(seed, p)
        for n, b in zip(names, bounds):
            if u < b:
                patient_split[p] = n
                break
        else:
            patient_split[p] = names[-1]

    for r in records:
        r[split_key] = patient_split[str(r[patient_key])]
    return patient_split


def carve_calibration(records, calib_frac=0.125, seed=42, from_split="train",
                      calib_name="calib", patient_key="patient_id", split_key="split"):
    """Move a deterministic `calib_frac` of `from_split` patients into `calib_name`.

    Use when honoring an official train/val/test split but still needing a conformal
    calibration set. `calib_frac` is a fraction of the *from_split* patients.
    """
    src_patients = sorted({str(r[patient_key]) for r in records
                           if r.get(split_key) == from_split})
    moved = {p for p in src_patients if _unit_hash(seed + 1, p) < calib_frac}
    for r in records:
        if r.get(split_key) == from_split and str(r[patient_key]) in moved:
            r[split_key] = calib_name
    return moved


def verify_no_leakage(records, patient_key="patient_id", split_key="split") -> bool:
    """Assert every patient appears in exactly one split. Raises on leakage."""
    seen = defaultdict(set)
    for r in records:
        seen[str(r[patient_key])].add(r.get(split_key))
    leaked = {p: sorted(s) for p, s in seen.items() if len(s) > 1}
    if leaked:
        examples = list(leaked.items())[:5]
        raise AssertionError(
            f"patient leakage across splits for {len(leaked)} patient(s); e.g. {examples}")
    return True


def unanchored_test_ids(records, unseen_keys, split_key="split", test_splits=("test",),
                        id_key="image_id"):
    """IDs of test-split images positive for any unseen (Task-2) concept — the held-out
    *unanchored* evaluation set. `unseen_keys` are the label-column names of the 6 CXR-LT
    open-world classes. A record is unanchored-positive if any unseen column == 1.
    """
    out = []
    tset = set(test_splits)
    for r in records:
        if r.get(split_key) in tset and any(_truthy(r.get(k)) for k in unseen_keys):
            out.append(r[id_key])
    return out


def _truthy(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() in ("1", "1.0", "true", "True", "yes")
    try:
        return float(v) >= 0.5
    except (TypeError, ValueError):
        return False


def split_summary(records, split_key="split", patient_key="patient_id"):
    """Return {split: {"images": n, "patients": m}} for logging."""
    img = Counter(r.get(split_key) for r in records)
    pat = defaultdict(set)
    for r in records:
        pat[r.get(split_key)].add(str(r[patient_key]))
    return {s: {"images": img[s], "patients": len(pat[s])} for s in sorted(img)}
