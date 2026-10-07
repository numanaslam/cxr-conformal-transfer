"""Smoke test: data-pipeline logic + method cores on synthetic data, no torch/numpy/data.

    pytest -q tests/test_smoke.py
    # or, without pytest:
    python tests/test_smoke.py
"""
from __future__ import annotations
import math
import os
import sys

# make the repo root importable whether run via pytest or directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ovcbmr.io import splits as S
from ovcbmr.concept import bank as B
from ovcbmr.concept.valve import anchoring, unexplained_statistic, decide, UNANCHORED, ANCHORED
from ovcbmr.concept.calibrate import sigmoid, apply_platt
from ovcbmr.ood.conformal import (split_conformal_threshold, empirical_flag_rate,
                                   crc_threshold, coverage_vs_alpha)
from ovcbmr.eval.metrics import auroc, fpr_at_tpr, aurc, bootstrap_ci


def _records(n_patients=120, per_patient=2):
    recs = []
    for p in range(n_patients):
        for j in range(per_patient):
            recs.append({"image_id": f"img_{p:03d}_{j}", "patient_id": f"pat_{p:03d}"})
    return recs


def test_patient_splits_no_leakage():
    recs = _records()
    fractions = {"train": 0.70, "calib": 0.10, "val": 0.10, "test": 0.10}
    S.assign_patient_splits(recs, fractions, seed=42)
    S.verify_no_leakage(recs)                       # raises on leakage
    summary = S.split_summary(recs)
    assert set(summary) == {"train", "calib", "val", "test"}
    for s in summary.values():
        assert s["images"] > 0 and s["patients"] > 0
    # a patient's images must all share one split (spot check)
    seen = {}
    for r in recs:
        seen.setdefault(r["patient_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in seen.values())


def test_carve_calibration_and_unanchored():
    # patient-consistent split: patients p00..p03 -> test, p04..p19 -> train
    recs = []
    for i in range(60):
        pid = i % 20
        split = "test" if pid < 4 else "train"
        unseen = "1" if (split == "test" and i % 2 == 0) else "0"
        recs.append({"image_id": f"i{i}", "patient_id": f"p{pid:02d}",
                     "split": split, "unseen1": unseen})
    moved = S.carve_calibration(recs, calib_frac=0.4, seed=7, from_split="train")
    assert 0 < len(moved) <= 16                     # 16 train patients available
    S.verify_no_leakage(recs)                       # whole patients moved -> still no leakage
    un = S.unanchored_test_ids(recs, unseen_keys=["unseen1"], id_key="image_id")
    assert len(un) == 6 and all(iid.startswith("i") for iid in un)


def test_concept_bank():
    bank = B.load_bank("default")
    vocab = B.active_vocabulary(bank)
    assert len(vocab) == 30 and "Normal" in vocab
    assert len(B.concept_names(bank, "unseen")) == 6
    pos = B.prompt_ensemble("Cardiomegaly", bank)
    neg = B.negative_prompts("Cardiomegaly", bank)
    assert len(pos) >= 5 and len(neg) >= 3 and "cardiomegaly" in pos[0]


def test_valve_decision():
    # well-anchored: high max anchoring -> low unexplained -> ANCHORED
    a = anchoring([0.95, 0.2, 0.1])
    s_anch = unexplained_statistic(0.9, a)
    assert decide(s_anch, q_hat=0.1) == ANCHORED
    # unanchored: abnormal but nothing anchors -> high unexplained -> UNANCHORED
    s_un = unexplained_statistic(0.9, anchoring([0.1, 0.05, 0.08]))
    assert decide(s_un, q_hat=0.1) == UNANCHORED


def test_conformal_g1_guarantee():
    cal = [i / 100.0 for i in range(100)]            # 0.00 .. 0.99
    q = split_conformal_threshold(cal, alpha=0.05)
    assert abs(q - 0.95) < 1e-9
    assert empirical_flag_rate(cal, q) <= 0.05 + 1e-9   # G1 holds on calibration
    assert split_conformal_threshold(cal, alpha=0.001) == float("inf")
    rows = coverage_vs_alpha(cal, cal, [0.05, 0.1, 0.2])
    assert len(rows) == 3 and all(r[2] <= r[0] + 1e-9 for r in rows)


def test_conformal_g2_crc():
    lambdas = [0.1, 0.2, 0.3]
    risks = [0.30, 0.12, 0.05]
    lam, idx = crc_threshold(risks, lambdas, n=100, alpha=0.10, B=1.0)
    assert idx == 2 and abs(lam - 0.3) < 1e-9


def test_metrics():
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1, 1]
    assert abs(auroc(scores, labels) - 1.0) < 1e-9
    assert fpr_at_tpr(scores, labels, tpr=0.95) == 0.0
    losses = [0, 0, 0, 1, 1, 1]
    conf = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    a = aurc(losses, conf)
    assert math.isfinite(a) and 0.0 <= a <= 1.0
    mean, lo, hi = bootstrap_ci([0, 0, 0, 1, 1, 1], n_boot=200, seed=0)
    assert lo <= mean <= hi and 0.0 <= lo and hi <= 1.0
    assert abs(sigmoid(0.0) - 0.5) < 1e-12 and abs(apply_platt(0.0, 1.0, 0.0) - 0.5) < 1e-12


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"ALL {len(fns)} SMOKE TESTS PASSED")


if __name__ == "__main__":
    _run_all()
