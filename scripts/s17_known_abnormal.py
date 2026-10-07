"""[G1 denominator + OOV] The G1 false-flag rate is defined on KNOWN-abnormal cases. Are
those "any-abnormal" (Normal=0 -- INCLUDES findings outside our 8-concept vocabulary) or
"in-vocabulary-abnormal" (>=1 of the 8 concepts labelled present)? Flagging an OOV-only case
(abnormal, but its finding is not in our vocabulary) is CORRECT -- it is genuinely unanchored
-- so counting it as a false flag inflates the rate. This script recomputes cross-hospital
coverage under BOTH denominators (with patient-clustered CIs), reports the NIH/VinDr
composition (normal / in-vocab / OOV-only), the CORRECT-flag rate on OOV cases (the valve
doing its job), and a data-driven VinDr class -> concept mapping table for the supplement.
Cached embeddings, no GPU.

    python -m scripts.s17_known_abnormal --config configs/vindr.yaml \
        --boxes data/raw/vindr/vinbigdata/train.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from collections import Counter

from ovcbmr.config import load_config
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.concept.valve import image_unexplained
from ovcbmr.ood.conformal import split_conformal_threshold, empirical_flag_rate
from ovcbmr.eval.metrics import bootstrap_ci

# mirrors build_manifest_vindr.VINDR2OURS (+ No finding -> Normal)
VINDR2OURS = {
    "Atelectasis": ["Atelectasis"], "Cardiomegaly": ["Cardiomegaly"],
    "Pleural effusion": ["Pleural Effusion"], "Infiltration": ["Infiltration"],
    "Nodule/Mass": ["Nodule", "Mass"], "Consolidation": ["Consolidation"],
    "Pleural thickening": ["Pleural Thickening"], "No finding": ["Normal"],
}


def _scored(emb_dir, proc, split, heads, vocab, nidx, ab_cols):
    """[(s, patient, cls)] for a hospital/split; cls in {normal, invocab, oov}."""
    import torch
    p = os.path.join(emb_dir, f"{split}_img.pt")
    if not os.path.exists(p):
        return []
    d = torch.load(p, weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    Pm = predict_matrix(d["emb"].numpy(), heads, vocab)
    out = []
    for i, iid in enumerate(d["ids"]):
        it = items.get(iid)
        if it is None:
            continue
        lab = it["labels_known"]
        if int(lab[nidx]) == 1:
            cls = "normal"
        elif any(int(lab[j]) == 1 for j in ab_cols):
            cls = "invocab"
        else:
            cls = "oov"
        s = image_unexplained(Pm[i].tolist(), vocab, "Normal")[1]
        out.append((s, it.get("patient_id", iid), cls))
    return out


def _sel(rows, keep):
    return [r for r in rows if r[2] in keep]


def _flag_ci(rows, q, n_boot, seed):
    if not rows or q == float("inf"):
        return (float("nan"), float("nan"), float("nan"))
    vals = [1.0 if s > q else 0.0 for s, _, _ in rows]
    pats = [p for _, p, _ in rows]
    return bootstrap_ci(vals, groups=pats, n_boot=n_boot, seed=seed)


def _compose(rows):
    c = Counter(r[2] for r in rows)
    return c["normal"], c["invocab"], c["oov"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vindr.yaml")
    ap.add_argument("--boxes", default="data/raw/vindr/vinbigdata/train.csv",
                    help="VinDr box CSV (for the class->concept mapping table); optional")
    args = ap.parse_args()
    cfg = load_config(args.config)

    ch = cfg.cross_hospital
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    nidx = vocab.index("Normal")
    ab_cols = [j for j, c in enumerate(vocab) if c != "Normal"]
    alpha = float(getattr(cfg.conformal, "alpha_flag", 0.05))
    n_boot = int(getattr(getattr(cfg, "eval", object()), "bootstrap", 1000))
    seed = int(getattr(cfg, "seed", 0))

    nih_cal = _scored(ch.nih_embeddings, ch.nih_processed, "calib", heads, vocab, nidx, ab_cols)
    vin_cal = _scored(cfg.paths.embeddings, cfg.paths.processed, "calib", heads, vocab, nidx, ab_cols)
    vin_te = _scored(cfg.paths.embeddings, cfg.paths.processed, "test", heads, vocab, nidx, ab_cols)

    # ---- composition ----
    print(f"[s17] alpha={alpha} | boot={n_boot} (patient-clustered)\n")
    print(f"{'pool':<18}{'normal':>8}{'in-vocab':>10}{'OOV-only':>10}")
    for nm, rows in (("NIH calib", nih_cal), ("VinDr calib", vin_cal), ("VinDr test", vin_te)):
        n, iv, oov = _compose(rows)
        print(f"{nm:<18}{n:>8}{iv:>10}{oov:>10}")

    # ---- coverage under both denominators ----
    print(f"\n{'denominator':<20}{'naive (cal NIH) [CI]':>28}{'group-cond (cal VinDr) [CI]':>30}")
    res = {}
    for label, keep in (("any-abnormal", {"invocab", "oov"}), ("in-vocab-abnormal", {"invocab"})):
        ncal, vcal, vte = _sel(nih_cal, keep), _sel(vin_cal, keep), _sel(vin_te, keep)
        q_naive = split_conformal_threshold([s for s, _, _ in ncal], alpha)
        q_group = split_conformal_threshold([s for s, _, _ in vcal], alpha)
        rn, lon, hin = _flag_ci(vte, q_naive, n_boot, seed)
        rg, log, hig = _flag_ci(vte, q_group, n_boot, seed)
        print(f"{label:<20}{rn:>16.3f} [{lon:.3f},{hin:.3f}]{rg:>18.3f} [{log:.3f},{hig:.3f}]")
        res[label] = {"n_vindr_test": len(vte),
                      "naive": {"rate": rn, "ci": [lon, hin], "q": q_naive},
                      "groupcond": {"rate": rg, "ci": [log, hig], "q": q_group}}

    # ---- OOV correct-flag rate (valve doing its job) at the in-vocab-calibrated q_hat ----
    q_ref = split_conformal_threshold([s for s, _, _ in _sel(nih_cal, {"invocab"})], alpha)
    oov_rows = _sel(vin_te, {"oov"})
    oov_flag = empirical_flag_rate([s for s, _, _ in oov_rows], q_ref) if oov_rows else float("nan")
    print(f"\n[s17] valve flags {oov_flag:.3f} of VinDr OOV-abnormal cases (n={len(oov_rows)}) as unanchored")
    print(f"       -- these are CORRECT unanchored detections (finding outside the vocabulary), NOT false flags.")
    res["oov_correct_flag_rate"] = {"rate": oov_flag, "n": len(oov_rows), "q_ref": q_ref}

    # ---- verdict ----
    a, b = res["any-abnormal"]["naive"], res["in-vocab-abnormal"]["naive"]
    print("\n[s17] VERDICT (naive NIH->VinDr false-flag, denominator effect):")
    print(f"  any-abnormal    : {a['rate']:.3f} [{a['ci'][0]:.3f},{a['ci'][1]:.3f}]  (current paper number)")
    print(f"  in-vocab only   : {b['rate']:.3f} [{b['ci'][0]:.3f},{b['ci'][1]:.3f}]  (correct G1 denominator)")
    if b["rate"] > alpha + 1e-9 and b["ci"][0] > alpha:
        print(f"  -> miscoverage SURVIVES the correct denominator (CI still above {alpha}); headline robust.")
    elif b["rate"] > alpha + 1e-9:
        print(f"  -> still above {alpha} but CI now touches it; report the softened significance.")
    else:
        print(f"  -> drops to/below {alpha} under the correct denominator; the 'any-abnormal' number was "
              f"inflated by correct OOV detections. REFRAME the headline honestly.")

    # ---- data-driven VinDr class -> concept mapping table (supplement) ----
    if args.boxes and os.path.exists(args.boxes):
        cnt = Counter()
        with open(args.boxes, newline="") as f:
            for r in csv.DictReader(f):
                cnt[str(r.get("class_name", "")).strip()] += 1
        print(f"\n[s17] VinDr class -> concept mapping ({args.boxes}):")
        print(f"  {'VinDr class':<24}{'mapped to':<28}{'box rows':>9}")
        maprows = []
        for cn, n in cnt.most_common():
            tgt = VINDR2OURS.get(cn)
            m = "+".join(tgt) if tgt else ("OOV (abnormal, unanchored)")
            print(f"  {cn:<24}{m:<28}{n:>9}")
            maprows.append({"vindr_class": cn, "mapped_to": m, "box_rows": n})
        res["vindr_class_map"] = maprows
    else:
        print(f"\n[s17] (--boxes {args.boxes} not found; skipping class-mapping table)")

    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, "s17_known_abnormal.json")
    with open(op, "w") as f:
        json.dump({"alpha": alpha,
                   "composition": {nm: dict(zip(("normal", "invocab", "oov"), _compose(r)))
                                   for nm, r in (("nih_calib", nih_cal), ("vindr_calib", vin_cal),
                                                 ("vindr_test", vin_te))},
                   **res}, f, indent=2)
    print(f"\n[s17] wrote {op}")


if __name__ == "__main__":
    main()
