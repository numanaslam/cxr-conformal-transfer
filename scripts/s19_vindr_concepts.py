"""[concept transfer] Do the NIH-trained concept heads still work on VinDr? Per-concept AUROC
and ECE of the SHARED (NIH-trained, Platt-calibrated) heads on NIH test vs VinDr test. If
AUROC holds cross-hospital, the frozen model transfers and only the conformal wrapper needs
site calibration; if it drops, that is itself an important finding. ECE gap shows whether the
NIH Platt scalars still calibrate on VinDr. Cached embeddings, no GPU.

    python -m scripts.s19_vindr_concepts --config configs/vindr.yaml
"""
from __future__ import annotations
import json
import os

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.eval.metrics import auroc, expected_calibration_error


def _pred_labels(emb_dir, proc, split, heads, vocab):
    import torch
    d = torch.load(os.path.join(emb_dir, f"{split}_img.pt"), weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    Pm = predict_matrix(d["emb"].numpy(), heads, vocab)
    P = {c: [] for c in vocab}
    Y = {c: [] for c in vocab}
    for i, iid in enumerate(d["ids"]):
        it = items.get(iid)
        if it is None:
            continue
        for j, c in enumerate(vocab):
            P[c].append(float(Pm[i][j]))
            Y[c].append(int(it["labels_known"][j]))
    return P, Y


def _auroc_ece(P, Y, c):
    y, p = Y[c], P[c]
    npos = sum(y)
    au = auroc(p, y) if 0 < npos < len(y) else float("nan")
    return au, expected_calibration_error(p, y), npos


def main():
    cfg = parse_config_arg()
    ch = cfg.cross_hospital
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]

    Pn, Yn = _pred_labels(ch.nih_embeddings, ch.nih_processed, "test", heads, vocab)
    Pv, Yv = _pred_labels(cfg.paths.embeddings, cfg.paths.processed, "test", heads, vocab)

    out = {}
    for c in vocab:
        an, en, _ = _auroc_ece(Pn, Yn, c)
        av, ev, nv = _auroc_ece(Pv, Yv, c)
        out[c] = {"nih_auroc": an, "vindr_auroc": av, "nih_ece": en, "vindr_ece": ev, "vindr_npos": nv}

    order = sorted(vocab, key=lambda c: (-(out[c]["nih_auroc"]) if out[c]["nih_auroc"] == out[c]["nih_auroc"] else 1))
    print(f"[s19] per-concept AUROC / ECE, NIH test vs VinDr test (shared NIH-trained heads)\n")
    print(f"{'concept':<20}{'NIH AUROC':>10}{'VinDr AUROC':>12}{'ΔAUROC':>9}{'NIH ECE':>9}{'VinDr ECE':>10}{'Vn+':>7}")
    for c in order:
        o = out[c]
        d_au = (o["vindr_auroc"] - o["nih_auroc"]) if o["vindr_auroc"] == o["vindr_auroc"] else float("nan")
        print(f"{c:<20}{o['nih_auroc']:>10.3f}{o['vindr_auroc']:>12.3f}{d_au:>9.3f}"
              f"{o['nih_ece']:>9.3f}{o['vindr_ece']:>10.3f}{o['vindr_npos']:>7}")

    abn = [c for c in vocab if c != "Normal"]
    def _mean(key, cs):
        vals = [out[c][key] for c in cs if out[c][key] == out[c][key]]
        return sum(vals) / len(vals) if vals else float("nan")
    vn_ok = [c for c in abn if out[c]["vindr_auroc"] == out[c]["vindr_auroc"]]
    print(f"\n[s19] mean abnormal AUROC : NIH={_mean('nih_auroc', abn):.3f}  VinDr={_mean('vindr_auroc', vn_ok):.3f}")
    print(f"[s19] mean abnormal ECE   : NIH={_mean('nih_ece', abn):.3f}  VinDr={_mean('vindr_ece', abn):.3f}")
    print("[s19] (VinDr AUROC on concepts with >=1 positive; Vn+ = VinDr positives)")

    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, "s19_vindr_concepts.json")
    with open(op, "w") as f:
        json.dump({"per_concept": out,
                   "mean_abnormal": {"nih_auroc": _mean("nih_auroc", abn), "vindr_auroc": _mean("vindr_auroc", vn_ok),
                                     "nih_ece": _mean("nih_ece", abn), "vindr_ece": _mean("vindr_ece", abn)}}, f, indent=2)
    print(f"\n[s19] wrote {op}")


if __name__ == "__main__":
    main()
