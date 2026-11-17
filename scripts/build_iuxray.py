"""Build the Indiana University chest X-ray collection (IU X-ray, Open-i; CC BY-NC-ND 4.0) as a third
deployment site, in the same form as VinDr: a labels.csv mapped onto the NIH concept vocabulary and
a folder of resized images.

Inputs are the two CSV files of the widely used Kaggle copy ("chest-xrays-indiana-university"):
  indiana_reports.csv      uid, MeSH, Problems, image, indication, comparison, findings, impression
  indiana_projections.csv  uid, filename, projection (Frontal / Lateral)
and a folder of the PNG images, either from that copy (images_normalized/, names such as
1_IM-0001-4001.dcm.png) or from the Open-i archive NLMCXR_png.tgz (names such as
CXR1_1_IM-0001-4001.png). Both naming schemes are matched.

Each report (uid) is one study. We keep ONE frontal image per report, so calibration and test
scores are exchangeable, as for NIH and VinDr. Findings come from the MeSH field, which was coded
from the reports by human annotators. A study is Normal only when its MeSH field is "normal";
studies coded "No Indexing" are dropped.

    python scripts/build_iuxray.py --reports D:/data/iuxray/indiana_reports.csv ^
        --projections D:/data/iuxray/indiana_projections.csv --inspect
    python scripts/build_iuxray.py --reports D:/data/iuxray/indiana_reports.csv ^
        --projections D:/data/iuxray/indiana_projections.csv ^
        --images D:/data/iuxray/images/images_normalized --img-out D:/data/iuxray/images_256

Sensitivity run (--nih-taxonomy-only, with --out data/raw/iuxray_nihtax/labels.csv and
configs/iuxray_nihtax.yaml): abnormal studies whose MeSH findings all fall outside the NIH
taxonomy (for example calcified granuloma or degenerative spine) are dropped, because NIH would
have labelled many of them "No Finding".
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MeSH heading (first part of a term, lower case) -> our concept name(s). Checked with --inspect.
HEAD2OURS = {
    "pulmonary atelectasis": ["Atelectasis"], "atelectasis": ["Atelectasis"],
    "cardiomegaly": ["Cardiomegaly"],
    "consolidation": ["Consolidation"],
    "pleural effusion": ["Pleural Effusion"],
    "infiltrate": ["Infiltration"],
    "nodule": ["Nodule"], "multiple pulmonary nodules": ["Nodule"],
    "mass": ["Mass"],
    "pleural thickening": ["Pleural Thickening"],
    # NIH held-out findings (no trained head; out-of-vocabulary, as at the other sites)
    "pulmonary edema": ["Pulmonary Edema"], "pulmonary emphysema": ["Emphysema"], "emphysema": ["Emphysema"],
    "pulmonary fibrosis": ["Fibrosis"], "fibrosis": ["Fibrosis"], "hernia, hiatal": ["Hernia"],
    "pneumonia": ["Pneumonia"], "pneumothorax": ["Pneumothorax"],
}


def _terms(mesh):
    """'Cardiomegaly/mild;Thickening/pleura/apex' -> [['cardiomegaly','mild'], ['thickening','pleura','apex']]"""
    return [[p.strip().lower() for p in t.split("/")] for t in str(mesh or "").split(";") if t.strip()]


def _concepts(terms):
    out = set()
    for t in terms:
        out.update(HEAD2OURS.get(t[0], []))
        if t[0] == "thickening" and any("pleura" in q for q in t[1:]):
            out.add("Pleural Thickening")
    return out


def _key(name):
    """Normalise both naming schemes to the core id, e.g. '1_IM-0001-4001'."""
    b = os.path.basename(name)
    b = re.sub(r"\.png$", "", b, flags=re.I)
    b = re.sub(r"\.dcm$", "", b, flags=re.I)
    return re.sub(r"^CXR", "", b)


def _readable(path):
    """True if the PNG decodes completely (catches half-written files)."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:                                    # noqa: BLE001
        return False


def _h(seed, key):
    return hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True, help="indiana_reports.csv")
    ap.add_argument("--projections", required=True, help="indiana_projections.csv")
    ap.add_argument("--images", default=None, help="folder with the PNG images (either naming scheme)")
    ap.add_argument("--img-out", default="D:/data/iuxray/images_256")
    ap.add_argument("--out", default="data/raw/iuxray/labels.csv")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bank", default="data/concepts/nih_concepts.yaml")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--nih-taxonomy-only", action="store_true",
                    help="drop abnormal studies with no finding in the NIH 14-label taxonomy (sensitivity run)")
    args = ap.parse_args()

    with open(args.reports, encoding="utf-8", newline="") as f:
        reps = {r["uid"]: r for r in csv.DictReader(f)}
    with open(args.projections, encoding="utf-8", newline="") as f:
        projs = list(csv.DictReader(f))
    frontal = defaultdict(list)
    for p in projs:
        if str(p.get("projection", "")).strip().lower() == "frontal":
            frontal[p["uid"]].append(p["filename"])

    heads = Counter(t[0] for r in reps.values() for t in _terms(r.get("MeSH")))
    if args.inspect:
        print(f"[iu] reports {len(reps)}, projection rows {len(projs)}, reports with a frontal image {len(frontal)}")
        print("[iu] projections:", Counter(str(p.get('projection', '')).strip() for p in projs).most_common())
        print("[iu] 70 most frequent MeSH headings (mapped ones marked *):")
        for k, v in heads.most_common(70):
            print(f"   {'*' if k in HEAD2OURS or k == 'thickening' else ' '} {v:>6}  {k}")
        th = Counter("/".join(t) for r in reps.values() for t in _terms(r.get("MeSH")) if t[0] == "thickening")
        print("[iu] 'thickening' terms:", th.most_common(10))
        print("[iu] example filename:", projs[0].get("filename"))
        return

    from ovcbmr.concept.bank import load_bank, active_vocabulary, concept_names
    bank = load_bank(args.bank)
    known, unseen = active_vocabulary(bank), concept_names(bank, "unseen")
    allc = known + unseen
    missing = sorted({c for v in HEAD2OURS.values() for c in v} - set(allc))
    if missing:
        print(f"[iu] WARNING: mapped names not in the bank {args.bank}: {missing}")

    avail = {}
    if args.images:
        for fn in os.listdir(args.images):
            if fn.lower().endswith(".png"):
                avail[_key(fn)] = os.path.join(args.images, fn)
    os.makedirs(args.img_out, exist_ok=True)

    from PIL import Image
    header = ["image_id", "patient_id", "projection", "split", "site"] + allc
    rows, drop, n_fixed = [], Counter(), 0
    for uid, r in reps.items():
        terms = _terms(r.get("MeSH"))
        if not terms or terms[0][0] == "no indexing":
            drop["no indexing"] += 1; continue
        cands = [fn for fn in frontal.get(uid, []) if _key(fn) in avail]
        if not cands:
            drop["no frontal image found"] += 1; continue
        fn = min(cands, key=lambda x: _h(args.seed, _key(x)))
        iid = _key(fn) + ".png"
        dst = os.path.join(args.img_out, iid)
        if os.path.exists(dst) and not _readable(dst):
            os.remove(dst); n_fixed += 1                 # half-written earlier: rebuild it
        if not os.path.exists(dst):
            try:
                im = Image.open(avail[_key(fn)]).convert("L")
                w, h = im.size
                s = args.size / min(w, h)
                if s < 1.0:
                    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
                im.save(dst)
            except Exception as e:                       # noqa: BLE001
                print(f"[iu] unreadable source {avail[_key(fn)]}: {e}")
                if os.path.exists(dst):
                    os.remove(dst)
                drop["unreadable source image"] += 1; continue
        if not _readable(dst):
            os.remove(dst); drop["unreadable after conversion"] += 1; continue
        mapped = _concepts(terms)
        normal = len(terms) == 1 and terms[0][0] == "normal"
        if args.nih_taxonomy_only and not normal and not mapped:
            drop["no NIH-taxonomy finding"] += 1; continue
        rec = {"image_id": iid, "patient_id": f"iu{uid}", "projection": "PA", "split": "", "site": "iuxray"}
        for c in allc:
            rec[c] = 1 if (c in mapped or (c == "Normal" and normal)) else 0
        rows.append(rec)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader(); w.writerows(rows)
    n = max(len(rows), 1)
    nn = sum(r["Normal"] for r in rows)
    inv = sum(1 for r in rows if any(r[c] for c in known if c != "Normal"))
    print(f"[iu] wrote {len(rows)} studies (one frontal image each) -> {args.out}; dropped {dict(drop)}; "
          f"rebuilt {n_fixed} broken image(s)")
    print(f"[iu] Normal {nn} ({100*nn/n:.1f}%), in-vocabulary abnormal {inv}, other abnormal {len(rows)-nn-inv}")
    print("[iu] positives: " + ", ".join(f"{c} {sum(r[c] for r in rows)}" for c in known if c != "Normal"))
    print("[iu] next: s01 and s02 with configs/iuxray.yaml")


if __name__ == "__main__":
    main()
