"""[revision 1] Transfer of both guarantees between three sites: NIH, VinDr-CXR and IU X-ray.

Generalises s31 (primary split, clean protocol) and s33 (re-splits) from one pair of sites to every
ordered pair of N sites, with one image per patient at NIH (as in s31/s33 --one-per-patient; VinDr
and IU X-ray already have one image per study). For each calibration site c and deployment site d:

    G1  false-deferral rate at d of the quantile learned at c (target alpha_flag)
    G2  acceptance and confident-error rate at d of (tau, lambda) learned at c (target alpha_risk)
    G2  one-sided accepted-and-missed rate (primary split only)

Per site: NIH  tau + Platt from val (all images), q_hat + lambda from calib (1/patient), test 1/patient
          other sites: calib halves -> tau / lambda, full calib -> q_hat, test -> rates
For VinDr the primary-split numbers equal those of s31 --one-per-patient and the re-split numbers
those of s33 --one-per-patient (same split rules and salts).

It also reports a measured shift between every pair of sites: the AUROC of a logistic domain
classifier on the frozen embeddings (held-out half), and the mean concept AUROC and ECE per site.

    python -m scripts.s35_multisite --config configs/vindr.yaml ^
        --sites vindr=configs/vindr.yaml iuxray=configs/iuxray.yaml --seeds 0 1 2 3 4
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ovcbmr.config import load_config
from ovcbmr.concept.heads import load_heads
from ovcbmr.eval.metrics import auroc
from scripts.s31_clean_protocol import (_split, _take, _halves, _refit_platt, _derive, _q, _ffr,
                                        _tau, _lam, _g2, _calib, one_per_patient_mask)
import scripts.s33_multiseed as M


def _pairs(names):
    return [(c, d) for c in names for d in names]


def _primary(cfg, sites, a1, a2, nb, seed):
    ch = cfg.cross_hospital
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    nidx = vocab.index("Normal")
    abn = [i for i, c in enumerate(vocab) if c != "Normal"]
    n_val = _split(ch.nih_embeddings, ch.nih_processed, "val")
    n_cal = _split(ch.nih_embeddings, ch.nih_processed, "calib")
    n_te = _split(ch.nih_embeddings, ch.nih_processed, "test")
    n_te_all = n_te
    n_cal = _take(n_cal, one_per_patient_mask(n_cal["pat"]))
    n_te = _take(n_te, one_per_patient_mask(n_te["pat"]))
    hd = _refit_platt(heads, vocab, n_val)
    dv = lambda D: _derive(D, hd, vocab, nidx, abn)   # noqa: E731

    S = {"nih": {"tune": dv(n_val), "fit": dv(n_cal), "q": dv(n_cal), "te": dv(n_te), "te_all": dv(n_te_all),
                 "E_te": n_te_all["E"]}}
    for name, path in sites.items():
        sc = load_config(path)
        cal = _split(sc.paths.embeddings, sc.paths.processed, "calib")
        te = _split(sc.paths.embeddings, sc.paths.processed, "test")
        if cal is None or te is None:
            raise SystemExit(f"[s35] {name}: missing calib/test embeddings under {sc.paths.embeddings}")
        tune, fit = _halves(cal, f"{name}-clean")
        S[name] = {"tune": dv(tune), "fit": dv(fit), "q": dv(cal), "te": dv(te), "te_all": dv(te), "E_te": te["E"]}
        print(f"[s35] {name}: calib {len(cal['pat'])} (halves {len(tune['pat'])}/{len(fit['pat'])}), test {len(te['pat'])}")

    names = list(S)
    for s in names:
        S[s]["qhat"] = _q(S[s]["q"], a1)
        S[s]["tau"] = _tau(S[s]["tune"])
        S[s]["lam"] = _lam(S[s]["fit"], S[s]["tau"], a2, "symmetric")
        S[s]["lam1"] = _lam(S[s]["fit"], S[s]["tau"], a2, "one-sided")
    out = {"sites": names, "per_site": {}, "pairs": {}}
    for s in names:
        T = S[s]["te_all"]
        auc = [auroc(T["P"][:, j].tolist(), T["Y"][:, j].tolist()) for j in abn
               if 0 < T["Y"][:, j].sum() < len(T["Y"])]
        ece, _ = _calib(T, [j for j in abn if 0 < T["Y"][:, j].sum() < len(T["Y"])])
        out["per_site"][s] = {"concept_auroc": float(sum(auc) / len(auc)), "concept_ece": ece,
                              "abn_auroc": auroc(T["A"].tolist(), T["y"].tolist()),
                              "n_test": int(len(S[s]["te"]["y"])), "n_inv_test": int(S[s]["te"]["known"].sum()),
                              "prevalence_abnormal": float(S[s]["te"]["y"].mean()),
                              "tau": S[s]["tau"], "lam": S[s]["lam"], "qhat": S[s]["qhat"]}
    for c, d in _pairs(names):
        Dt = S[d]["te"]
        g1 = _ffr(Dt, S[c]["qhat"], nb, seed)
        g2 = _g2(Dt, S[c]["tau"], S[c]["lam"], "symmetric", a2, nb, seed)
        g21 = _g2(Dt, S[c]["tau"], S[c]["lam1"], "one-sided", a2, nb, seed)
        out["pairs"][f"{c}->{d}"] = {"g1": g1, "g2": g2, "g2_onesided": g21}
    return out, S


def _domain_auc(S, names, seed, nmax=3000):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(seed)
    res = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            Ea, Eb = S[a]["E_te"], S[b]["E_te"]
            ia = rng.permutation(len(Ea))[:nmax]; ib = rng.permutation(len(Eb))[:nmax]
            X = np.concatenate([Ea[ia], Eb[ib]]); y = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
            p = rng.permutation(len(y)); h = len(y) // 2
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[p[:h]], y[p[:h]])
            res[f"{a}|{b}"] = auroc(clf.decision_function(X[p[h:]]).tolist(), y[p[h:]].tolist())
    return res


def _resplits(cfg, sites, seeds, a1, a2):
    import numpy as np
    from scripts.s31_clean_protocol import one_per_patient_mask as opm
    ch = cfg.cross_hospital
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    nidx = vocab.index("Normal")
    abn = [i for i, c in enumerate(vocab) if c != "Normal"]
    En, Yn, Pn = M._pool(ch.nih_embeddings, ch.nih_processed, ("train", "calib", "val", "test"))
    pools = {}
    for name, path in sites.items():
        sc = load_config(path)
        pools[name] = M._pool(sc.paths.embeddings, sc.paths.processed, ("calib", "test"))
    rows = []
    for seed in seeds:
        sp = M._assign(Pn, seed)
        m = {k: sp == k for k in ("train", "calib", "val", "test")}
        heads = M._train_heads(En[m["train"]], Yn[m["train"]], En[m["val"]], Yn[m["val"]], vocab)
        dv = lambda E, Y: M._derive(E, Y, heads, vocab, nidx, abn)   # noqa: E731
        sel = lambda D, k: {key: v[opm(Pn[m[k]].tolist())] for key, v in D.items()}   # noqa: E731
        N_val = dv(En[m["val"]], Yn[m["val"]])
        S = {"nih": {"tune": N_val, "fit": sel(dv(En[m["calib"]], Yn[m["calib"]]), "calib"),
                     "te": sel(dv(En[m["test"]], Yn[m["test"]]), "test")}}
        S["nih"]["q"] = S["nih"]["fit"]
        for name, (Ev, Yv, Pv) in pools.items():
            vcal = np.array([M._unit_hash(seed, p) < 0.5 for p in Pv])
            vtun = np.array([M._unit_hash(seed + 1000, p) < 0.5 for p in Pv])
            S[name] = {"q": dv(Ev[vcal], Yv[vcal]), "te": dv(Ev[~vcal], Yv[~vcal]),
                       "tune": dv(Ev[vcal & vtun], Yv[vcal & vtun]), "fit": dv(Ev[vcal & ~vtun], Yv[vcal & ~vtun])}
        for s in S:
            S[s]["qhat"] = M.split_conformal_threshold(S[s]["q"]["s"][S[s]["q"]["known"]].tolist(), a1)
            S[s]["tau"] = M._tau(S[s]["tune"])
            S[s]["lam"] = M._lam(S[s]["fit"], S[s]["tau"], a2)
        r = {"seed": seed}
        for c, d in _pairs(list(S)):
            r[f"g1|{c}->{d}"] = M._ffr(S[d]["te"], S[c]["qhat"])
            r[f"acc|{c}->{d}"], r[f"cer|{c}->{d}"] = M._g2(S[d]["te"], S[c]["tau"], S[c]["lam"])
        rows.append(r)
        print(f"[s35] seed {seed}: " + "  ".join(f"{k.split('|')[1]} G1 {r[k]:.3f}" for k in r if k.startswith("g1|")))
    keys = [k for k in rows[0] if k != "seed"]
    summ = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        summ[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                   "min": float(v.min()), "max": float(v.max())}
    return {"seeds": seeds, "per_seed": rows, "summary": summ}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vindr.yaml", help="supplies the NIH paths and alphas")
    ap.add_argument("--sites", nargs="+", default=["vindr=configs/vindr.yaml", "iuxray=configs/iuxray.yaml"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--out", default="s35_multisite.json", help="file name under cfg.paths.results")
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config)
    sites = dict(s.split("=", 1) for s in args.sites)
    a1 = float(getattr(cfg.conformal, "alpha_flag", 0.05))
    a2 = float(getattr(cfg.conformal, "alpha_risk", 0.10))
    seed = int(getattr(cfg, "seed", 0))

    prim, S = _primary(cfg, sites, a1, a2, args.nboot, seed)
    names = prim["sites"]
    print(f"\n=== primary split, clean protocol, one image per patient at NIH (G1 target {a1}, G2 target {a2}) ===")
    print(f"  {'calibrated -> deployed':<22}{'G1 [CI]':>24}{'accept':>8}{'conf-err [CI]':>24}{'err|acc':>9}{'1-sided':>9}")
    for c, d in _pairs(names):
        p = prim["pairs"][f"{c}->{d}"]
        g1, g2, g21 = p["g1"], p["g2"], p["g2_onesided"]
        print(f"  {c+' -> '+d:<22}{g1['ffr']:>8.3f} [{g1['ci'][0]:.3f},{g1['ci'][1]:.3f}]"
              f"{g2['accept']:>8.2f}{g2['controlled']:>8.3f} [{g2['ci'][0]:.3f},{g2['ci'][1]:.3f}]"
              f"{g2['err_given_accept']:>9.3f}{g21['controlled']:>9.3f}")
    print("\n=== per site (test): concept AUROC / ECE, abnormality AUROC, abnormal prevalence, n, n in-vocab ===")
    for s in names:
        q = prim["per_site"][s]
        print(f"  {s:<8} AUROC {q['concept_auroc']:.3f}  ECE {q['concept_ece']:.3f}  abn-AUROC {q['abn_auroc']:.3f}"
              f"  prev {q['prevalence_abnormal']:.2f}  n {q['n_test']}  n_inv {q['n_inv_test']}"
              f"  tau {q['tau']:.3f} lam {q['lam']:.3f}")
    prim["domain_auroc"] = _domain_auc(S, names, seed)
    print("\n=== domain-classifier AUROC between sites (frozen embeddings, held-out half) ===")
    for k, v in prim["domain_auroc"].items():
        print(f"  {k:<16} {v:.3f}")

    out = {"alpha_flag": a1, "alpha_risk": a2, "primary": prim}
    if args.seeds:
        rs = _resplits(cfg, sites, args.seeds, a1, a2)
        out["resplits"] = rs
        print(f"\n=== re-splits {args.seeds}: mean (sd) [min, max] ===")
        for k, v in rs["summary"].items():
            print(f"  {k:<28} {v['mean']:.3f} ({v['sd']:.3f}) [{v['min']:.3f}, {v['max']:.3f}]")
    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, args.out)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[s35] wrote {op}")


if __name__ == "__main__":
    main()
