"""[analysis] G2 transfer with a stronger abnormality classifier: a one-hidden-layer network (256 units)
on the same frozen embeddings, in place of the Normal probe.

Same protocol as s35/s36 for G2: the network is trained on NIH train; Platt scaling and the decision
threshold tau are fit on NIH validation; lambda on NIH calibration (one image per patient); at VinDr and
Indiana tau and lambda on the two fixed calibration halves. Reports abnormality AUROC per site, the
primary-split G2 matrix with bootstrap intervals, 200 calibration/test redraws with the network fixed,
and 20 re-splits with the network retrained.

    python -m scripts.s39_nonlinear_g2 --config configs\\vindr.yaml ^
        --sites vindr=configs\\vindr.yaml iuxray=configs\\iuxray_nihtax.yaml --seeds 20 --redraws 200
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ovcbmr.config import load_config
from ovcbmr.eval.metrics import auroc
from scripts.s31_clean_protocol import _split, _take, _halves, _tau, _lam, _g2, one_per_patient_mask
from scripts.s35_multisite import _pairs
from scripts.s36_resplits import _sub, _cat, _summ, _rates
import scripts.s33_multiseed as M


def _abnormal(Y, nidx):
    return (Y[:, nidx] != 1).astype(int)


def _fit_net(E, y, seed, hidden):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=(hidden,), early_stopping=True, validation_fraction=0.1,
                         max_iter=200, random_state=seed).fit(E, y)


def _platt(net, E, y):
    """Platt scaling of the network's logit on a held-out split; returns a scoring function."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    def logit(E_):
        p = np.clip(net.predict_proba(E_)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(E), y)
    return lambda E_: lr.predict_proba(logit(E_))[:, 1]


def _D(score, E, y, pat):
    return {"A": score(E), "y": y, "pat": list(pat)}


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vindr.yaml")
    ap.add_argument("--sites", nargs="+", default=["vindr=configs/vindr.yaml", "iuxray=configs/iuxray_nihtax.yaml"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--redraws", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--out", default="s39_nonlinear_g2.json")
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config)
    ch = cfg.cross_hospital
    sites = dict(s.split("=", 1) for s in args.sites)
    a2 = float(getattr(cfg.conformal, "alpha_risk", 0.10))
    seed = int(getattr(cfg, "seed", 0))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        nidx = json.load(f)["vocabulary"].index("Normal")

    # ---------------- primary split
    sp = {k: _split(ch.nih_embeddings, ch.nih_processed, k) for k in ("train", "val", "calib", "test")}
    for k in ("calib", "test"):
        sp[k] = _take(sp[k], one_per_patient_mask(sp[k]["pat"]))
    print(f"[s39] training the network on {len(sp['train']['pat'])} NIH training images ...")
    net = _fit_net(sp["train"]["E"], _abnormal(sp["train"]["Y"], nidx), seed, args.hidden)
    score = _platt(net, sp["val"]["E"], _abnormal(sp["val"]["Y"], nidx))
    S = {"nih": {"tune": _D(score, sp["val"]["E"], _abnormal(sp["val"]["Y"], nidx), sp["val"]["pat"]),
                 "fit": _D(score, sp["calib"]["E"], _abnormal(sp["calib"]["Y"], nidx), sp["calib"]["pat"]),
                 "te": _D(score, sp["test"]["E"], _abnormal(sp["test"]["Y"], nidx), sp["test"]["pat"])}}
    for name, path in sites.items():
        sc = load_config(path)
        cal = _split(sc.paths.embeddings, sc.paths.processed, "calib")
        te = _split(sc.paths.embeddings, sc.paths.processed, "test")
        tune, fit = _halves(cal, f"{name}-clean")
        S[name] = {k: _D(score, d["E"], _abnormal(d["Y"], nidx), d["pat"]) for k, d in
                   (("tune", tune), ("fit", fit), ("te", te), ("cal", cal))}
    names = list(S)
    for s in names:
        S[s]["tau"] = _tau(S[s]["tune"])
        S[s]["lam"] = _lam(S[s]["fit"], S[s]["tau"], a2, "symmetric")
    out = {"alpha_risk": a2, "hidden": args.hidden, "per_site": {}, "primary": {}}
    print("\n=== per site: abnormality AUROC of the network (test) ===")
    for s in names:
        T = S[s]["te"]
        out["per_site"][s] = {"abn_auroc": auroc(T["A"].tolist(), T["y"].tolist()), "tau": S[s]["tau"], "lam": S[s]["lam"]}
        print(f"  {s:<8} AUROC {out['per_site'][s]['abn_auroc']:.3f}  tau {S[s]['tau']:.3f}  lam {S[s]['lam']:.3f}")
    print(f"\n=== primary split, G2 (target {a2}) ===")
    for c, d in _pairs(names):
        g = _g2(S[d]["te"], S[c]["tau"], S[c]["lam"], "symmetric", a2, args.nboot, seed)
        out["primary"][f"{c}->{d}"] = g
        print(f"  {c+' -> '+d:<18} acc {g['accept']:.2f}  cer {g['controlled']:.3f} [{g['ci'][0]:.3f},{g['ci'][1]:.3f}]"
              f"  err|acc {g['err_given_accept']:.3f}  miss|acc {g['miss_given_accept']:.3f}")

    # ---------------- redraws, network fixed
    rng = np.random.RandomState(seed)
    pools = {s: _cat(S[s]["fit"], S[s]["te"]) if s == "nih" else _cat(S[s]["cal"], S[s]["te"]) for s in names}
    ncal = {s: len(S[s]["fit"]["y"]) if s == "nih" else len(S[s]["cal"]["y"]) for s in names}
    rows = []
    for _ in range(args.redraws):
        fits, tests = {}, {}
        for s in names:
            P = pools[s]
            p = rng.permutation(len(P["y"]))
            cal, te = _sub(P, p[: ncal[s]]), _sub(P, p[ncal[s]:])
            if s == "nih":
                tune, fit = S["nih"]["tune"], cal
            else:
                h = rng.permutation(len(cal["y"]))
                tune, fit = _sub(cal, h[: len(h) // 2]), _sub(cal, h[len(h) // 2:])
            tau = _tau(tune)
            fits[s] = (tau, _lam(fit, tau, a2, "symmetric"))
            tests[s] = te
        row = {}
        for c, d in _pairs(names):
            g = _rates(tests[d], *fits[c])
            row[f"acc|{c}->{d}"], row[f"cer|{c}->{d}"] = g["accept"], g["cer"]
        rows.append(row)
    out["redraws"] = {k: _summ([r[k] for r in rows], a2 if k.startswith("cer") else None) for k in rows[0]}

    # ---------------- re-splits, network retrained
    En, Yn, Pn = M._pool(ch.nih_embeddings, ch.nih_processed, ("train", "calib", "val", "test"))
    yn = _abnormal(Yn, nidx)
    site_pools = {}
    for name, path in sites.items():
        sc = load_config(path)
        Ev, Yv, Pv = M._pool(sc.paths.embeddings, sc.paths.processed, ("calib", "test"))
        site_pools[name] = (Ev, _abnormal(Yv, nidx), Pv)
    rs = []
    for sd in range(args.seeds):
        a = M._assign(Pn, sd)
        m = {k: a == k for k in ("train", "calib", "val", "test")}
        net_s = _fit_net(En[m["train"]], yn[m["train"]], sd, args.hidden)
        sc_s = _platt(net_s, En[m["val"]], yn[m["val"]])
        def nih_part(k):
            idx = np.where(m[k])[0]
            keep = idx[one_per_patient_mask([Pn[i] for i in idx])]
            return _D(sc_s, En[keep], yn[keep], [Pn[i] for i in keep])
        R = {"nih": {"tune": _D(sc_s, En[m["val"]], yn[m["val"]], [Pn[i] for i in np.where(m["val"])[0]]),
                     "fit": nih_part("calib"), "te": nih_part("test")}}
        for name, (Ev, yv, Pv) in site_pools.items():
            vcal = np.array([M._unit_hash(sd, p) < 0.5 for p in Pv])
            vtun = np.array([M._unit_hash(sd + 1000, p) < 0.5 for p in Pv])
            R[name] = {"te": _D(sc_s, Ev[~vcal], yv[~vcal], [p for p, v in zip(Pv, vcal) if not v]),
                       "tune": _D(sc_s, Ev[vcal & vtun], yv[vcal & vtun], []),
                       "fit": _D(sc_s, Ev[vcal & ~vtun], yv[vcal & ~vtun], [])}
        for s in R:
            R[s]["tau"] = _tau(R[s]["tune"])
            R[s]["lam"] = _lam(R[s]["fit"], R[s]["tau"], a2, "symmetric")
        row = {"seed": sd}
        for c, d in _pairs(list(R)):
            g = _rates(R[d]["te"], R[c]["tau"], R[c]["lam"])
            row[f"acc|{c}->{d}"], row[f"cer|{c}->{d}"] = g["accept"], g["cer"]
        row["auroc_vindr"] = auroc(R["vindr"]["te"]["A"].tolist(), R["vindr"]["te"]["y"].tolist()) if "vindr" in R else None
        rs.append(row)
        print(f"[s39] re-split {sd}: " + "  ".join(f"{k.split('|')[1]} {row[k]:.3f}" for k in row if k.startswith("cer|")))
    out["resplits"] = {k: _summ([r[k] for r in rs], a2 if k.startswith("cer") else None)
                       for k in rs[0] if k not in ("seed",) and rs[0][k] is not None}
    print(f"\n=== {args.redraws} redraws (network fixed) and {args.seeds} re-splits (network retrained): G2 mean, share above ===")
    for c, d in _pairs(names):
        r1, r2 = out["redraws"][f"cer|{c}->{d}"], out["resplits"][f"cer|{c}->{d}"]
        a1, a2s = out["redraws"][f"acc|{c}->{d}"], out["resplits"][f"acc|{c}->{d}"]
        print(f"  {c+' -> '+d:<18} redraws cer {r1['mean']:.3f} above {r1['frac_above']:.2f} acc {a1['mean']:.2f} | "
              f"re-splits cer {r2['mean']:.3f} ({r2['sd']:.3f}) above {r2['frac_above']:.2f} acc {a2s['mean']:.2f}")
    os.makedirs(cfg.paths.results, exist_ok=True)
    op = os.path.join(cfg.paths.results, args.out)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[s39] wrote {op}")


if __name__ == "__main__":
    main()
