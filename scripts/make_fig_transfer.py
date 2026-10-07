"""[figure] What transfers between hospitals, shown per patient-level split.

Three panels over the same four settings (on-site at each hospital and both transfer
directions): the G1 false-deferral rate, the G2 confident-error rate, and the G2 acceptance
rate. Open circles are the five re-split seeds of s33 (everything after the encoder refit per
seed); the filled marker is the primary seed-42 split from s31 under the same clean protocol,
with its clustered bootstrap interval. Dashed lines are the targets. Reads result JSONs only.

    python scripts/make_fig_transfer.py --results experiments/results --out docs/fig_transfer.png
"""
from __future__ import annotations
import argparse
import json
import os

SETTINGS = [("NIH on-site", "nih_onsite", "NIH\non-site"),
            ("NIH->VinDr", "nih_to_vindr", "NIH$\\to$VinDr\ntransfer"),
            ("VinDr on-site", "vindr_onsite", "VinDr\non-site"),
            ("VinDr->NIH", "vindr_to_nih", "VinDr$\\to$NIH\ntransfer")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--out", default="docs/fig_transfer.png")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ms = json.load(open(os.path.join(a.results, "s33_multiseed.json")))
    cp = json.load(open(os.path.join(a.results, "s31_clean_protocol.json")))["protocols"]["clean"]
    seeds = ms["per_seed"]

    panels = [
        ("G1 false-deferral rate (target 0.05)", "g1_{}",
         lambda k: (cp["g1"][k]["ffr"], cp["g1"][k]["ci"]), 0.05, (0.0, 0.10)),
        ("G2 confident-error rate (target 0.10)", "g2_conferr_{}",
         lambda k: (cp["g2"]["symmetric"][k]["controlled"], cp["g2"]["symmetric"][k]["ci"]), 0.10, (0.0, 0.22)),
        ("G2 acceptance rate", "g2_accept_{}",
         lambda k: (cp["g2"]["symmetric"][k]["accept"], None), None, (0.0, 0.85)),
    ]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.0))
    grey, blue, red = "#6b6b6b", "#1f4e9c", "#c0392b"
    for ax, (title, key, primary, target, ylim) in zip(axes, panels):
        for i, (jkey, skey, label) in enumerate(SETTINGS):
            if i in (1, 3):
                ax.axvspan(i - 0.45, i + 0.45, color="#f0f0f0", zorder=0)
            vals = [r[key.format(skey)] for r in seeds]
            xs = [i - 0.20 + 0.08 * j for j in range(len(vals))]
            ax.scatter(xs, vals, s=34, facecolors="none", edgecolors=grey, linewidths=1.2, zorder=3)
            m = sum(vals) / len(vals)
            ax.plot([i - 0.28, i + 0.20], [m, m], color=grey, lw=1.6, zorder=2)
            v, ci = primary(jkey)
            if ci is not None:
                ax.errorbar([i + 0.32], [v], yerr=[[v - ci[0]], [ci[1] - v]], fmt="D", ms=6, color=blue,
                            capsize=3, lw=1.3, zorder=4)
            else:
                ax.scatter([i + 0.32], [v], marker="D", s=40, color=blue, zorder=4)
        if target is not None:
            ax.axhline(target, color=red, ls="--", lw=1.2, zorder=1)
        ax.set_xticks(range(len(SETTINGS)))
        ax.set_xticklabels([s[2] for s in SETTINGS], fontsize=9.5)
        ax.set_xlim(-0.55, len(SETTINGS) - 0.45)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=11)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="none", mfc="none", mec=grey, mew=1.2, ms=6, label="five re-splits (seeds 0 to 4)"),
        Line2D([], [], color=grey, lw=1.6, label="mean of the five"),
        Line2D([], [], marker="D", ls="none", color=blue, ms=6, label="primary split (seed 42), 95% interval"),
        Line2D([], [], color=red, ls="--", lw=1.2, label="target"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300)
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
