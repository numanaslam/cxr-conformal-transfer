"""[figure] What transfers between the three sites, shown per split (revision 1, Figure 2).

Three stacked panels over the nine calibrated -> deployed settings of s35, grouped by the site
of deployment (on-site first, then the two transferred thresholds, shaded): the G1
false-deferral rate, the G2 confident-error rate, and the G2 acceptance rate. Open circles are
the five re-splits, the grey bar is their mean, and the filled marker is the primary split with
its clustered bootstrap interval. Dashed lines are the targets. Reads s35 result JSON only.

    python scripts/make_fig_multisite.py --json experiments/results/s35_multisite_nihtax.json \
        --out revision1/fig_transfer.png
"""
from __future__ import annotations
import argparse
import json
import os

NAME = {"nih": "NIH", "vindr": "VinDr", "iuxray": "Indiana"}
ORDER = [("nih", "nih"), ("vindr", "nih"), ("iuxray", "nih"),
         ("vindr", "vindr"), ("nih", "vindr"), ("iuxray", "vindr"),
         ("iuxray", "iuxray"), ("nih", "iuxray"), ("vindr", "iuxray")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="experiments/results/s35_multisite_nihtax.json")
    ap.add_argument("--out", default="revision1/fig_transfer.png")
    ap.add_argument("--resplits", default=None,
                    help="JSON whose resplits.per_seed to plot instead (e.g. s36_revision2.json, 20 re-splits)")
    ap.add_argument("--height-in", type=float, default=6.9, help="figure height in inches in --cmc mode")
    ap.add_argument("--cmc", action="store_true",
                    help="CMC layout: 6.5 in (16.51 cm) wide, Arial labels of at least 8 pt in black, panel letters, "
                         "and an RGB TIFF at 600 dpi next to the PNG")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = json.load(open(a.json))
    pairs, seeds = d["primary"]["pairs"], d["resplits"]["per_seed"]
    if a.resplits:
        seeds = json.load(open(a.resplits))["resplits"]["per_seed"]
    a1, a2 = d["alpha_flag"], d["alpha_risk"]
    panels = [
        ("G1 false-deferral rate (target 0.05)", "g1", lambda p: (p["g1"]["ffr"], p["g1"]["ci"]), a1, (0.0, 0.22)),
        ("G2 confident-error rate (target 0.10)", "cer", lambda p: (p["g2"]["controlled"], p["g2"]["ci"]), a2, (0.0, 0.45)),
        ("G2 acceptance rate", "acc", lambda p: (p["g2"]["accept"], None), None, (0.0, 0.9)),
    ]
    # font sizes (points at the printed size): title, tick labels, site labels, legend
    fs = dict(base=10.5, title=11, tick=9.5, site=10.5, legend=10, site_color="#444444")
    figsize = (10.6, 8.6)
    if a.cmc:
        fs = dict(base=9, title=9.5, tick=8.5, site=9, legend=8.5, site_color="black")
        figsize = (6.5, a.height_in)
        plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "text.color": "black",
                             "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
                             "mathtext.fontset": "custom", "mathtext.rm": "Arial"})
    plt.rcParams.update({"font.size": fs["base"], "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    grey, blue, red = "#6b6b6b", "#1f4e9c", "#c0392b"
    for ax, (title, key, primary, target, ylim) in zip(axes, panels):
        for i, (c, t) in enumerate(ORDER):
            if c != t:
                ax.axvspan(i - 0.45, i + 0.45, color="#f0f0f0", zorder=0)
            vals = [r[f"{key}|{c}->{t}"] for r in seeds]
            xs = [i - 0.24 + 0.40 * j / max(len(vals) - 1, 1) for j in range(len(vals))]
            ax.scatter(xs, vals, s=16 if len(vals) > 8 else 30, facecolors="none", edgecolors=grey, linewidths=1.1, zorder=3)
            m = sum(vals) / len(vals)
            ax.plot([i - 0.28, i + 0.20], [m, m], color=grey, lw=1.6, zorder=2)
            v, ci = primary(pairs[f"{c}->{t}"])
            if ci is not None:
                ax.errorbar([i + 0.32], [v], yerr=[[v - ci[0]], [ci[1] - v]], fmt="D", ms=5.5, color=blue,
                            capsize=3, lw=1.2, zorder=4)
            else:
                ax.scatter([i + 0.32], [v], marker="D", s=36, color=blue, zorder=4)
        for x in (2.5, 5.5):
            ax.axvline(x, color="#bbbbbb", lw=0.8, zorder=1)
        if target is not None:
            ax.axhline(target, color=red, ls="--", lw=1.2, zorder=1)
        ax.set_xlim(-0.55, len(ORDER) - 0.45)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=fs["title"], loc="left")
        if a.cmc:
            ax.text(-0.075, 1.04, "abc"[list(axes).index(ax)], transform=ax.transAxes, fontsize=fs["title"] + 1,
                    fontweight="bold", va="bottom", ha="left")
    axes[-1].set_xticks(range(len(ORDER)))
    axes[-1].set_xticklabels([f"{NAME[c]}\n$\\to$ {NAME[t]}" for c, t in ORDER], fontsize=fs["tick"])
    for j, t in enumerate(("nih", "vindr", "iuxray")):
        axes[0].text(3 * j + 1, 1.30 if a.cmc else 1.16, f"deployed at {NAME[t]}", ha="center", va="bottom", fontsize=fs["site"],
                     color=fs["site_color"], transform=axes[0].get_xaxis_transform())
    handles = [
        Line2D([], [], marker="o", ls="none", mfc="none", mec=grey, mew=1.1, ms=6, label=f"{len(seeds)} re-splits"),
        Line2D([], [], color=grey, lw=1.6, label="their mean"),
        Line2D([], [], marker="D", ls="none", color=blue, ms=6, label="primary split, 95% interval"),
        Line2D([], [], color=red, ls="--", lw=1.2, label="target"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=fs["legend"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.93 if a.cmc else 0.95))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=600 if a.cmc else 300)
    if a.cmc:
        from PIL import Image, ImageChops
        im = Image.open(a.out).convert("RGB")
        bbox = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()   # trim white margins (CMC)
        pad = 40
        im = im.crop((max(bbox[0] - pad, 0), max(bbox[1] - pad, 0), min(bbox[2] + pad, im.width), min(bbox[3] + pad, im.height)))
        im.save(a.out, dpi=(600, 600))
        im.save(os.path.splitext(a.out)[0] + ".tif", dpi=(600, 600),
                                               compression="tiff_lzw")
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
