"""[figure] Method overview (Figure 1): (a) three sites, (b) the frozen concept model,
(c) the two separate conformal rules. Drawn with matplotlib and numpy only; the radiograph is a
procedural drawing and the small distributions are illustrative, not data. Sized for a full-width
(about 7 inch) figure, so that no text prints below 6 pt.

    python scripts/make_fig_method.py --out fig_method.png
"""
from __future__ import annotations
import argparse
import os

INK, MUTED, LINE = "#1f2933", "#55606b", "#c5ced8"
BLUE, GREEN, ORANGE, RED = "#2456a6", "#2e7d4f", "#c0661f", "#b5382b"
PANEL = {"a": "#f1f4f9", "b": "#f1f7f2", "c": "#fbf3ec"}
EDGE = {"a": "#a9bbd6", "b": "#a5cbb0", "c": "#e4b997"}
BODY = 9.5
FIG_W, FIG_H, DPI = 11.0, 4.3, 600


def radiograph(n=160):
    """A stylised frontal chest radiograph: dark lungs with faint vessels, bright spine, heart and ribs."""
    import numpy as np
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(3)
    y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    img = np.zeros((n, n))
    torso = ((x / 0.86) ** 2 + ((y - 0.25) / 1.15) ** 2) < 1
    img += 0.50 * torso
    img += 0.22 * (((x / 1.1) ** 2 + ((y + 0.95) / 0.32) ** 2) < 1) * (y < -0.62)       # shoulders
    lung = np.zeros((n, n))
    for cx, sx in ((-0.36, 0.27), (0.36, 0.29)):
        lung += ((((x - cx) / sx) ** 2 + ((y + 0.05) / 0.66) ** 2) < 1)
    vessels = gaussian_filter(rng.normal(size=(n, n)), 2.2)
    img -= lung * (0.34 - 0.05 * vessels / vessels.std())
    img += 0.30 * (np.abs(x) < 0.08) * (y > -0.9)                                       # spine
    img += 0.26 * ((((x - 0.10) / 0.28) ** 2 + ((y - 0.32) / 0.22) ** 2) < 1)           # heart
    for k in range(7):                                                                  # ribs
        c = -0.66 + 0.17 * k
        rib = np.abs(y - (c + 0.30 * x ** 2)) < 0.022
        img += 0.10 * rib * (np.abs(x) > 0.09) * (np.abs(x) < 0.72)
    img += 0.16 * (np.abs(y - (-0.70 + 0.08 * np.abs(x))) < 0.028) * (np.abs(x) < 0.58)   # clavicles
    img += 0.18 * (y > 0.60) * torso                                                    # below the diaphragm
    return np.clip(gaussian_filter(img, 1.2) + 0.03, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_method.png")
    ap.add_argument("--images", default="fig_assets",
                    help="folder with <site>_<k>.png from export_fig_images.py; a drawing is used if absent")
    ap.add_argument("--pick", nargs="*", default=["nih=0", "vindr=3", "iuxray=0"],
                    help="which exported image to show per site (vindr 0 and 1 carry a necklace and a marker)")
    a = ap.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle, Wedge

    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"],
                         "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 62.5)
    ax.axis("off")

    def box(x, y, w, h, fc="white", ec=LINE, lw=0.9, r=1.2, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw, zorder=z))

    def text(x, y, s, size=BODY, color=INK, weight="normal", ha="center", va="center", z=6, **kw):
        ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, zorder=z, **kw)

    def arrow(x0, y0, x1, y1, color=INK, lw=1.4, z=4):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=color,
                                     lw=lw, zorder=z, shrinkA=0, shrinkB=0))

    def inset(x, y, w, h):
        return fig.add_axes([x / 160, y / 62.5, w / 160, h / 62.5])

    def hospital(x, y, s, color):
        ax.add_patch(Rectangle((x, y), s, 0.72 * s, fc=color, ec="none", zorder=4))
        ax.add_patch(Rectangle((x + 0.3 * s, y + 0.72 * s), 0.4 * s, 0.32 * s, fc=color, ec="none", zorder=4))
        ax.add_patch(Rectangle((x + 0.46 * s, y + 0.77 * s), 0.08 * s, 0.22 * s, fc="white", ec="none", zorder=5))
        ax.add_patch(Rectangle((x + 0.39 * s, y + 0.84 * s), 0.22 * s, 0.08 * s, fc="white", ec="none", zorder=5))
        for i in range(3):
            for j in range(2):
                ax.add_patch(Rectangle((x + (0.12 + 0.3 * i) * s, y + (0.14 + 0.28 * j) * s), 0.16 * s, 0.14 * s,
                                       fc="white", ec="none", alpha=0.9, zorder=5))

    def snowflake(x, y, r, color):
        for ang in (0, 60, 120):
            t = np.deg2rad(ang)
            ax.plot([x - r * np.cos(t), x + r * np.cos(t)], [y - r * np.sin(t) * 1.3, y + r * np.sin(t) * 1.3],
                    color=color, lw=1.1, zorder=7, solid_capstyle="round")

    def person(x, y, s, color):
        ax.add_patch(Circle((x, y + 0.78 * s), 0.22 * s, fc=color, ec="none", zorder=5))
        ax.add_patch(Wedge((x, y), 0.45 * s, 0, 180, fc=color, ec="none", zorder=5))

    def check(x, y, s, color):
        ax.add_patch(Circle((x, y), s, fc=color, ec="none", zorder=5))
        ax.plot([x - 0.45 * s, x - 0.1 * s, x + 0.5 * s], [y, y - 0.4 * s, y + 0.45 * s],
                color="white", lw=1.6, zorder=6, solid_capstyle="round", solid_joinstyle="round")

    # ------------------------------------------------------------------ panels
    for x, w, key, title in [(0.8, 44.2, "a", "Three sites"), (46.4, 58.2, "b", "Frozen concept model"),
                             (106.0, 53.2, "c", "Two separate conformal rules")]:
        box(x, 0.8, w, 61.0, fc=PANEL[key], ec=EDGE[key], lw=1.0, r=2.2, z=0)
        text(x + 2.0, 58.6, f"({key})", size=11.5, weight="bold", ha="left", color=MUTED)
        text(x + 7.0, 58.6, title, size=11.5, weight="bold", ha="left")

    def real(site):
        pick = dict(x.split("=") for x in a.pick)
        v = pick.get(site, "0")
        # "vindr=nih_1" shows an NIH image in the VinDr slot (licence-safe illustration)
        p = os.path.join(a.images, f"{v}.png" if "_" in v else f"{site}_{v}.png")
        if not os.path.exists(p):
            return None
        from PIL import Image
        return np.asarray(Image.open(p).convert("L"), dtype=float) / 255.0

    def thumb(img, x, y, w, h, grid=0):
        """Show img in a w x h slot without distortion: centre-crop to the slot's aspect ratio, then
        resample once (Lanczos) to the exact output pixel size and draw it pixel for pixel."""
        from PIL import Image
        w_px, h_px = round(w / 160 * FIG_W * DPI), round(h / 62.5 * FIG_H * DPI)
        ih, iw = img.shape
        r = h_px / w_px
        if ih / iw > r:                                  # too tall: crop top and bottom
            ch = round(iw * r)
            o = (ih - ch) // 2
            img = img[o:o + ch, :]
        else:                                            # too wide: crop the sides
            cw = round(ih / r)
            o = (iw - cw) // 2
            img = img[:, o:o + cw]
        im = Image.fromarray((img * 255).astype("uint8")).resize((w_px, h_px), Image.LANCZOS)
        img = np.asarray(im, dtype=float) / 255.0
        t = inset(x, y, w, h)
        t.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="none")
        for k in range(1, grid):
            t.axhline(k * h_px / grid - 0.5, color="white", lw=0.5, alpha=0.75)
            t.axvline(k * w_px / grid - 0.5, color="white", lw=0.5, alpha=0.75)
        t.set_xticks([]); t.set_yticks([])
        for sp in t.spines.values():
            sp.set_edgecolor("#333333"); sp.set_linewidth(0.7)

    # ------------------------------------------------------------------ (a) sites
    sites = [(42.0, "NIH ChestX-ray14", "one hospital, USA", ["train", "val", "calib", "test"], BLUE, "source"),
             (27.6, "VinDr-CXR", "two hospitals, Vietnam", ["calib", "test"], ORANGE, "deployment"),
             (13.2, "Indiana University", "Indiana, USA", ["calib", "test"], GREEN, "deployment")]
    keys = ["nih", "vindr", "iuxray"]
    imgs = {k: real(k) for k in keys}
    have = all(v is not None for v in imgs.values())
    for (y, name, where, splits, color, role), key in zip(sites, keys):
        box(2.6, y, 40.6, 12.6, ec=color, lw=1.1)
        hospital(4.4, y + 5.3, 5.4, color)
        text(11.8, y + 9.9, name, size=10, weight="bold", ha="left")
        text(11.8, y + 6.6, where, color=MUTED, ha="left")
        if have:
            thumb(imgs[key], 32.2, y + 1.1, 10.4, 10.4)
        else:
            text(41.6, y + 9.9, role, size=8.6, color=color, weight="bold", ha="right")
        x = 4.4
        for sp in splits:
            w = 2.0 + 0.95 * len(sp)
            box(x, y + 1.2, w, 3.3, fc=color + "18", ec=color + "66", lw=0.6, r=0.7, z=3)
            text(x + w / 2, y + 2.85, sp, size=9)
            x += w + 0.8
        ax.plot([43.2, 44.6], [y + 6.3, y + 6.3], color=color, lw=1.3, zorder=3)
    ax.plot([44.6, 44.6], [19.5, 48.3], color=INK, lw=1.3, zorder=3)
    arrow(44.6, 34.0, 48.6, 34.0, lw=1.6)
    text(22.9, 8.6, "one model for every site", color=MUTED)
    text(22.9, 5.6, "heads and Platt scalars fit at NIH", color=MUTED)
    text(22.9, 2.6, r"$\hat q,\ \tau,\ \lambda$ calibrated per site", color=MUTED)

    # ------------------------------------------------------------------ (b) model
    if have:
        thumb(imgs["nih"], 48.3, 27.6, 12.8, 12.8, grid=4)
    else:
        thumb(radiograph() / 0.85, 48.3, 27.6, 12.8, 12.8, grid=4)
    text(54.7, 26.2, "frontal radiograph,\nsplit into patches", size=8.6, color=MUTED, va="top", linespacing=1.05)
    arrow(61.3, 34.0, 62.5, 34.0)

    ax.add_patch(Polygon([[62.6, 24.6], [74.6, 27.0], [74.6, 41.0], [62.6, 43.4]], closed=True,
                         fc="#dde7f6", ec=BLUE, lw=1.1, zorder=3))
    for k in range(1, 6):
        xx = 62.6 + 2.0 * k
        top = 43.4 - (43.4 - 41.0) * (xx - 62.6) / 12.0
        bot = 24.6 + (27.0 - 24.6) * (xx - 62.6) / 12.0
        ax.plot([xx, xx], [bot + 0.8, top - 0.8], color=BLUE, lw=0.6, alpha=0.55, zorder=4)
    text(68.6, 36.6, "BiomedCLIP", size=8.8, weight="bold", bbox=dict(fc="#dde7f6", ec="none", pad=0.6))
    text(68.6, 33.6, "ViT-B/16", size=8.8, color=MUTED, bbox=dict(fc="#dde7f6", ec="none", pad=0.6))
    snowflake(66.2, 30.0, 0.9, BLUE)
    text(67.6, 30.0, "frozen", size=8.8, color=BLUE, weight="bold", ha="left", bbox=dict(fc="#dde7f6", ec="none", pad=0.6))
    arrow(74.8, 34.0, 76.6, 34.0)

    vals = np.array([0.15, 0.62, 0.35, 0.88, 0.22, 0.51, 0.74, 0.30, 0.45, 0.66, 0.10, 0.56])
    cmap = plt.get_cmap("Blues")
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((76.9, 25.3 + 1.45 * i), 2.2, 1.45, fc=cmap(0.25 + 0.7 * v), ec="white",
                               lw=0.5, zorder=4))
    text(78.0, 23.8, r"$z\in\mathbb{R}^{512}$", size=9, va="top")
    concepts = ["Atelectasis", "Cardiomegaly", "Consolidation", "Pleural Effusion", "Infiltration", "Mass",
                "Nodule", "Pleural Thickening", "Normal"]
    pk = [0.18, 0.71, 0.09, 0.42, 0.25, 0.06, 0.12, 0.15, 0.22]
    ys = [50.2 - 3.05 * i for i in range(9)]
    box(82.0, 22.0, 21.4, 33.8, ec=EDGE["b"], lw=0.9)
    text(92.7, 53.6, "linear heads + Platt", size=9, weight="bold")
    for y, c, v in zip(ys, concepts, pk):
        ax.plot([79.1, 82.0], [34.0, y], color="#9aa5b1", lw=0.45, zorder=3)
        col = BLUE if c == "Normal" else GREEN
        text(83.0, y, c, size=8.8, ha="left")
        ax.add_patch(Rectangle((98.4, y - 0.75), 4.4, 1.5, fc="#e7ecf0", ec="none", zorder=3))
        ax.add_patch(Rectangle((98.4, y - 0.75), 4.4 * v, 1.5, fc=col, ec="none", zorder=4))
    text(92.7, 23.8, r"calibrated $p_k(x)$", size=8.8, color=MUTED)

    box(48.6, 2.6, 54.8, 15.4, ec=LINE)
    text(50.0, 15.6, "two scores per study", size=9, weight="bold", ha="left", color=MUTED)
    box(50.0, 8.6, 52.0, 5.0, fc=GREEN + "14", ec=GREEN + "88", lw=0.8, r=0.8, z=3)
    text(51.4, 11.1, r"$s(x)=A(x)\,(1-\max_{k}\,p_k(x))$", size=9.5, color=GREEN, ha="left")
    text(101.0, 11.1, "valve statistic", size=9, color=MUTED, ha="right")
    box(50.0, 3.4, 52.0, 4.6, fc=BLUE + "14", ec=BLUE + "88", lw=0.8, r=0.8, z=3)
    text(51.4, 5.7, r"$A(x)=1-p_{\mathrm{Normal}}(x)$", size=9.5, color=BLUE, ha="left")
    text(101.0, 5.7, "abnormality score", size=9, color=MUTED, ha="right")
    arrow(92.7, 21.8, 92.7, 18.2)

    # ------------------------------------------------------------------ (c) rules
    def card(y, h, tag, name, color):
        box(107.6, y, 50.0, h, ec=color, lw=1.1)
        box(109.0, y + h - 4.4, 6.4, 3.4, fc=color, ec="none", r=0.8, z=3)
        text(112.2, y + h - 2.7, tag, size=9.5, weight="bold", color="white")
        text(116.6, y + h - 2.7, name, size=9.5, weight="bold", ha="left")

    xs = np.linspace(0, 1, 300)

    # G1: calibration scores of in-vocabulary abnormal studies, threshold at the 95th percentile
    card(36.8, 19.8, "G1", "false-deferral rule, split conformal", GREEN)
    a1 = inset(109.4, 39.6, 23.4, 11.0)
    dens = xs ** 1.6 * (1 - xs) ** 4.5
    dens /= dens.max()
    cdf = np.cumsum(dens) / dens.sum()
    q = xs[np.searchsorted(cdf, 0.95)]
    a1.fill_between(xs, 0, dens, where=xs <= q, color=GREEN, alpha=0.35, lw=0)
    a1.fill_between(xs, 0, dens, where=xs >= q, color=RED, alpha=0.55, lw=0)
    a1.plot(xs, dens, color=GREEN, lw=1.0)
    a1.axvline(q, color=INK, lw=1.0, ls="--")
    a1.text(q + 0.03, 0.80, r"$\hat q$", fontsize=9.5, color=INK)
    a1.text(0.22, 0.25, "accept", fontsize=8.6, color=GREEN, ha="center", weight="bold")
    a1.text(q + 0.17, 0.25, "defer", fontsize=8.6, color=RED, ha="center", weight="bold")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1.12)
    a1.set_xlabel(r"$s(x)$", fontsize=8.6, labelpad=0.5, color=MUTED)
    text(134.6, 49.2, r"defer if $s(x)>\hat q$", size=9.5, ha="left")
    text(134.6, 45.6, "known-abnormal", size=8.8, color=MUTED, ha="left")
    text(134.6, 42.8, r"deferral $\leq 0.05$", size=8.8, color=MUTED, ha="left")

    # G2: abnormality score of normal and abnormal studies, abstention band around tau
    card(14.6, 21.2, "G2", "selective rule, conformal risk control", BLUE)
    a2 = inset(109.4, 17.4, 23.4, 12.2)
    nrm = np.exp(-0.5 * ((xs - 0.30) / 0.12) ** 2)
    abn = 0.8 * np.exp(-0.5 * ((xs - 0.66) / 0.14) ** 2)
    tau, lam = 0.47, 0.12
    a2.axvspan(tau - lam, tau + lam, color="#9aa5b1", alpha=0.35, lw=0)
    a2.fill_between(xs, 0, nrm, color=BLUE, alpha=0.30, lw=0)
    a2.fill_between(xs, 0, abn, color=ORANGE, alpha=0.35, lw=0)
    a2.plot(xs, nrm, color=BLUE, lw=0.9)
    a2.plot(xs, abn, color=ORANGE, lw=0.9)
    a2.axvline(tau, color=INK, lw=1.0, ls="--")
    a2.text(tau - 0.03, 1.13, r"$\tau$", fontsize=9.5, ha="right", va="bottom")
    a2.annotate("", xy=(tau + lam, 1.17), xytext=(tau, 1.17),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=INK, shrinkA=0, shrinkB=0))
    a2.text(tau + lam / 2, 1.20, r"$\lambda$", fontsize=9, ha="center", va="bottom")
    a2.text(0.03, 1.02, "normal", fontsize=8.6, color=BLUE, ha="left")
    a2.text(0.99, 0.86, "abnormal", fontsize=8.6, color=ORANGE, ha="right")
    a2.text(tau, 0.50, "defer", fontsize=8.6, color=RED, ha="center", weight="bold", zorder=5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
    a2.set_xlim(0, 1); a2.set_ylim(0, 1.45)
    a2.set_xlabel(r"$A(x)$", fontsize=8.6, labelpad=0.5, color=MUTED)
    text(134.6, 28.0, r"accept if $|A(x)-\tau|\geq\lambda$", size=9.5, ha="left")
    text(134.6, 24.4, "joint confident-error rate", size=8.8, color=MUTED, ha="left")
    text(134.6, 21.6, r"$P(\mathrm{accept}\wedge\mathrm{error})\leq 0.10$", size=8.8, color=MUTED, ha="left")
    text(134.6, 17.8, r"not $P(\mathrm{error}\mid\mathrm{accept})$;" "\n" "rules not combined", size=8.6, color=MUTED, ha="left",
         style="italic", linespacing=1.05)
    for axi in (a1, a2):
        axi.set_yticks([]); axi.set_xticks([])
        axi.patch.set_alpha(0)
        for k, sp in axi.spines.items():
            sp.set_visible(k == "bottom"); sp.set_color(MUTED); sp.set_linewidth(0.8)

    # outcomes
    box(107.6, 2.4, 24.2, 10.0, fc="#e9f4ed", ec=GREEN, lw=1.0, r=1.0)
    check(111.6, 7.4, 2.1, GREEN)
    text(115.0, 9.0, "accept", size=9.5, weight="bold", color=GREEN, ha="left")
    text(115.0, 5.6, "call + concepts", size=8.8, ha="left")
    box(133.4, 2.4, 24.2, 10.0, fc="#f9ebe8", ec=RED, lw=1.0, r=1.0)
    person(137.6, 4.9, 4.6, RED)
    text(141.2, 9.0, "defer", size=9.5, weight="bold", color=RED, ha="left")
    text(141.2, 5.6, "to a radiologist", size=8.8, ha="left")

    # scores -> rules
    arrow(102.0, 11.1, 107.4, 46.0, color=GREEN, lw=1.5)
    arrow(102.0, 5.7, 107.4, 24.0, color=BLUE, lw=1.5)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=DPI)
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
