"""[figure] Method overview (Figure 1): (a) three sites, (b) the frozen concept model,
(c) the two separate conformal rules and, with --audit, (d) the transfer audit. Drawn at the printed size, 165 mm wide (a full
page width of 16.51 cm), in millimetre coordinates, so every label prints at its stated point size: body text 8 pt,
headings 9 pt, panel letters 10 pt, all Arial and black. The radiographs are NIH ChestX-ray14 test
images (licence-safe); the small distributions in (c) are schematic, not data.

    python scripts/make_fig_method_2row.py --audit --px 3000 --out fig1.png \
        --pick nih=0 vindr=nih_1 iuxray=nih_2

Writes the PNG (600 dpi, RGB) and a TIFF next to it.
"""
from __future__ import annotations
import argparse
import os

W_MM, H_MM, DPI = 165.0, 103.0, 600
AUDIT_H = 39.0             # height added by panel (d), --audit
INK = "#000000"            # all labels in black
SOFT = "#5f6b76"           # rules and axis lines only, never text
LINE = "#c9d1d9"
BLUE, GREEN, ORANGE, RED = "#2456a6", "#2e7d4f", "#c0661f", "#b5382b"
BG = "#f5f7fa"
BODY, HEAD, LETTER = 8.0, 9.0, 10.0
MM = 1 / 25.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_method.png")
    ap.add_argument("--images", default="fig_assets")
    ap.add_argument("--audit", action="store_true", help="add panel (d), the transfer audit, as a third row")
    ap.add_argument("--px", type=int, default=0, help="also resize the PNG to this width in pixels")
    ap.add_argument("--pick", nargs="*", default=["nih=0", "vindr=nih_1", "iuxray=nih_2"])
    a = ap.parse_args()
    global H_MM
    OFF = AUDIT_H if a.audit else 0.0
    H_MM = H_MM + OFF
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle, Wedge
    from PIL import Image

    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "mathtext.fontset": "custom",
                         "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                         "text.color": INK})
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_MM); ax.set_ylim(0, H_MM); ax.axis("off")

    def box(x, y, w, h, fc="white", ec=LINE, lw=0.6, r=1.2, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw, zorder=z))

    def text(x, y, s, size=BODY, weight="normal", ha="left", va="center", z=6, **kw):
        assert size >= 8.0
        ax.text(x, y, s, fontsize=size, color=kw.pop("color", INK), weight=weight, ha=ha, va=va, zorder=z, **kw)

    def arrow(x0, y0, x1, y1, color=INK, lw=0.9, z=4, style="-|>", ms=7):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color,
                                     lw=lw, zorder=z, shrinkA=0, shrinkB=0))

    def inset(x, y, w, h):
        return fig.add_axes([x / W_MM, y / H_MM, w / W_MM, h / H_MM])

    def image(site):
        v = dict(p.split("=") for p in a.pick).get(site, "0")
        p = os.path.join(a.images, f"{v}.png" if "_" in v else f"{site}_{v}.png")
        return np.asarray(Image.open(p).convert("L"), dtype=float) / 255.0

    def thumb(img, x, y, w, h, grid=0):
        w_px, h_px = round(w * MM * DPI), round(h * MM * DPI)
        ih, iw = img.shape
        r = h_px / w_px
        if ih / iw > r:
            c = round(iw * r); o = (ih - c) // 2; img = img[o:o + c, :]
        else:
            c = round(ih / r); o = (iw - c) // 2; img = img[:, o:o + c]
        img = np.asarray(Image.fromarray((img * 255).astype("uint8")).resize((w_px, h_px), Image.LANCZOS)) / 255.0
        t = inset(x, y, w, h)
        t.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="none")
        for k in range(1, grid):
            t.axhline(k * h_px / grid - 0.5, color="white", lw=0.4, alpha=0.8)
            t.axvline(k * w_px / grid - 0.5, color="white", lw=0.4, alpha=0.8)
        t.set_xticks([]); t.set_yticks([])
        for sp in t.spines.values():
            sp.set_edgecolor("#333333"); sp.set_linewidth(0.5)

    def panel(x, y, w, h, letter, title):
        box(x, y, w, h, fc=BG, ec=LINE, lw=0.6, r=1.8, z=0)
        text(x + 2.2, y + h - 3.6, letter, size=LETTER, weight="bold")
        text(x + 6.6, y + h - 3.6, title, size=HEAD, weight="bold")

    # ================================================================ row 1
    R1, H1 = 48.0 + OFF, 54.0                       # row 1 spans y = 48 .. 102
    panel(1.0, R1, 61.0, H1, "a", "Three sites")
    panel(64.0, R1, 100.0, H1, "b", "Frozen concept model, fit once at NIH")

    # ---- (a) sites
    sites = [("nih", "NIH ChestX-ray14", "One hospital, USA", ["train", "val", "calib", "test"], BLUE),
             ("vindr", "VinDr-CXR", "Two hospitals, Vietnam", ["calib", "test"], ORANGE),
             ("iuxray", "Indiana University", "Indiana, USA", ["calib", "test"], GREEN)]
    for i, (key, name, where, splits, col) in enumerate(sites):
        y = R1 + H1 - 7.5 - 13.2 * (i + 1) + 1.0
        box(3.0, y, 57.0, 12.2, ec=LINE, lw=0.6, r=1.0)
        ax.add_patch(Rectangle((3.0, y + 0.25), 1.4, 11.7, fc=col, ec="none", zorder=3))
        text(6.2, y + 9.7, name, size=BODY, weight="bold")
        text(6.2, y + 6.5, where)
        x = 6.2
        for sp in splits:
            w = 1.55 * len(sp) + 2.6
            box(x, y + 1.1, w, 3.4, fc=col + "1f", ec=col + "80", lw=0.5, r=0.6, z=3)
            text(x + w / 2, y + 2.8, sp, ha="center")
            x += w + 1.0
        thumb(image(key), 47.0, y + 0.8, 11.2, 10.6)
    text(31.5, R1 + 4.9, "Heads and Platt scalars fit at NIH only", ha="center")
    text(31.5, R1 + 1.9, r"$\hat q$, $\tau$ and $\lambda$ calibrated at each site", ha="center")
    arrow(60.6, R1 + 31.0, 67.4, R1 + 31.0, lw=1.0)

    # ---- (b) model pipeline
    yc = R1 + 31.0                            # centre line of the pipeline
    thumb(image("nih"), 68.8, yc - 7.0, 14.0, 14.0, grid=4)
    text(75.8, yc - 8.4, "Frontal image,\n16-pixel patches", ha="center", va="top", linespacing=1.1)
    arrow(84.4, yc, 87.0, yc)
    ex0, ex1 = 87.2, 107.0
    ax.add_patch(Polygon([[ex0, yc - 10.5], [ex1, yc - 7.5], [ex1, yc + 7.5], [ex0, yc + 10.5]], closed=True,
                         fc="#e3ebf7", ec=BLUE, lw=0.8, zorder=3))
    text((ex0 + ex1) / 2, yc + 3.4, "BiomedCLIP", ha="center", weight="bold")
    text((ex0 + ex1) / 2, yc - 0.1, "ViT-B/16", ha="center")
    text((ex0 + ex1) / 2, yc - 3.6, "frozen", ha="center", style="italic")
    arrow(107.2, yc, 109.6, yc)
    vals = np.array([0.15, 0.62, 0.35, 0.88, 0.22, 0.51, 0.74, 0.30, 0.45, 0.66, 0.10, 0.56])
    cmap = plt.get_cmap("Blues")
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((109.8, yc - 9.0 + 1.5 * i), 3.0, 1.5, fc=cmap(0.25 + 0.7 * v), ec="white",
                               lw=0.4, zorder=4))
    text(110.0, yc - 10.4, r"$z\in\mathbb{R}^{512}$", ha="center", va="top")

    concepts = ["Atelectasis", "Cardiomegaly", "Consolidation", "Pleural effusion", "Infiltration", "Mass",
                "Nodule", "Pleural thickening", "Normal (anchor)"]
    pk = [0.18, 0.71, 0.09, 0.42, 0.25, 0.06, 0.12, 0.15, 0.22]
    hx, hy, hw, hh = 118.6, R1 + 3.0, 43.4, 43.0
    box(hx, hy, hw, hh, ec=LINE, lw=0.6, r=1.0)
    text(hx + hw / 2, hy + hh - 3.4, "Linear heads, Platt-scaled", ha="center", weight="bold")
    for i, (c, v) in enumerate(zip(concepts, pk)):
        y = hy + hh - 8.0 - 3.75 * i
        ax.plot([112.8, hx], [yc, y], color=SOFT, lw=0.3, alpha=0.6, zorder=3)
        text(hx + 2.0, y, c)
        bx = hx + 30.0
        ax.add_patch(Rectangle((bx, y - 0.9), 11.0, 1.8, fc="#e4e9ee", ec="none", zorder=3))
        ax.add_patch(Rectangle((bx, y - 0.9), 11.0 * v, 1.8, fc=BLUE if c.startswith("Normal") else GREEN,
                               ec="none", zorder=4))

    # two scores, bottom of (b)
    box(66.0, R1 + 1.5, 50.6, 13.2, ec=LINE, lw=0.6, r=1.0)
    text(68.0, R1 + 11.9, "Two scores per study", weight="bold")
    text(68.0, R1 + 7.9, r"$A(x)=1-p_{\mathrm{Normal}}(x)$")
    text(114.6, R1 + 7.9, "abnormality", ha="right")
    text(68.0, R1 + 3.8, r"$s(x)=A(x)\,\left(1-\max_k\,p_k(x)\right)$")
    text(114.6, R1 + 3.8, "valve", ha="right")
    arrow(hx, R1 + 8.0, 116.8, R1 + 8.0, style="-|>", lw=0.8)


    # ================================================================ row 2
    R2, H2 = 1.0 + OFF, 45.0
    # row link
    arrow(91.3, R1 + 1.4, 91.3, R2 + H2 + 0.3, lw=1.0)
    panel(1.0, R2, 163.0, H2, "c", "Two separate conformal rules, calibrated on labelled studies at the site of use")

    def card(x, tag, name, col):
        box(x, R2 + 2.0, 79.0, 31.0, ec=col, lw=0.8, r=1.0)
        box(x + 2.0, R2 + 27.6, 7.6, 3.8, fc=col, ec="none", r=0.7, z=3)
        text(x + 5.8, R2 + 29.5, tag, ha="center", weight="bold", color="white")
        text(x + 11.4, R2 + 29.5, name, weight="bold")

    def axes_style(t):
        t.set_xticks([]); t.set_yticks([]); t.patch.set_alpha(0)
        for k, sp in t.spines.items():
            sp.set_visible(k == "bottom"); sp.set_color(SOFT); sp.set_linewidth(0.6)

    xs = np.linspace(0, 1, 400)
    # ---- G1
    card(3.0, "G1", "False-deferral rule (split conformal)", GREEN)
    a1 = inset(6.0, R2 + 8.6, 36.0, 16.0)
    d = xs ** 1.6 * (1 - xs) ** 4.5; d /= d.max()
    q = xs[np.searchsorted(np.cumsum(d) / d.sum(), 0.95)]
    a1.fill_between(xs, 0, d, where=xs <= q, color=GREEN, alpha=0.30, lw=0)
    a1.fill_between(xs, 0, d, where=xs >= q, color=RED, alpha=0.55, lw=0)
    a1.plot(xs, d, color=GREEN, lw=0.8)
    a1.axvline(q, color=INK, lw=0.7, ls="--")
    a1.text(q + 0.02, 0.98, r"$\hat q$", fontsize=BODY, va="top")
    a1.text(0.25, 0.30, "accept", fontsize=BODY, ha="center")
    a1.text(q + 0.16, 0.30, "defer", fontsize=BODY, ha="center")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1.1); axes_style(a1)
    text(24.0, R2 + 5.6, r"$s(x)$ of abnormal studies", ha="center")
    text(45.0, R2 + 22.6, r"Defer if $s(x)>\hat q$", weight="bold")
    text(45.0, R2 + 17.8, "Target: at most 5% of")
    text(45.0, R2 + 14.3, "in-vocabulary abnormal")
    text(45.0, R2 + 10.8, "studies deferred")

    # ---- G2
    card(84.0, "G2", "Selective rule (conformal risk control)", BLUE)
    a2 = inset(87.0, R2 + 8.6, 36.0, 16.0)
    nrm = np.exp(-0.5 * ((xs - 0.30) / 0.12) ** 2)
    abn = 0.8 * np.exp(-0.5 * ((xs - 0.66) / 0.14) ** 2)
    tau, lam = 0.47, 0.12
    a2.axvspan(tau - lam, tau + lam, color="#9aa5b1", alpha=0.30, lw=0)
    a2.fill_between(xs, 0, nrm, color=BLUE, alpha=0.28, lw=0)
    a2.fill_between(xs, 0, abn, color=ORANGE, alpha=0.32, lw=0)
    a2.plot(xs, nrm, color=BLUE, lw=0.8); a2.plot(xs, abn, color=ORANGE, lw=0.8)
    a2.axvline(tau, color=INK, lw=0.7, ls="--")
    a2.annotate("", xy=(tau + lam, 1.16), xytext=(tau, 1.16),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK, shrinkA=0, shrinkB=0))
    a2.text(tau - 0.02, 1.16, r"$\tau$", fontsize=BODY, ha="right", va="center")
    a2.text(tau + lam + 0.02, 1.16, r"$\lambda$", fontsize=BODY, ha="left", va="center")
    a2.text(0.04, 1.06, "normal", fontsize=BODY, ha="left", va="bottom")
    a2.text(0.99, 0.86, "abnormal", fontsize=BODY, ha="right", va="bottom")
    a2.text(tau, 0.42, "defer", fontsize=BODY, ha="center",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
    a2.set_xlim(0, 1); a2.set_ylim(0, 1.32); axes_style(a2)
    text(105.0, R2 + 5.6, r"$A(x)$ of all studies", ha="center")
    text(126.0, R2 + 22.6, r"Accept if $|A(x)-\tau|\geq\lambda$", weight="bold")
    text(126.0, R2 + 17.8, "Target: P(accepted and")
    text(126.0, R2 + 14.3, "misclassified) at most 0.10")
    text(126.0, R2 + 10.8, "(joint, not conditional, rate)")

    text(7.6, R2 + 37.0, "Accepted studies return the call and named findings; deferred studies go to a radiologist.")

    # ================================================================ row 3: the transfer audit
    if a.audit:
        R3, H3 = 1.0, AUDIT_H - 2.0
        ACC = "#8a3b12"
        box(1.0, R3, 163.0, H3, fc="#fbf3ea", ec=ACC, lw=1.1, r=1.8, z=0)
        text(3.2, R3 + H3 - 3.6, "d", size=LETTER, weight="bold")
        text(7.6, R3 + H3 - 3.6, "Transfer audit, the design this study contributes", size=HEAD, weight="bold")
        arrow(82.5, R2 - 0.2, 82.5, R3 + H3 + 0.3, lw=1.0)
        # two thresholds
        def tbox(y, head, l1, l2, col):
            box(3.0, y, 37.0, 12.0, ec=col, lw=0.8, r=1.0)
            ax.add_patch(Rectangle((3.0, y + 0.25), 1.4, 11.5, fc=col, ec="none", zorder=3))
            text(6.2, y + 9.0, head, weight="bold")
            text(6.2, y + 5.6, l1)
            text(6.2, y + 2.3, l2)
        tbox(R3 + 17.0, "Transferred threshold", "calibrated at site c,", "applied at site d", RED)
        tbox(R3 + 2.0, "Local threshold", "recalibrated on labelled", "studies at site d", GREEN)
        # comparison
        box(45.5, R3 + 6.0, 34.0, 19.0, ec=LINE, lw=0.6, r=1.0)
        text(62.5, R3 + 21.4, "Compare with the", ha="center", weight="bold")
        text(62.5, R3 + 18.0, "stated target", ha="center", weight="bold")
        text(62.5, R3 + 13.8, "rates on the test", ha="center")
        text(62.5, R3 + 10.5, "studies of site d", ha="center")
        arrow(40.2, R3 + 23.0, 45.3, R3 + 19.5, lw=0.8)
        arrow(40.2, R3 + 8.0, 45.3, R3 + 11.5, lw=0.8)
        arrow(79.7, R3 + 15.5, 84.3, R3 + 15.5, lw=0.8)
        # schematic of the outcome over repartitions
        a3 = inset(87.0, R3 + 6.5, 33.0, 20.5)
        rng = np.random.default_rng(3)
        loc = 0.10 + rng.normal(0, 0.012, 40); trn = 0.165 + rng.normal(0, 0.012, 40)
        a3.scatter(0 + rng.uniform(-0.18, 0.18, 40), loc, s=5, fc="none", ec=GREEN, lw=0.5)
        a3.scatter(1 + rng.uniform(-0.18, 0.18, 40), trn, s=5, fc="none", ec=RED, lw=0.5)
        a3.axhline(0.10, color=INK, lw=0.7, ls="--")
        a3.set_xlim(-0.55, 1.55); a3.set_ylim(0.055, 0.215)
        a3.set_xticks([]); a3.set_yticks([]); a3.patch.set_alpha(0)
        for k, sp in a3.spines.items():
            sp.set_visible(k in ("left", "bottom")); sp.set_color(SOFT); sp.set_linewidth(0.6)
        text(94.6, R3 + 4.2, "local", ha="center")
        text(112.4, R3 + 4.2, "transferred", ha="center")
        text(85.8, R3 + 23.0, "rate", ha="center", rotation=90)
        text(118.6, R3 + 10.6, "target", ha="right")
        text(103.5, R3 + 29.4, "Each circle is one repartition", ha="center")
        # what is repeated
        x0 = 124.0
        rows = [("Repeated for", True), ("every ordered pair of sites,", False), ("both rules, 200 repartitions", False),
                ("with the predictor fixed", False), ("and 20 re-splits", False),
                ("Design choices tested", True), ("label definition, sampling", False), ("unit, encoder, classifier", False)]
        for i, (s, b) in enumerate(rows):
            text(x0, R3 + 29.0 - 3.55 * i - (1.0 if i >= 5 else 0.0), s, weight="bold" if b else "normal")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=DPI)
    Image.open(a.out).convert("RGB").save(a.out)
    Image.open(a.out).save(os.path.splitext(a.out)[0] + ".tif", dpi=(DPI, DPI), compression="tiff_lzw")
    if a.px:
        im = Image.open(a.out); im.resize((a.px, round(im.height * a.px / im.width)), Image.LANCZOS).save(a.out)
    print(f"[fig] wrote {a.out} ({W_MM:.0f} mm x {H_MM:.0f} mm)")


if __name__ == "__main__":
    main()
