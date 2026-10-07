"""[figures] The three 'money' figures for the NIH+VinDr paper (matplotlib, no GPU).
Requires NIH + VinDr cached embeddings, the NIH-trained heads, and VinDr images on disk.

    python -m scripts.s13_figures --config configs/vindr.yaml
"""
from __future__ import annotations
import json
import os

from ovcbmr.config import parse_config_arg
from ovcbmr.concept.heads import load_heads, predict_matrix
from ovcbmr.concept.valve import image_unexplained
from ovcbmr.ood.conformal import split_conformal_threshold, empirical_flag_rate


def _load(emb_dir, proc, split):
    import torch
    d = torch.load(os.path.join(emb_dir, f"{split}_img.pt"), weights_only=False)
    with open(os.path.join(proc, "index.json")) as f:
        items = {it["id"]: it for it in json.load(f)}
    return d["emb"].numpy(), d["ids"], items


def _abnormal(emb, ids, items, heads, vocab):
    """(score, id, prob_vector) for IN-VOCABULARY known-abnormal images (>=1 concept-head
    finding present). Excludes No-finding AND abnormal-but-out-of-vocabulary cases, so the
    false-flag rate is the correct G1 quantity (flagging an OOV case is a correct unanchored
    detection, not a false flag)."""
    Pm = predict_matrix(emb, heads, vocab)
    ab_cols = [j for j, c in enumerate(vocab) if c != "Normal"]
    rows = []
    for i, iid in enumerate(ids):
        it = items.get(iid)
        if it is None or not any(int(it["labels_known"][j]) == 1 for j in ab_cols):
            continue
        rows.append((image_unexplained(Pm[i].tolist(), vocab, "Normal")[1], iid, Pm[i]))
    return rows


def main():
    cfg = parse_config_arg()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ch = cfg.cross_hospital
    heads = load_heads(os.path.join(ch.nih_processed, "concept_heads.json"))
    with open(os.path.join(ch.nih_processed, "meta.json")) as f:
        vocab = json.load(f)["vocabulary"]
    alpha = float(getattr(cfg.conformal, "alpha_flag", 0.05))
    res = cfg.paths.results
    os.makedirs(res, exist_ok=True)

    nih_cal = _abnormal(*_load(ch.nih_embeddings, ch.nih_processed, "calib"), heads, vocab)
    vin_cal = _abnormal(*_load(cfg.paths.embeddings, cfg.paths.processed, "calib"), heads, vocab)
    vin_te = _abnormal(*_load(cfg.paths.embeddings, cfg.paths.processed, "test"), heads, vocab)
    nih_cal_s = [r[0] for r in nih_cal]
    vin_cal_s = [r[0] for r in vin_cal]
    vin_te_s = [r[0] for r in vin_te]
    q_nih = split_conformal_threshold(nih_cal_s, alpha)

    # ---------------- Figure: valve-score shift (distribution; method fig is fig_method.svg) ----------------
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.hist(nih_cal_s, bins=40, density=True, alpha=0.55, color="#3b6fb0", label="NIH calib (abnormal)")
    ax.hist(vin_te_s, bins=40, density=True, alpha=0.55, color="#e08b2d", label="VinDr test (abnormal)")
    if q_nih != float("inf"):
        ax.axvline(q_nih, color="k", ls="--", lw=1.5, label=f"NIH $\\hat{{q}}$ (α={alpha})")
        ax.axvspan(q_nih, max(vin_te_s), color="#e08b2d", alpha=0.18)
        ff = empirical_flag_rate(vin_te_s, q_nih)
        ax.annotate(f"VinDr false-deferral {ff:.3f}\n(target {alpha})", xy=(q_nih, ax.get_ylim()[1] * 0.08),
                    xycoords="data", xytext=(0.60, 0.62), textcoords="axes fraction", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#8a5410"))
    ax.set_xlabel("unexplained valve score s(x)"); ax.set_ylabel("density")
    ax.set_title("Valve-score shift: NIH → VinDr"); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(res, "fig_score_shift.png"), dpi=200); plt.close(fig)

    # ---------------- Figure 2: coverage calibration ----------------
    alphas = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
    naive = [empirical_flag_rate(vin_te_s, split_conformal_threshold(nih_cal_s, a)) for a in alphas]
    gc = [empirical_flag_rate(vin_te_s, split_conformal_threshold(vin_cal_s, a)) for a in alphas]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot(alphas, alphas, "k--", label="target (ideal)")
    ax.plot(alphas, naive, "o-", color="#c0392b", label="transferred (NIH-calibrated)")
    ax.plot(alphas, gc, "s-", color="#27ae60", label="local (VinDr-calibrated)")
    ax.axhline(alpha, color="red", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("target false-deferral rate α"); ax.set_ylabel("empirical false-deferral rate on VinDr")
    ax.set_title("G1 coverage, NIH → VinDr"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(res, "fig2_coverage.png"), dpi=200); plt.close(fig)

    # ---------------- Figure 3: interpretable deferrals ----------------
    from PIL import Image
    ab_cols = [j for j, c in enumerate(vocab) if c != "Normal"]
    flagged = [r for r in vin_te if q_nih != float("inf") and r[0] > q_nih]
    flagged.sort(key=lambda r: -r[0])
    picks = flagged[:4] if len(flagged) >= 4 else vin_te[:4]
    img_root = cfg.data.image_root
    fig, axes = plt.subplots(2, 2, figsize=(8, 8.6))
    for ax, (s, iid, pv) in zip(axes.ravel(), picks):
        margins = sorted((abs(float(pv[j]) - 0.5), vocab[j]) for j in ab_cols)[:2]
        drivers = ", ".join(m[1] for m in margins)
        try:
            ax.imshow(Image.open(os.path.join(img_root, iid)).convert("L"), cmap="gray")
        except Exception:                                       # noqa: BLE001
            ax.text(0.5, 0.5, "(image not found)", ha="center", va="center")
        ax.set_title(f"Deferred — low confidence:\n{drivers}", fontsize=9); ax.axis("off")
    fig.suptitle("Interpretable abstention on VinDr (deferred cases)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(res, "fig3_interpretability.png"), dpi=200); plt.close(fig)

    print(f"[s13] wrote fig1_pipeline_shift.png, fig2_coverage.png, fig3_interpretability.png -> {res}")
    print(f"[s13] q_nih={q_nih:.3f} | VinDr false-flag @ α={alpha}: {empirical_flag_rate(vin_te_s, q_nih):.3f} "
          f"| deferred cases for Fig.3: {len(flagged)}")


if __name__ == "__main__":
    main()
