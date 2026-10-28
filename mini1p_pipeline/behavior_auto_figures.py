"""Figures of behavior_auto_annotate.py: the automatic annotation next to the user's, for the
whole session and two zoomed 30-s windows, and a summary of the agreement.

  python behavior_auto_figures.py --dir OUT --stim "Pin prick,Heat,Mild touch,Light touch" \
      --zoom 400,696 --title "5941 SBI (2026-10-19)"
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from behavior_supervised import BLOCKS  # noqa: E402

FPS = 30.0
KEYC = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c", 4: "#9467bd", 0: "#7f7f7f"}


def spans(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def draw_rows(ax, A, stim, t0, t1, with_prob=False):
    rows = ["your deliveries", "auto deliveries", "your withdrawal", "auto withdrawal", "your flinch",
            "auto flinch", "your escape/rear", "auto escape/rear", "auto general movement",
            "auto grooming (candidates)"]
    ny = len(rows)
    y = {r: ny - 1 - i for i, r in enumerate(rows)}
    for k, (b, s0, s1) in enumerate(BLOCKS[1:], 1):
        if s1 > t0 and s0 < t1:
            ax.axvspan(max(s0, t0), min(s1, t1), color=KEYC[k], alpha=0.07, lw=0)
            ax.text((max(s0, t0) + min(s1, t1)) / 2, ny - 0.35, f"block {k}: {stim[k - 1]}", ha="center",
                    va="bottom", fontsize=8, color=KEYC[k])

    def ticks(fr, row, colors, lw=1.2):
        t = fr / FPS
        m = (t >= t0) & (t <= t1)
        ax.vlines(t[m], y[row] - 0.38, y[row] + 0.38, colors=np.array(colors, object)[m] if len(fr) else "k", lw=lw)

    ticks(A["u_del"], "your deliveries", [KEYC.get(int(k), "k") for k in A["u_typ"]])
    ticks(A["a_del"], "auto deliveries", [KEYC.get(int(k), "k") for k in A["a_typ"]])
    ticks(A["u_w"], "your withdrawal", ["#e6a700"] * len(A["u_w"]))
    ticks(A["w_ev"], "auto withdrawal", ["#e6a700"] * len(A["w_ev"]))
    ticks(A["u_f"], "your flinch", ["#555555"] * len(A["u_f"]))
    ticks(A["f_ev"], "auto flinch", ["#555555"] * len(A["f_ev"]))
    for row, mk, c in (("your escape/rear", A["u_esc"], "#2ca02c"), ("auto escape/rear", A["rear"], "#2ca02c"),
                       ("auto general movement", A["move"], "#17a2b8"),
                       ("auto grooming (candidates)", A.get("groom", np.zeros(1, bool)), "#d96fa6")):
        for s, e in spans(mk):
            if e / FPS >= t0 and s / FPS <= t1:
                ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                    (s / FPS, y[row] - 0.35), (e - s) / FPS, 0.7, color=c, lw=0))
    # deliveries the algorithm missed (red ring on your tick) and extra ones (red x)
    for f in A["missed"]:
        if t0 <= f / FPS <= t1:
            ax.plot(f / FPS, y["your deliveries"], "o", mfc="none", mec="r", ms=7, mew=1.2)
    for f in A["extra"]:
        if t0 <= f / FPS <= t1:
            ax.plot(f / FPS, y["auto deliveries"], "x", color="r", ms=6, mew=1.2)
    if with_prob:
        t = np.arange(len(A["Pd"])) / FPS
        m = (t >= t0) & (t <= t1)
        ax.plot(t[m], y["auto deliveries"] - 0.45 + 0.9 * A["Pd"][m] / max(A["Pd"][m].max(), 1e-3), color="k", lw=0.5, alpha=0.5)
    ax.set_yticks([y[r] for r in rows], rows, fontsize=8)
    ax.set_ylim(-0.6, ny + 0.2)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("time in session (s)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stim", required=True)
    ap.add_argument("--zoom", default="400,696")
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    stim = [s.strip() for s in a.stim.split(",")]
    Z = np.load(os.path.join(a.dir, "auto_annotation_arrays.npz"))
    A = {k: Z[k] for k in Z.files}
    R = json.load(open(os.path.join(a.dir, "auto_vs_your_annotation.json")))
    # matched / missed / extra deliveries (same rule as the annotator: one-to-one, <= 0.3 s)
    from behavior_auto_annotate import match
    tp, fp, fn, M = match(A["a_del"], A["u_del"])
    mi = {i for i, _ in M}
    mu = {j for _, j in M}
    A["missed"] = np.array([f for j, f in enumerate(A["u_del"]) if j not in mu], int)
    A["extra"] = np.array([f for i, f in enumerate(A["a_del"]) if i not in mi], int)
    n = len(A["Pd"])

    # ---- 1. timeline -----------------------------------------------------------------------
    z = [float(v) for v in a.zoom.split(",")]
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 0.22], hspace=0.35, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    draw_rows(ax, A, stim, 0, n / FPS)
    d = R["delivery"]
    ax.set_title(f"Whole session: your annotation vs the automatic one (cross-validated: no frame was predicted "
                 f"by a model trained on it). Deliveries: {d['TP']} of your {d['TP'] + d['FN']} found, "
                 f"{d['FP']} extra.", fontsize=10)
    for j, t0 in enumerate(z):
        axz = fig.add_subplot(gs[1, j])
        draw_rows(axz, A, stim, t0, t0 + 30, with_prob=True)
        axz.set_title(f"zoom {t0:.0f}-{t0 + 30:.0f} s (grey line = delivery probability)", fontsize=10)
    axl = fig.add_subplot(gs[2, :])
    axl.axis("off")
    h = [Line2D([0], [0], color=KEYC[k], lw=2, label=f"key {k} = {stim[k - 1]}") for k in range(1, 5)]
    h += [Line2D([0], [0], marker="o", mfc="none", mec="r", ls="", label="your delivery the algorithm missed"),
          Line2D([0], [0], marker="x", color="r", ls="", label="automatic delivery not in yours"),
          Line2D([0], [0], color="#e6a700", lw=2, label="withdrawal tap"),
          Line2D([0], [0], color="#555555", lw=2, label="flinch tap"),
          Line2D([0], [0], color="#2ca02c", lw=6, label="escape / rearing"),
          Line2D([0], [0], color="#17a2b8", lw=6, label="general movement (new, automatic only)"),
          Line2D([0], [0], color="#d96fa6", lw=6, label="grooming candidates (new, automatic only)")]
    axl.legend(handles=h, ncol=5, loc="center", frameon=False, fontsize=9)
    fig.suptitle(f"{a.title}: automatic annotation that mimics yours (matched when <= 0.3 s apart)", fontsize=13)
    fig.savefig(os.path.join(a.dir, "auto_vs_your_timeline.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. summary --------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.2))
    cats = [("delivery", "deliveries"), ("withdrawal_taps_around_your_deliveries", "withdrawal\n(your\ndeliveries)"),
            ("flinch_taps_around_your_deliveries", "flinch\n(your\ndeliveries)"),
            ("withdrawal_taps_fully_automatic", "withdrawal\n(fully\nautomatic)"),
            ("flinch_taps_fully_automatic", "flinch\n(fully\nautomatic)")]
    x = np.arange(len(cats))
    P = [R[c]["precision"] for c, _ in cats]
    Rc = [R[c]["recall"] for c, _ in cats]
    ax[0].bar(x - 0.2, Rc, 0.4, label="recall (yours found)")
    ax[0].bar(x + 0.2, P, 0.4, label="precision (automatic correct)")
    for i, (c, _) in enumerate(cats):
        ax[0].text(i, 1.02, f"n={R[c]['TP'] + R[c]['FN']}", ha="center", fontsize=8)
    ax[0].set_xticks(x, [l for _, l in cats], fontsize=8)
    ax[0].set_ylim(0, 1.15)
    ax[0].set_ylabel("fraction")
    ax[0].set_title("A. Event agreement (<= 0.3 s)", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2)
    for j, nm in enumerate(("withdrawal", "flinch")):
        T = R[f"{nm}_per_delivery_by_block"]
        ks = list(T.keys())
        xx = np.arange(len(ks))
        ax[1 + j].bar(xx - 0.2, [T[k]["user_pct"] for k in ks], 0.4, label="yours")
        ax[1 + j].bar(xx + 0.2, [T[k]["auto_pct"] for k in ks], 0.4, label="automatic")
        for i, k in enumerate(ks):
            ax[1 + j].text(i, 104, f"n={T[k]['n']}\nagree {100 * T[k]['agreement']:.0f}%", ha="center", fontsize=7)
        ax[1 + j].set_xticks(xx, [k.replace(" ", "\n", 1) for k in ks], fontsize=8)
        ax[1 + j].set_ylim(0, 140)
        ax[1 + j].set_ylabel(f"% of your deliveries with a {nm}")
        pdl = R[f"{nm}_per_delivery"]
        ax[1 + j].set_title(f"{'BC'[j]}. {nm.capitalize()} per delivery: agreement {100 * pdl['agreement']:.0f}%, "
                            f"kappa {pdl['kappa']:.2f}", fontsize=10)
        ax[1 + j].legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    pt = R["general_movement_pct_time_by_block"]
    ks = list(pt.keys())
    ax[3].bar(np.arange(len(ks)), [pt[k] for k in ks], color="#17a2b8")
    lab = ["baseline"] + [f"block{i}\n{stim[i - 1]}" for i in range(1, 5)]
    ax[3].set_xticks(np.arange(len(ks)), lab, fontsize=8)
    ax[3].set_ylabel("% of time in general movement")
    ma = R["general_movement_starts_after_your_deliveries_pct"]
    for i, k in enumerate(list(ma.keys()), 1):
        ax[3].text(i, pt[ks[i]] + 0.3, f"{ma[k]:.0f}% of\ndeliveries\nfollowed", ha="center", va="bottom", fontsize=7)
    ax[3].set_ylim(0, 1.45 * max(pt.values()) + 1)
    ax[3].set_title("D. General movement (automatic only)", fontsize=10)
    fig.suptitle(f"{a.title}: how well the automatic annotation reproduces yours (cross-validated)", fontsize=12)
    fig.text(0.5, -0.02, "Precision = share of automatic marks that match one of yours; recall = share of yours "
             "that the algorithm found. Kappa: chance-corrected agreement (0 = chance, 1 = perfect).",
             ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, "auto_vs_your_summary.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(os.path.join(a.dir, "auto_vs_your_timeline.png"))
    print(os.path.join(a.dir, "auto_vs_your_summary.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
