"""ca_event_p2_versions.py - two NEW versions of basic plot 2 (the population average over the 4 mice), 2026-10-22; the deck
figure P2_average_over_mice.png is left as it is.

  v2   the same 3 columns (baseline / vehicle / SBI-553) with a colour scale and units on every panel
  4p   4 columns in the crossover's own pairs: SBI-553 day baseline | after SBI-553 | vehicle day baseline | after vehicle
       (each injection next to the baseline of the same day and the same mouse)

Top row: one row per mouse = the mean z of all its neurons (5-s bins); bottom row: mean +- SEM over the 4 mice (thin = each
mouse). z = dF/F / robust SD (MAD) of the session.

  python ca_event_p2_versions.py
"""
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401
import ca_event_analysis as E  # noqa: E402
import figfoot  # noqa: E402  (10-24: notes next to the PNG, not under the figure)

BIN = 5.0
NB = 200                                   # 200 x 5 s = 1000 s
VMIN, VMAX = 0.2, 1.4                        # the range the population means actually use
ZLAB = "population mean z\n(dF/F / robust SD)"
FSC = 1.0                                    # font scale (the slide parts are drawn bigger)
VALS = {}                                    # panel title -> first 3 min / stimulus blocks (for the slide sentences)


def pop_trace(s):
    Zb, _ = E.binned(s, BIN)
    return np.nanmean(Zb, 1)[:NB]


def column(fig, gs, c_, title, col, rows, labels, short=None, ylab="mouse"):
    """one column: heatmap (rows = mice) + mean +- SEM; rows = list of 1-D traces; returns the heatmap image, the line axis.
    10-24 figure review: the long row labels of the 2nd / 3rd heatmap ran into the heatmap on their left -> the full label
    (mouse ...) only on the first heatmap, the short one (day, n neurons) on the others; the 'first 3 min / stimulus blocks'
    values sat on the traces -> they are the title of the lower panel"""
    L_ = min(len(r) for r in rows)
    H = np.vstack([r[:L_] for r in rows])
    x = np.arange(L_) * BIN / 60
    a = fig.add_subplot(gs[0, c_])
    im = a.imshow(H, aspect="auto", cmap="magma", vmin=VMIN, vmax=VMAX, extent=[0, L_ * BIN / 60, len(H), 0],
                  interpolation="nearest")
    for b0, b1 in E.BLOCK_S:
        a.axvline(b0 / 60, color="w", lw=0.8, ls="--")
        a.axvline(b1 / 60, color="w", lw=0.8, ls="--")
    a.set_yticks(np.arange(len(H)) + 0.5, labels if (c_ == 0 or short is None) else short, fontsize=9.5 * FSC)
    a.tick_params(axis="x", labelsize=9.5 * FSC)
    a.set_xlabel("time in session (min)", fontsize=10 * FSC)
    a.set_title(title, color=col, fontsize=12.5 * FSC, fontweight="bold")
    if c_ == 0:
        a.set_ylabel(ylab, fontsize=10.5 * FSC)
    fig._p2_heat = getattr(fig, "_p2_heat", None) or a          # the first heatmap: the colour bar takes its height
    fig._p2_last = a                                           # the last heatmap: the colour bar sits right of it
    b = fig.add_subplot(gs[1, c_])
    for r, lab in zip(rows, labels):
        m = lab.split()[0]
        b.plot(x, r[:L_], color=E.MC.get(m, "0.5"), lw=0.8, alpha=0.65)
    mu, se = H.mean(0), H.std(0) / np.sqrt(len(H))
    b.plot(x, mu, color=col, lw=2.6)
    b.fill_between(x, mu - se, mu + se, color=col, alpha=0.25, lw=0)
    for b0, b1 in E.BLOCK_S:
        b.axvspan(b0 / 60, b1 / 60, color="#f5b041", alpha=0.13, lw=0)
    b.set_xlabel("time in session (min)", fontsize=10.5 * FSC)
    b.set_ylabel(ZLAB, fontsize=10.5 * FSC)
    b.tick_params(labelsize=9.5 * FSC)
    nost, blk = (x > 0.25) & (x < 3), x >= 3
    b.set_title(f"first 3 min {mu[nost].mean():.2f} z\nstimulus blocks {mu[blk].mean():.2f} z", loc="left",
                fontsize=9.5 * FSC, linespacing=1.15)
    VALS[title] = dict(first3=float(mu[nost].mean()), blocks=float(mu[blk].mean()), n_mice=int(len(H)))
    return im, b


def finish(fig, im, axes_b, out, note):
    lo = min(a.get_ylim()[0] for a in axes_b)
    hi = max(a.get_ylim()[1] for a in axes_b)
    for a in axes_b:
        a.set_ylim(lo, hi)
    # 10-24 figure review: the label was taller than the bar ('5-s bins' cut) and only two ticks showed -> the bar takes the
    # height of the heatmaps, a 3-line label, ticks at min, max and between
    p0 = fig._p2_heat.get_position()
    cax = fig.add_axes([fig._p2_last.get_position().x1 + 0.015, p0.y0, 0.011, p0.height])
    cb = fig.colorbar(im, cax=cax, ticks=np.linspace(VMIN, VMAX, 4))
    cb.set_label("mean z\n(dF/F / robust SD)\n5-s bins", fontsize=min(10 * FSC, 12), linespacing=1.1)
    cb.ax.tick_params(labelsize=10 * FSC)
    fig.legend(handles=[Line2D([], [], color=E.MC[m], lw=1.5, label=m) for m in E.MO]
               + [Line2D([], [], color="k", lw=2.6, label="mean +- SEM of the 4 mice"),
                  Patch(color="#f5b041", alpha=0.35, label="stimulus blocks (white dashed lines in the heatmaps)")],
               loc="lower center", ncol=6 if FSC <= 1.2 else 3, frameon=False, fontsize=10 * FSC,
               bbox_to_anchor=(0.47, -0.01 if FSC <= 1.2 else -0.11))
    figfoot.note(fig, note, "en")                     # 10-24: the methods note goes into the slide notes (all versions)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    global FSC
    S = E.load_all()
    T = {k: pop_trace(s) for k, s in S.items()}
    nn = {k: len(s["ids"]) for k, s in S.items()}
    # v2: the same three columns, with scale and units
    FSC = 1.3                                                    # bigger fonts on the slide version
    fig = plt.figure(figsize=(14.5, 6.4))                      # 10-24: <= ~6.8 in with the legend: shown at >= 76 %
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.4], hspace=0.66, wspace=0.28, right=0.88)
    axes_b, im = [], None
    for c_, c in enumerate(E.COND):
        rows, labs, sh = [], [], []
        for m in E.MO:
            ks = [k for k in T if k[0] == m and E.cond_of(*k) == c]
            if not ks:
                continue
            L_ = min(len(T[k]) for k in ks)
            rows.append(np.nanmean(np.vstack([T[k][:L_] for k in ks]), 0))
            dd = ' + '.join(f'D{k[1]}' for k in sorted(ks))
            labs.append(f"{m} ({dd})")
            sh.append(dd)
        ttl = {"baseline": "baseline (both days, mean)", "vehicle": "after vehicle", "SBI-553": "after SBI-553"}[c]
        im, b = column(fig, gs, c_, ttl, E.CC[c], rows, labs, short=sh, ylab="mouse (day)")
        axes_b.append(b)
    finish(fig, im, axes_b, os.path.join(E.OD, "P2_average_over_mice_v2_colorbar.png"),
           "n = 4 mice; each mouse = the mean of all its neurons in that session (baseline = mean of its two baseline sessions). "
           "Descriptive: z is scaled per session, so levels between sessions are only roughly comparable.")
    FSC = 1.0
    # 4 panels: each injection next to its own day's baseline
    fig = plt.figure(figsize=(21, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.45], hspace=0.55, wspace=0.34, right=0.9)
    axes_b = []
    cols = [("SBI-553", "pre", "SBI-553 day: baseline", E.CC["baseline"]), ("SBI-553", "post", "SBI-553 day: after SBI-553", E.CC["SBI-553"]),
            ("vehicle", "pre", "vehicle day: baseline", E.CC["baseline"]), ("vehicle", "post", "vehicle day: after vehicle", E.CC["vehicle"])]
    YL4 = "mouse, day\n(n neurons)"                              # 10-24: '(67)' was unexplained
    for c_, (drug, ph, ttl, col) in enumerate(cols):
        rows, labs, sh = [], [], []
        for m in E.MO:
            d = [dd for dd in (1, 2) if E.ARMS[dd][m] == drug][0]
            if (m, d, ph) in T:
                rows.append(T[(m, d, ph)])
                labs.append(f"{m} D{d} ({nn[(m, d, ph)]})")
                sh.append(f"D{d} ({nn[(m, d, ph)]})")
        im, b = column(fig, gs, c_, ttl, col, rows, labs, short=sh, ylab=YL4)
        axes_b.append(b)
    finish(fig, im, axes_b, os.path.join(E.OD, "P2b_average_4panels_SBIday_vehicleday.png"),
           "Crossover pairs: 5941 + 5939 got SBI-553 on Day 1 and vehicle on Day 2, 5937 + 1191 the reverse; each injection "
           "session is next to the baseline of the SAME day and mouse. Row labels: mouse, Day, (neurons). n = 4 mice per panel.")
    # the 4-panel figure also as two parts (one drug day each, bigger fonts) for the slides
    FSC = 1.3
    for suf, cc in (("_sbi", cols[:2]), ("_veh", cols[2:])):
        fig = plt.figure(figsize=(12.4, 5.8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.75, wspace=0.3, right=0.86)
        axes_b = []
        for c_, (drug, ph, ttl, col) in enumerate(cc):
            rows, labs, sh = [], [], []
            for m in E.MO:
                d = [dd for dd in (1, 2) if E.ARMS[dd][m] == drug][0]
                if (m, d, ph) in T:
                    rows.append(T[(m, d, ph)])
                    labs.append(f"{m} D{d} ({nn[(m, d, ph)]})")
                    sh.append(f"D{d} ({nn[(m, d, ph)]})")
            im, b = column(fig, gs, c_, ttl, col, rows, labs, short=sh, ylab=YL4)
            axes_b.append(b)
        finish(fig, im, axes_b, os.path.join(E.OD, f"P2b_average_4panels_SBIday_vehicleday{suf}.png"),
               "Each injection session next to the baseline of the same day and mouse. Row labels: mouse, Day, (neurons). n = 4 mice.")
    FSC = 1.0
    E.jdump(VALS, os.path.join(E.OD, "P2_versions.json"))
    print("->", os.path.join(E.OD, "P2_average_over_mice_v2_colorbar.png"))
    print("->", os.path.join(E.OD, "P2b_average_4panels_SBIday_vehicleday.png"))


if __name__ == "__main__":
    main()
