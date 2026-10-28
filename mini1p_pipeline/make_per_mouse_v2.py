"""make_per_mouse_v2.py - the per-mouse slides with the BEST window (2026-10-22), Day 1 and Day 2, in the new-style layout
(title, ONE key sentence, the figure, what we did / found with n and the test at the bottom, details in the notes).

Every neuronal number comes from the event analysis (ca_event_analysis.py): the best window per event (stimuli 0-0.5 s after vs
the 0.5 s before; behaviours 0-0.5 s after vs the 1 s before), the within-block time-shift test (1000 shuffles, BH q <= 0.20 per
session) and its classes (UP = activated, DOWN = inhibited). Traces are shown minus the 3 s before the event (display only).
Events: the 4 stimuli, withdrawal, flinch, movement, escape / rearing, grooming and attending onsets - every figure is split into
three groups of events so that the fonts stay readable.

Per mouse and day: every neuron labelled | all neurons with the scored events (baseline, injection) | where the responsive neurons
are (3 groups) | UP / DOWN / other neurons (3) | population response (3) | counts | neuron x event matrix | responsive neurons
after the injection (2).

  python make_per_mouse_v2.py --lang en|kr --out <deck.pptx> [--days 1 2]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401
import ca_day1_analysis as CA  # noqa: E402
import pairstars as PS  # noqa: E402
import make_new_deck as ND  # noqa: E402

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
EA = os.path.join(B, "analysis", "event_analysis")
ROOT = {1: os.path.join(B, "mini1p_out", "20261019"), 2: os.path.join(B, "mini1p_out", "20261021")}
# 10-23: the timing-corrected copies of the final scoring (scoring_timefix.py: withdrawal / flinch -0.40 s Day 1, -0.37 s
# Day 2; deliveries moved only where their own reflex came first; 3 Day-2 clicker fixes). SCORING_SUFFIX="" = the originals.
SCORING_SFX = os.environ.get("SCORING_SUFFIX", "_timefix")
SCORE = {1: os.path.join(B, "scoring_mini1p", "20261019_final" + SCORING_SFX),
         2: os.path.join(B, "scoring_mini1p", "20261021_final" + SCORING_SFX)}
MO, ARMS, CC = ND.MO, ND.ARMS, ND.CC
SEX = {"5941": "F", "1191": "F", "5939": "M", "5937": "M"}
STIM = ND.STIM
GROUPS = [("stimuli", "자극", STIM), ("withdrawal, flinch, movement", "withdrawal, flinch, movement",
                                     ["withdrawal", "flinch", "movement onset"]),
          ("escape, grooming, attending", "escape, grooming, attending", ["escape/rearing onset", "grooming onset", "attending onset"])]
ALL = [e for g in GROUPS for e in g[2]]
SH = {"Light touch": "light touch", "Mild touch": "mild touch", "Pin prick": "pin prick", "Heat": "heat", "withdrawal": "withdrawal",
      "flinch": "flinch", "movement onset": "movement", "escape/rearing onset": "escape", "grooming onset": "grooming",
      "attending onset": "attending"}
ECOL = {"Light touch": "#5dade2", "Mild touch": "#f5b041", "Pin prick": "#e74c3c", "Heat": "#8e44ad", "withdrawal": "#6c3483",
        "flinch": "#1e8449", "movement onset": "#17a2b8", "escape/rearing onset": "#7f8c8d", "grooming onset": "#b9770e",
        "attending onset": "#c0392b"}
UPC, DNC, NOC = "#c0392b", "#2471a3", "0.6"
PH = {"pre": "baseline", "post": "after injection"}
RC = {"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
      "legend.fontsize": 10.5, "axes.spines.top": False, "axes.spines.right": False}
plt.rcParams.update(RC)
WIN = {}
ZLAB = "z (dF/F / robust SD), minus mean of the 3 s before"           # the display baseline of the traces (disp())
NOC_MAP = "#d9d9d9"                                                    # 'no response' outlines on the maps (10-24: lighter, thicker)


def wwin(ev):
    pre, gap, a, b = WIN.get(ev, (2.0, 0.0, 0.0, 2.0))
    a0 = min(a, 0.0) - gap
    return (a0 - pre, a0), (a, b)


def wtxt(ev):
    (p0, p1), (a, b) = wwin(ev)
    return f"{a:g}-{b:g} s vs {p0:g}..{p1:g} s"


def pstr(p):
    return "n.a." if p is None or not np.isfinite(p) else ("p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}")


# ------------------------------------------------------------------------------------------------ data
def load_classes():
    T = pd.read_csv(os.path.join(EA, "cells_classes.csv"))
    T["mouse"] = T.mouse.astype(str)
    T["cls_shift"] = T.cls.map({"activated": "UP", "inhibited": "DOWN"}).fillna("no response")
    return T


def session(m, d, ph):
    s = CA.load_session(ROOT[d], m, ph, SCORE[d], day=d)
    if s is None or s["B"] is None:
        return None
    Sm, _ = CA.scoring_for(SCORE[d], m, ph)
    s["B"]["ev"]["attending onset"] = CA.onsets(np.atleast_1d(Sm["score"]).astype(int) == 1) + 1
    return s


def psth(s, ev):
    """trial-mean z (time x cells), tt, n trials; None if < CA.MIN_TRIALS"""
    X, tt, used = CA.peri(s, s["B"]["ev"].get(ev, []))
    if X is None or len(used) < CA.MIN_TRIALS:
        return None
    return np.nanmean(X, 0), tt, len(used)


def disp(mu, tt):
    return mu - np.nanmean(mu[(tt >= -3) & (tt < 0)], 0, keepdims=True)


def wresp(mu, tt, ev):
    (p0, p1), (a, b) = wwin(ev)
    return np.nanmean(mu[(tt >= a) & (tt < b)], 0) - np.nanmean(mu[(tt >= p0) & (tt < p1)], 0)


def fov(m, d):
    tag = f"{m}_d{d}"
    r = os.path.join(ROOT[d], tag)
    S = np.asarray(sio.loadmat(os.path.join(r, "3_extract", f"{tag}_extract.mat"), variable_names=["spatial_weights"])["spatial_weights"],
                   np.float32)
    R = pd.read_csv(os.path.join(r, "4_cell_review", f"{tag}_cell_review.csv"))
    bg = np.asarray(sio.loadmat(os.path.join(r, "3_extract", f"{tag}_projections.mat"), variable_names=["max_dff"])["max_dff"], float)
    lo, hi = np.percentile(bg, [1, 99.8])
    return S, R, np.clip((bg - lo) / (hi - lo + 1e-9), 0, 1) ** 0.7


def contours(S, ids):
    out = {}
    for c in ids:
        if c - 1 >= S.shape[2]:
            continue
        f = S[:, :, c - 1]
        if f.max() <= 0:
            continue
        cs, _ = cv2.findContours((f >= 0.3 * f.max()).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out[c] = [cc.reshape(-1, 2) for cc in cs if len(cc) > 2]
    return out


def tracked(m, d):
    f = os.path.join(B, "analysis", "crossday", f"{m}_pairs.csv")
    if not os.path.isfile(f):
        return set()
    P = pd.read_csv(f)
    return set(P[f"day{d}_id"].astype(int))


# ------------------------------------------------------------------------------------------------ figures
def _rc():
    """this script's style again before each figure (another module may have changed the global fonts)"""
    plt.rcParams.update(RC)


def fig_fov(m, d, ids, out):
    _rc()
    S, R, img = fov(m, d)
    C = contours(S, ids)
    tr = tracked(m, d)
    pos = R.set_index("id")
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.6))                  # 10-24: shown at 93 %: numbers 10 pt -> 9.3 pt
    for a in ax:
        a.imshow(img, cmap="gray"); a.set_axis_off()
    ax[0].set_title(f"{m} Day {d}: max dF/F projection", fontsize=13)
    one = "#f012be" if d == 1 else "#00d0ff"
    for c in ids:
        col = "#2ecc40" if c in tr else one
        for cc in C.get(c, []):
            ax[1].plot(cc[:, 0], cc[:, 1], color=col, lw=1.0)
        if c in pos.index:
            ax[1].text(pos.at[c, "col"] + 5, pos.at[c, "row"] - 5, str(c), color=col, fontsize=10, fontweight="bold")
    ax[1].set_title(f"{len(ids)} neurons (number = neuron): green = on both days ({len(set(ids) & tr)}),\n"
                    f"{'magenta' if d == 1 else 'cyan'} = Day {d} only ({len(set(ids) - tr)}); same background", fontsize=12.5)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return len(set(ids) & tr), len(set(ids) - tr)


def fig_traces(s, cls_any, m, d, ph, out):
    """10-24: three columns of traces (two columns needed 6.5-pt numbers; now >= 9 pt on the slide), event rows labelled once,
    a key for the trace colours and the orange blocks"""
    _rc()
    from matplotlib.lines import Line2D
    T = s["T"]
    ids = T["cell_ids"].astype(int); t = T["t_s"]; Z = T["z"]
    k = 6
    n5 = len(t) // k
    t5 = t[:n5 * k].reshape(n5, k).mean(1)
    Z5 = np.nanmean(Z[:n5 * k].reshape(n5, k, -1), 1)
    ncol = 3
    per = int(np.ceil(len(ids) / ncol))
    fig = plt.figure(figsize=(12.4, 6.0))                              # shown at 87 %: 10.5 pt -> 9.1 pt
    rows = STIM + ["withdrawal", "flinch", "movement onset"]
    for col in range(ncol):
        sel = np.arange(col * per, min((col + 1) * per, len(ids)))
        x0 = 0.155 + col * 0.285
        ax = fig.add_axes([x0, 0.305, 0.25, 0.6])
        for i, j in enumerate(sel):
            z = Z5[:, j]
            lo, hi = np.nanmedian(z), np.nanpercentile(z, 99.5)
            c = {"UP": UPC, "DOWN": DNC}.get(cls_any.get(int(ids[j]), ""), NOC)
            ax.plot(t5, (z - lo) / (hi - lo + 1e-9) * 0.85 + (per - i), lw=0.4, color=c)
            ax.text(-12, per - i + 0.3, str(ids[j]), ha="right", va="center", fontsize=10.5, color=c)
        ax.set_xlim(0, t5[-1]); ax.set_ylim(0.3, per + 1.3); ax.set_yticks([])
        ax.tick_params(bottom=False, labelbottom=False)               # (set_xticks([]) also emptied the shared raster axis)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        for b0, b1 in [(180, 360), (390, 570), (600, 780), (810, 990)]:
            ax.axvspan(b0, b1, color="#f5b041", alpha=0.12, lw=0)
        rb = fig.add_axes([x0, 0.085, 0.25, 0.2], sharex=ax)
        for r_i, nm in enumerate(rows):
            f = np.asarray(s["B"]["ev"].get(nm, []))
            rb.plot(f / 30.0, np.full(len(f), len(rows) - 1 - r_i), "|", color=ECOL[nm], ms=6, mew=1.1)
        rb.set_yticks(range(len(rows)), [f"{SH[r]} ({len(s['B']['ev'].get(r, []))})" for r in rows][::-1] if col == 0 else [],
                      fontsize=10.5)
        rb.set_ylim(-0.7, len(rows) - 0.3)
        rb.tick_params(axis="x", labelsize=10.5)
        rb.set_xlabel("time in session (s)", fontsize=11)
    fig.text(0.01, 0.6, "neurons (1 line = 1 neuron, number on the left;\neach scaled to its own range)", rotation=90,
             ha="left", va="center", fontsize=11)
    fig.legend(handles=[Line2D([], [], color=UPC, lw=2, label="UP after >= 1 stimulus"),
                        Line2D([], [], color=DNC, lw=2, label="only DOWN"), Line2D([], [], color=NOC, lw=2, label="no response"),
                        Patch(color="#f5b041", alpha=0.35, label="stimulus blocks"),
                        Line2D([], [], ls="none", marker="|", color="k", ms=10, mew=1.5, label="event (count)")],
               loc="upper center", ncol=5, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 1.0), handlelength=1.6,
               columnspacing=1.4)
    fig.savefig(out, dpi=150); plt.close(fig)
    v = np.array([cls_any.get(int(c), "") for c in ids])
    return int((v == "UP").sum()), int((v == "DOWN").sum()), int(len(ids))


def fig_maps(m, d, evs, CL, out):
    """10-24: readable titles and key (outline style), 'of n neurons', untested panels greyed with the reason; the 7-pt neuron
    numbers on the outlines are left out (they could not be made >= 9 pt without covering the map; the numbered map is the
    first slide of each mouse / day and the neuron x event matrix)"""
    _rc()
    from matplotlib.lines import Line2D
    S, R, img = fov(m, d)
    fig, ax = plt.subplots(2, len(evs), figsize=(3.4 * len(evs) + 0.2, 6.6), squeeze=False)   # shown at 79 %
    counts = {}
    for a in ax.ravel():
        a.set_anchor("N")
    for r_i, ph in enumerate(("pre", "post")):
        for e_i, ev in enumerate(evs):
            a = ax[r_i, e_i]; a.set_axis_off()
            cl = CL.get((ph, ev))
            if cl is None:
                a.imshow(img, cmap="gray", alpha=0.25)
                a.text(0.5, 0.5, f"fewer than {CA.MIN_TRIALS}\nevents: not tested", transform=a.transAxes, ha="center",
                       va="center", fontsize=12, color="0.2")
                a.set_title(f"{SH[ev]} - {PH[ph]}\nnot tested", fontsize=12, color="0.5"); continue
            a.imshow(img, cmap="gray", alpha=0.9)
            C = contours(S, list(cl.keys()))
            for c, k in cl.items():
                col = {"UP": UPC, "DOWN": DNC}.get(k, NOC_MAP)
                for cc in C.get(c, []):
                    a.plot(cc[:, 0], cc[:, 1], color=col, lw=1.6 if k in ("UP", "DOWN") else 0.9)
            nu = sum(v == "UP" for v in cl.values()); nd = sum(v == "DOWN" for v in cl.values())
            counts[(ph, ev)] = (nu, nd, len(cl))
            a.set_title(f"{SH[ev]} - {PH[ph]}\n{nu} UP, {nd} DOWN of {len(cl)} neurons", fontsize=12,
                        color=ECOL[ev] if r_i == 0 else "k")
    oc = lambda c, w, lab: Line2D([], [], ls="none", marker="o", ms=12, mfc="none", mec=c, mew=w, label=lab)
    fig.legend(handles=[oc(UPC, 2.2, "UP (rose)"), oc(DNC, 2.2, "DOWN (fell)"), oc("#b0b0b0", 1.4, "no response"),
                        Patch(facecolor="0.25", edgecolor="0.25", label="background = max dF/F projection")],
               loc="upper center", ncol=4, frameon=False, fontsize=12, columnspacing=1.3, handletextpad=0.4)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=1 - 1.15 / 6.6, wspace=0.05, hspace=0.32)
    fig.savefig(out, dpi=130); plt.close(fig)
    return counts


def band_key(fig, fs=11.5, extra=()):
    """the key of the shaded windows and the event line (10-24: they were explained only on the slide)"""
    from matplotlib.lines import Line2D
    fig.legend(handles=list(extra) + [Patch(color="0.88", label="response window"), Patch(color="#dbe9f6", label="baseline window"),
                                      Line2D([], [], color="k", lw=0.8, label="event onset (0 s)")],
               loc="upper center", ncol=len(extra) + 3, frameon=False, fontsize=fs, bbox_to_anchor=(0.5, 1.0), handlelength=1.6,
               columnspacing=1.3)


def fig_groups(PSN, CL, evs, out):
    """10-24: n of each group in the panel title (the in-plot legends sat on the traces), one key on top, >= 9 pt on the slide"""
    _rc()
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(2, len(evs), figsize=(3.3 * len(evs) + 0.8, 6.4), squeeze=False, sharex=True)   # shown at 81 %
    best = {}
    for r_i, ph in enumerate(("pre", "post")):
        for e_i, ev in enumerate(evs):
            a = ax[r_i, e_i]
            P_, cl = PSN.get((ph, ev)), CL.get((ph, ev))
            hd = f"{SH[ev]}\n{wtxt(ev)}\n" if r_i == 0 else ""
            if P_ is None or cl is None:
                a.set_axis_off()
                a.set_title(hd + "not tested", fontsize=11.5, color=ECOL[ev] if r_i == 0 else "0.4")
                a.text(0.5, 0.5, f"fewer than {CA.MIN_TRIALS} events", transform=a.transAxes, ha="center", va="center",
                       fontsize=11.5, color="0.4")
                continue
            mu, tt, ids = P_
            D = CA.sm(disp(mu, tt))
            show = (tt >= -3) & (tt <= 6)
            lab = np.array([cl.get(int(c), "no response") for c in ids])
            nn = {}
            for k, col in (("no response", NOC), ("DOWN", DNC), ("UP", UPC)):
                sel = lab == k
                nn[k] = int(sel.sum())
                if not sel.any():
                    continue
                y = np.nanmean(D[:, sel], 1); se = np.nanstd(D[:, sel], 1) / np.sqrt(max(sel.sum(), 1))
                a.fill_between(tt[show], (y - se)[show], (y + se)[show], color=col, alpha=0.25, lw=0)
                a.plot(tt[show], y[show], color=col, lw=1.6)
                if k == "UP":
                    best[(ph, ev)] = float(np.nanmean(wresp(mu[:, sel], tt, ev)))
            (p0, p1), (a0, b0) = wwin(ev)
            a.axvspan(a0, b0, color="0.88", zorder=0, lw=0); a.axvspan(p0, p1, color="#dbe9f6", zorder=0, lw=0)
            a.axvline(0, color="k", lw=0.7); a.axhline(0, color="0.6", lw=0.5)
            a.set_title(hd + f"n: UP {nn['UP']}, DOWN {nn['DOWN']}, rest {nn['no response']}", fontsize=11.5,
                        color=ECOL[ev] if r_i == 0 else "0.15")
            a.tick_params(labelsize=11.5, labelbottom=True)
            if e_i == 0:
                a.set_ylabel(f"{PH[ph]}", fontsize=12)
            if r_i == 1:
                a.set_xlabel("time from event (s)", fontsize=11.5)
    fig.supylabel(ZLAB, fontsize=11.5, x=0.005)
    fig.tight_layout(rect=(0.02, 0, 1, 1 - 0.4 / 6.4), h_pad=0.8)
    band_key(fig, extra=[Line2D([], [], color=UPC, lw=2, label="UP"), Line2D([], [], color=DNC, lw=2, label="DOWN"),
                         Line2D([], [], color=NOC, lw=2, label="rest (mean +- SEM)")])
    fig.savefig(out, dpi=140); plt.close(fig)
    return best


def fig_population(PSN, evs, col1, out):
    """10-24: one key on top (n of neurons, windows, event line), >= 9 pt on the slide; the star key at the bottom"""
    _rc()
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(2, len(evs), figsize=(3.3 * len(evs) + 0.8, 6.4), squeeze=False, gridspec_kw=dict(height_ratios=[1.2, 1]))
    res = {}
    nlab = {}
    for e_i, ev in enumerate(evs):
        a, b = ax[0, e_i], ax[1, e_i]
        r = {}
        for ph, col in (("pre", "0.45"), ("post", col1)):
            P_ = PSN.get((ph, ev))
            if P_ is None:
                continue
            mu, tt, ids = P_
            D = CA.sm(disp(mu, tt)); show = (tt >= -3) & (tt <= 6)
            y = np.nanmean(D, 1); se = np.nanstd(D, 1) / np.sqrt(D.shape[1])
            a.fill_between(tt[show], (y - se)[show], (y + se)[show], color=col, alpha=0.25, lw=0)
            a.plot(tt[show], y[show], color=col, lw=1.6)
            nlab[ph] = D.shape[1]
            r[ph] = dict(zip(ids.astype(int), wresp(mu, tt, ev)))
        (p0, p1), (a0, b0) = wwin(ev)
        a.axvspan(a0, b0, color="0.88", zorder=0, lw=0); a.axvspan(p0, p1, color="#dbe9f6", zorder=0, lw=0)
        a.axvline(0, color="k", lw=0.7); a.axhline(0, color="0.6", lw=0.5)
        a.set_title(f"{SH[ev]}\n{wtxt(ev)}", fontsize=12, color=ECOL[ev]); a.set_xlabel("time from event (s)", fontsize=11.5)
        a.tick_params(labelsize=11.5); b.tick_params(labelsize=11.5)
        if not r:
            a.text(0.5, 0.5, f"fewer than {CA.MIN_TRIALS} events", transform=a.transAxes, ha="center", va="center", fontsize=11.5,
                   color="0.4")
        if e_i == 0:
            a.set_ylabel("all neurons, mean +- SEM\nz, minus mean of 3 s before", fontsize=11.5)
        pv = {}
        for i, (ph, col) in enumerate((("pre", "0.45"), ("post", col1))):
            if ph not in r:
                continue
            v = np.array(list(r[ph].values())); v = v[np.isfinite(v)]
            pv[ph] = wilcoxon(v).pvalue if len(v) >= 5 and np.any(v != 0) else np.nan
            b.bar(i, np.mean(v), yerr=np.std(v) / np.sqrt(len(v)), color=col, width=0.6, capsize=3)
            PS.vs_ref(b, i, 0.98, pv[ph], "vs 0", "k", fs=11, axes_y=True, va="top")
        pp = np.nan
        if "pre" in r and "post" in r:
            common = sorted(set(r["pre"]) & set(r["post"]))
            dd = np.array([r["post"][c] - r["pre"][c] for c in common]); dd = dd[np.isfinite(dd)]
            pp = wilcoxon(dd).pvalue if len(dd) >= 5 and np.any(dd != 0) else np.nan
        lo_, hi_ = b.get_ylim(); b.set_ylim(lo_, hi_ + 0.75 * (hi_ - lo_))      # 10-24: room for 'vs 0' over the bracket
        if np.isfinite(pp):
            PS.bracket(b, 0, 1, hi_ + 0.08 * (hi_ - lo_), pp, "k", fs=13, inset=0.05)
        res[ev] = (pv.get("pre", np.nan), pv.get("post", np.nan), pp,
                   float(np.nanmean(list(r["pre"].values()))) if "pre" in r else np.nan,
                   float(np.nanmean(list(r["post"].values()))) if "post" in r else np.nan)
        b.axhline(0, color="0.4", lw=0.6)
        b.set_xticks([0, 1], ["base", "inj."])
        if e_i == 0:
            b.set_ylabel("response (z)\nbest window", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.045, 1, 1 - 0.4 / 6.4), h_pad=0.8)
    band_key(fig, extra=[Line2D([], [], color="0.45", lw=2, label=f"baseline ({nlab.get('pre', '?')} neurons)"),
                         Line2D([], [], color=col1, lw=2, label=f"after injection ({nlab.get('post', '?')})")])
    fig.text(0.01, 0.008, "* p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant (Wilcoxon over neurons)", fontsize=11.5,
             color="0.25", ha="left", va="bottom")
    fig.savefig(out, dpi=140); plt.close(fig)
    return res


def fig_counts(CL, nn, evs, out):
    _rc()
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.6), sharey=True)                 # shown at 93 %
    for k, ph in enumerate(("pre", "post")):
        a = ax[k]
        n = nn[ph]
        for y, ev in enumerate(evs[::-1]):
            cl = CL.get((ph, ev))
            if cl is None:
                a.text(n / 2, y, f"not tested (< {CA.MIN_TRIALS} events)", va="center", ha="center", fontsize=10.5, color="0.4")
                continue
            v = np.array(list(cl.values()))
            nd, nu = int((v == "DOWN").sum()), int((v == "UP").sum())
            a.barh(y, nd, color=DNC, height=0.62); a.barh(y, len(v) - nd - nu, left=nd, color="0.86", height=0.62)
            a.barh(y, nu, left=len(v) - nu, color="#e69f00", height=0.62)
            a.text(len(v) * 1.02, y, f"{nu} / {nd}", va="center", fontsize=10.5)
        a.set_yticks(range(len(evs)), [SH[e] for e in evs[::-1]], fontsize=11)
        a.set_xlim(0, n * 1.2); a.set_xlabel(f"neurons (n = {n})")
        a.set_title(f"{PH[ph]} session", fontsize=12.5, loc="left")
    fig.legend(handles=[Patch(color=DNC, label="DOWN"), Patch(color="0.86", label="no response"), Patch(color="#e69f00", label="UP"),
                        Patch(color="w", label="numbers right of each bar = n UP / n DOWN")],
               loc="upper center", ncol=4, frameon=False, fontsize=11, handlelength=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out, dpi=140); plt.close(fig)


def fig_matrix(CL, ids, evs, out):
    _rc()
    rows, lab = [], []
    for e in evs:
        for ph in ("pre", "post"):
            cl = CL.get((ph, e))
            rows.append([np.nan if cl is None or c not in cl else 1 if cl[c] == "UP" else -1 if cl[c] == "DOWN" else 0 for c in ids])
            lab.append(f"{SH[e]}  {'base' if ph == 'pre' else 'inj.'}")
    M = np.array(rows, float)
    fig = plt.figure(figsize=(12.4, 5.6))                               # 10-24: shown at 93 % (was 79 %): >= 9 pt
    ax = fig.add_axes([0.16, 0.15, 0.74, 0.76])
    ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=matplotlib.colors.ListedColormap([DNC, "#f2f2f2", UPC]), vmin=-1.5, vmax=1.5,
              interpolation="nearest")
    ax.set_yticks(range(len(lab)), lab, fontsize=10)
    for i in range(2, len(lab), 2):
        ax.axhline(i - 0.5, color="k", lw=0.6)
    step = max(1, int(np.ceil(len(ids) / 36)))
    ax.set_xticks(range(0, len(ids), step), [str(c) for c in ids[::step]], fontsize=10, rotation=90)
    ax.set_xlabel("neuron number (as on the field-of-view map" + (f", every {step}{({2: 'nd', 3: 'rd'}).get(step, 'th')} labelled)" if step > 1 else ")"), fontsize=11)
    ax3 = fig.add_axes([0.915, 0.15, 0.075, 0.76], sharey=ax)
    ax3.barh(range(len(lab)), np.nansum(M == 1, 1), color=UPC, height=0.8)
    ax3.barh(range(len(lab)), -np.nansum(M == -1, 1), color=DNC, height=0.8)
    ax3.axvline(0, color="k", lw=0.5); ax3.tick_params(labelsize=10); ax3.tick_params(axis="y", left=False, labelleft=False)  # (set_yticks([]) emptied the shared row labels)
    ax3.set_xlabel("n DOWN | UP", fontsize=10.5)
    fig.legend(handles=[Patch(color=UPC, label="UP"), Patch(color=DNC, label="DOWN"), Patch(color="#f2f2f2", label="no response"),
                        Patch(facecolor="w", edgecolor="0.6", label=f"not tested (< {CA.MIN_TRIALS} events)")],
               loc="upper center", ncol=4, frameon=False, fontsize=11, handlelength=1.4)
    fig.savefig(out, dpi=140); plt.close(fig)
    base = M[0::2]
    return int((np.nansum(base == 1, 0) >= 2).sum())


def fig_fate(F, evs, col, out):
    _rc()
    X = F[F.event.isin(evs)]
    evs = [e for e in evs if (X.event == e).any()]
    if not evs:
        return None
    fig, ax = plt.subplots(2, len(evs), figsize=(2.3 * len(evs) + 0.9, 6.0), squeeze=False, sharey="row")   # 10-24: >= 9 pt
    res = {}
    for r_i, g in enumerate(("activated", "inhibited")):
        for e_i, e in enumerate(evs):
            a = ax[r_i, e_i]
            Z = X[(X.event == e) & (X.group == g)]
            if len(Z):
                b0, b1 = Z.base_heldout.values, Z.inj.values
                a.bar(0, b0.mean(), color="white", edgecolor=col, width=0.7, lw=1.4)
                a.bar(1, b1.mean(), color=col, width=0.7)
                for u, v in zip(b0, b1):
                    a.plot([0, 1], [u, v], color="0.6", lw=0.5, zorder=3)
                p = wilcoxon(b1 - b0).pvalue if len(Z) >= 5 and np.any(b1 != b0) else np.nan
                res[(g, e)] = (len(Z), b0.mean(), b1.mean(), p)
                if np.isfinite(p):
                    lo_, hi_ = a.get_ylim()
                    PS.bracket(a, 0, 1, hi_ * 0.98 if g == "activated" else lo_ * 0.98, p, "k", fs=12.5, inset=0.05,
                               below=g != "activated")
            a.set_title(f"{SH[e]} (n = {len(Z)})", fontsize=11.5)
            a.axhline(0, color="0.4", lw=0.6); a.set_xticks([0, 1], ["base", "inj."])
            a.tick_params(labelsize=11.5)
            if e_i == 0:
                a.set_ylabel(f"{g} at baseline\nresponse (z)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
    fig.legend(handles=[Patch(facecolor="w", edgecolor=col, lw=1.4, label="baseline, held-out (even) trials"),
                        Patch(color=col, label="after the injection"),
                        matplotlib.lines.Line2D([], [], color="0.6", lw=1, label="one neuron")],
               loc="upper center", ncol=3, frameon=False, fontsize=11.5, handlelength=1.4)
    fig.text(0.01, 0.008, "* p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant (paired Wilcoxon)", fontsize=11,
             color="0.25", ha="left", va="bottom")
    fig.savefig(out, dpi=140); plt.close(fig)
    return res


# ------------------------------------------------------------------------------------------------ slides
def build(dk, lg, days, od, short=False):
    L = lambda en, kr: kr if lg == "kr" else en
    js = os.path.join(EA, "summary.json")
    for ev, w in json.load(open(js, encoding="utf-8")).get("windows", {}).items():
        WIN[ev] = tuple(float(x) for x in w["1"])
    T = load_classes()
    F = pd.read_csv(os.path.join(EA, "P5_fate.csv")); F["mouse"] = F.mouse.astype(str)
    f = lambda nm: os.path.join(od, nm)
    os.makedirs(od, exist_ok=True)
    import winwords as WW
    note_win = L(WW.prefix_en()[0].upper() + WW.prefix_en()[1:] + ": " + WW.en() + ". "
                 "UP / DOWN = within-block time-shift test per neuron, 1000 shuffles, BH q <= 0.20 per session (event analysis). "
                 "Traces: z (dF/F / robust SD) minus the 3 s before the event, display only.",
                 WW.prefix_kr() + ": " + WW.kr() + ". UP / DOWN = 뉴런별 블록 내 시간 이동 검정, 1000회, "
                 "세션별 BH q <= 0.20 (이벤트 분석). trace = 사건 전 3 s 기준 z (표시용).")
    n = 0
    for m in MO:
        for d in days:
            drug = ARMS[d][m]
            sec = f"{L('PER MOUSE', '마우스별')}: {m} ({SEX[m]}) Day {d}, {drug}"
            S = {ph: session(m, d, ph) for ph in ("pre", "post")}
            if S["pre"] is None or S["post"] is None:
                continue
            ids = S["pre"]["T"]["cell_ids"].astype(int)
            X = T[(T.mouse == m) & (T.day == d)]
            CL = {(ph, ev): dict(zip(g.cell.astype(int), g.cls_shift)) for (ph, ev), g in X.groupby(["phase", "event"])}
            PSN = {}
            for ph in ("pre", "post"):
                for ev in ALL:
                    r = psth(S[ph], ev)
                    if r is not None:
                        PSN[(ph, ev)] = (r[0], r[1], S[ph]["T"]["cell_ids"].astype(int))
            nb, no = fig_fov(m, d, ids, f(f"{m}_d{d}_01_fov.png"))
            dk.slide(sec, L(f"{m} Day {d}: every neuron, labelled", f"{m} Day {d}: 모든 뉴런, 번호 표시"),
                     L(f"{len(ids)} neurons: {nb} also on the other day, {no} on Day {d} only",
                       f"뉴런 {len(ids)}개: 다른 날에도 있음 {nb}, Day {d}만 {no}"),
                     f(f"{m}_d{d}_01_fov.png"),
                     L("Left: the whole-day max projection; right: every analysed neuron outlined and numbered (the same numbers on "
                       "every slide of this mouse and day).", "왼쪽: 하루 전체 max projection; 오른쪽: 분석한 모든 뉴런 윤곽과 번호 (이 마우스·날의 "
                       "모든 슬라이드에서 같은 번호)."),
                     L("Green = followed on both days (cross-day tracking you confirmed), magenta / cyan = one day only.",
                       "초록 = 두 날 추적 (확인한 매칭), 자홍 / 청록 = 하루만."))
            n += 1
            for ph in ("pre", "post"):
                anyc = {}
                for ev in STIM:
                    for c, k in CL.get((ph, ev), {}).items():
                        if k == "UP":
                            anyc[c] = "UP"
                        elif k == "DOWN" and anyc.get(c) != "UP":
                            anyc[c] = "DOWN"
                nu, nd, nall = fig_traces(S[ph], anyc, m, d, ph, f(f"{m}_d{d}_02_traces_{ph}.png"))
                dk.slide(sec, L(f"{m} Day {d} {PH[ph]}: all neurons with the scored events", f"{m} Day {d} {PH[ph]}: 모든 뉴런과 채점 사건"),
                         L(f"{nu} neurons rose after at least one stimulus (red), {nd} only fell (blue), of {nall}",
                           f"{nall}개 중 {nu}개가 자극 후 상승(빨강), {nd}개는 하강만(파랑)"),
                         f(f"{m}_d{d}_02_traces_{ph}.png"),
                         L("One line per neuron (number on the left), below on the same time axis every scored delivery, reflex and "
                           "movement onset (count in brackets). Orange = stimulus blocks.",
                           "뉴런당 한 선 (왼쪽 번호), 아래는 같은 시간축의 모든 자극, 반사, 움직임 시작 (괄호 = 수). 주황 = 자극 블록."),
                         note_win)
                n += 1
            col1 = CC[drug]
            for gi, (gen, gkr, evs) in enumerate(GROUPS, 1):
                cnt = fig_maps(m, d, evs, CL, f(f"{m}_d{d}_03_maps_{gi}.png"))
                dk.slide(sec, L(f"{m} Day {d}: where the responsive neurons are ({gen})", f"{m} Day {d}: 반응 뉴런의 위치 ({gkr})"),
                         L("UP after the injection: " + ", ".join(f"{SH[e]} {cnt[('post', e)][0]}" for e in evs if ('post', e) in cnt)
                           + " (baseline: " + ", ".join(f"{cnt[('pre', e)][0]}" for e in evs if ('pre', e) in cnt) + ")",
                           "주사 후 UP: " + ", ".join(f"{SH[e]} {cnt[('post', e)][0]}" for e in evs if ('post', e) in cnt)
                           + " (기준: " + ", ".join(f"{cnt[('pre', e)][0]}" for e in evs if ('pre', e) in cnt) + ")"),
                         f(f"{m}_d{d}_03_maps_{gi}.png"),
                         L(f"Every neuron on the max projection, coloured by its test result (red = UP, blue = DOWN); top = baseline, "
                           f"bottom = after {drug}. One mouse, {len(ids)} neurons.",
                           f"max projection 위 모든 뉴런, 검정 결과 색 (빨강 = UP, 파랑 = DOWN); 위 = 기준, 아래 = {drug} 후. 한 마리, 뉴런 {len(ids)}."),
                         note_win)
                best = fig_groups(PSN, CL, evs, f(f"{m}_d{d}_04_groups_{gi}.png"))
                bb = sorted(((v, ph, e) for (ph, e), v in best.items()), reverse=True)[:2]
                dk.slide(sec, L(f"{m} Day {d}: UP, DOWN and the other neurons ({gen})", f"{m} Day {d}: UP, DOWN, 나머지 뉴런 ({gkr})"),
                         L(("Largest UP responses: " + ", ".join(f"{SH[e]} {PH[ph]} {v:+.2f} z" for v, ph, e in bb)) if bb else "No UP neurons here",
                           ("가장 큰 UP 반응: " + ", ".join(f"{SH[e]} {PH[ph]} {v:+.2f} z" for v, ph, e in bb)) if bb else "UP 뉴런 없음"),
                         f(f"{m}_d{d}_04_groups_{gi}.png"),
                         L("Mean +- SEM of the UP (red), DOWN (blue) and other (grey) neurons, n in brackets; grey band = response "
                           "window, blue band = baseline window.", "UP(빨강), DOWN(파랑), 나머지(회색) 뉴런의 평균 +- SEM, 괄호 = n; 회색 = 반응 창, "
                           "파랑 = 기준 창."),
                         note_win)
                res = fig_population(PSN, evs, col1, f(f"{m}_d{d}_05_population_{gi}.png"))
                chg = [f"{SH[e]} ({pstr(r[2])})" for e, r in res.items() if np.isfinite(r[2]) and r[2] < 0.05]
                dk.slide(sec, L(f"{m} Day {d}: population response, baseline vs after {drug} ({gen})",
                                f"{m} Day {d}: 집단 반응, 기준 vs {drug} 후 ({gkr})"),
                         L(("Baseline vs injection differs for: " + ", ".join(chg)) if chg else "No event's population response differs between the sessions",
                           ("기준 vs 주사가 다름: " + ", ".join(chg)) if chg else "두 세션의 집단 반응이 다른 사건 없음"),
                         f(f"{m}_d{d}_05_population_{gi}.png"),
                         L(f"Top: all {len(ids)} neurons around each event (grey = baseline, colour = after {drug}); bottom: the response "
                           "in the best window per session. Wilcoxon over neurons: each bar vs 0, bracket = baseline vs injection (paired).",
                           f"위: 모든 뉴런 {len(ids)}개의 사건 주변 평균 (회색 = 기준, 색 = {drug} 후); 아래: 최적 창 반응. 뉴런 Wilcoxon: 막대 vs 0, "
                           "괄호 = 기준 vs 주사 (짝)."),
                         note_win)
                n += 3
            nn = {ph: len(S[ph]["T"]["cell_ids"]) for ph in ("pre", "post")}
            fig_counts(CL, nn, ALL, f(f"{m}_d{d}_06_counts.png"))
            nup = {ph: sum(sum(v == "UP" for v in CL.get((ph, e), {}).values()) for e in ALL) for ph in ("pre", "post")}
            dk.slide(sec, L(f"{m} Day {d}: how many neurons go up or down, per event", f"{m} Day {d}: 사건별 상승·하강 뉴런 수"),
                     L(f"UP classifications summed over the 10 events: baseline {nup['pre']}, after {drug} {nup['post']}",
                       f"10개 사건의 UP 분류 합: 기준 {nup['pre']}, {drug} 후 {nup['post']}"),
                     f(f"{m}_d{d}_06_counts.png"),
                     L(f"For every event: neurons that fell (blue), did not change (grey) or rose (orange). n = {nn['pre']} neurons.",
                       f"사건마다 하강(파랑), 변화 없음(회색), 상승(주황) 뉴런. 뉴런 n = {nn['pre']}."),
                     note_win)
            multi = fig_matrix(CL, list(ids), ALL, f(f"{m}_d{d}_07_matrix.png"))
            dk.slide(sec, L(f"{m} Day {d}: which neuron responds to which event", f"{m} Day {d}: 어떤 뉴런이 어떤 사건에 반응하나"),
                     L(f"At baseline {multi} of {len(ids)} neurons rose after two or more different events",
                       f"기준에서 뉴런 {len(ids)}개 중 {multi}개가 두 개 이상의 사건 후 상승"),
                     f(f"{m}_d{d}_07_matrix.png"),
                     L("Columns = neurons (numbers as on the map), rows = event x session; red = UP, blue = DOWN, light grey = no "
                       "response, white = not tested (< 5 events). Right: neurons per row.",
                       "열 = 뉴런 (지도 번호), 행 = 사건 x 세션; 빨강 = UP, 파랑 = DOWN, 연회색 = 반응 없음, 흰색 = 검정 안 함 (< 5번). 오른쪽: 행별 수."),
                     note_win)
            n += 2
            Fm = F[(F.mouse == m) & (F.day == d)]
            for gi, (gen, gkr, evs) in enumerate((("stimuli", "자극", STIM), ("behaviours", "행동", ALL[4:])), 1):
                rc = fig_fate(Fm, evs, col1, f(f"{m}_d{d}_08_fate_{gi}.png"))
                if rc is None:
                    continue
                sg = [f"{SH[e]} {g} {b0:+.2f} -> {b1:+.2f} z ({pstr(p)})" for (g, e), (n_, b0, b1, p) in rc.items() if np.isfinite(p) and p < 0.05]
                dk.slide(sec, L(f"{m} Day {d}: responsive neurons after {drug} ({gen})", f"{m} Day {d}: {drug} 후 반응 뉴런 ({gkr})"),
                         L(("Changed: " + "; ".join(sg)) if sg else "No group of responsive neurons changed (paired Wilcoxon p < 0.05)",
                           ("변화: " + "; ".join(sg)) if sg else "변한 반응 뉴런 그룹 없음 (짝 Wilcoxon p < 0.05)"),
                         f(f"{m}_d{d}_08_fate_{gi}.png"),
                         L("Neurons activated (top) or inhibited (bottom) on the ODD baseline trials; open bar = their EVEN baseline trials, "
                           "filled = after the injection; grey line = one neuron. Paired Wilcoxon; n in each title.",
                           "홀수 기준 trial에서 활성화(위)·억제(아래)된 뉴런; 빈 막대 = 짝수 기준 trial, 채움 = 주사 후; 회색 선 = 뉴런. 짝 Wilcoxon; "
                           "제목에 n."),
                         note_win)
                n += 1
            print(f"{m} d{d}: done", flush=True)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=["en", "kr"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", nargs="+", type=int, default=[1, 2])
    a = ap.parse_args(argv)
    dk = ND.Deck(a.lang, "")
    dk.n_txt = "n = 1 mouse per slide" if a.lang != "kr" else "n = 슬라이드당 1마리"
    n = build(dk, a.lang, a.days, os.path.splitext(a.out)[0] + "_figures")
    dk.prs.save(a.out)
    print("->", a.out, n, "slides")


if __name__ == "__main__":
    main()
