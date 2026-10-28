"""make_day1_full_deck.py - the Day-1 mini1p deck from the beginning (PI rules), in four parts:

  SUMMARY    1  what we found (four numbers)
  IMAGING    2  design      3  from movie to neurons (best case, 5941)      4  neurons per mouse
  NEURONS    5  whole-session heatmap (5941)   6  example session (video frame + traces)
             7  every neuron's response to each event (heatmap)   8  how many respond (strict) + population
             9  stimulus-block activity per mouse   10  SBI-553 vs vehicle
  BEHAVIOUR  11 what we score (examples)   12 behaviour per delivery, every mouse   13 per stimulus + spontaneous movement
  LIMITS     14 limits of the current data

Rules (PI, 2026-10-16): few polished slides; every graph with a title, labelled axes, legend, n per
group and the statistical test; a title at the top and one or two sentences at the bottom. Every
number is read from the analysis files; the build stops if the analysed cells differ from the
confirmed selection. Behaviour is always normalised by the number of deliveries (they differ
between sessions: you, 2026-10-21): events per delivery = all events in the response windows /
deliveries, compared per mouse with an exact Poisson rate test.

  python make_day1_full_deck.py --base C:/Users/hsollim/Documents/HEAL_mini1p_SBI553 --out <pptx>
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import binomtest, beta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
import make_day1_ca_slides as S1                                    # noqa: E402
import deck_cfg as C                                                # noqa: E402  (CA_DAY: which day)
import pairstars as PS  # noqa: E402  (user rule 10-22: every star shows the pair it compares)
import ca_day1_analysis as A                                        # noqa: E402
import behavior_day_analysis as BA                                  # noqa: E402

MICE, ARMC, STIM, FW, FH, X0 = S1.MICE, S1.ARMC, S1.STIM, S1.FW, S1.FH, S1.X0
ARM = {m: a for m, a, _ in MICE}
SCOL = {"Light touch": "#5dade2", "Mild touch": "#f5b041", "Pin prick": "#e74c3c", "Heat": "#8e44ad",
        "withdrawal": "#6c3483", "movement onset": "#17a2b8"}
EVS = STIM + ["withdrawal", "movement onset"]
SCORE = C.SCORE
clean = S1.clean


def FD_REVIEW(B, m):
    return mpath(B, f"{m}{C.T}", "4_cell_review", f"{m}{C.T}_cell_review.csv")


def mpath(B, *p):
    return os.path.join(B, "mini1p_out", C.DATE, *p)


def norm01(x, lo=1, hi=99.7, g=1.0):
    a, b = np.nanpercentile(x, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-12), 0, 1) ** g


def outlines(ax, S, ids, col, lw):
    for j in ids:
        f = S[:, :, j]
        if f.max() > 0:
            ax.contour((f >= 0.3 * f.max()).astype(float), levels=[0.5], colors=[col], linewidths=lw)


# -------------------------------------------------------------------------------------- IMAGING
def fig_design2(SS, nk, out):
    # 10-25 (figure fix round 2): methods text removed from the figure (it is in the slide text / notes); the timeline uses the
    # full width; crossover wording ('<day> drug', swapped on the other day) instead of 'n = 2 per group'; the table is
    # narrower than its axes so its right border is not cut off at the image edge; fonts >= 10 pt (the PNG is shown at 100 %)
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.035, 0.74, 0.93, 0.10])
    blocks = [("no stimulus", 0, 180, "#bbbbbb")] + [(f"block {i + 1}", s0, s1, "#f5b041")
                                                    for i, (s0, s1) in enumerate([(180, 360), (390, 570), (600, 780), (810, 990)])]
    for nm, s0, s1, c in blocks:
        ax.barh(0, (s1 - s0) / 60, left=s0 / 60, height=0.8, color=c, edgecolor="k", lw=0.5)
        ax.text((s0 + s1) / 120, 0, nm, ha="center", va="center", fontsize=10.5)
    ax.set_xlim(0, 17); ax.set_ylim(-0.5, 0.5); ax.set_yticks([])
    ax.set_xlabel("time in session (min)", fontsize=10.5); ax.tick_params(labelsize=10)
    ax.set_title("A  one session (16.5 min): 3 min without stimulus, then 4 stimulus blocks of 3 min, 30 s apart", fontsize=11.5, loc="left")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    rows = []
    for m, arm, sex in MICE:
        o0 = SS[(m, "pre")]["B"]["stim_order"]; o1 = SS[(m, "post")]["B"]["stim_order"]
        rows.append([m, sex, arm, str(nk[m][0]), "  >  ".join(o0), "  >  ".join(o1)])
    tb = fig.add_axes([0.035, 0.03, 0.93, 0.50]); tb.set_axis_off()
    swap = "swapped on Day 2" if C.DAY == 1 else "swapped from Day 1"
    tb.set_title(f"B  the four mice ({C.LABEL} drug; {swap}) and the order of the stimulus blocks", fontsize=11.5, loc="left")
    T = tb.table(cellText=rows, colLabels=["mouse", "sex", f"{C.LABEL} drug", "neurons", "baseline session: block 1 > 2 > 3 > 4",
                                           "after injection: block 1 > 2 > 3 > 4"], loc="upper left", cellLoc="left", colLoc="left",
                 colWidths=[0.07, 0.05, 0.10, 0.07, 0.345, 0.345])
    T.auto_set_font_size(False); T.set_fontsize(10.5); T.scale(1, 2.2)
    for (r, c), cell in T.get_celld().items():
        cell.set_edgecolor("0.85")
        if r == 0:
            cell.set_text_props(fontweight="bold"); cell.set_facecolor("#f3f5f8")
        elif c == 2:
            cell.get_text().set_color(ARMC[rows[r - 1][2]])
    fig.savefig(out, dpi=160); plt.close(fig)
    return ("Block order is randomised per mouse and session, so stimulus identity is not tied to the time in the session. "
            "The same field of view is imaged in both sessions; the neurons are detected once on the whole day.")


def fig_pipeline(B, out, m=C.EX):
    tag = f"{m}{C.T}"
    P = sio.loadmat(mpath(B, tag, "3_extract", f"{tag}_projections.mat"), variable_names=["mean_img", "Cn", "PNR", "max_dff"])
    S = np.asarray(sio.loadmat(mpath(B, tag, "3_extract", f"{tag}_extract.mat"), variable_names=["spatial_weights"])["spatial_weights"], np.float32)
    R = pd.read_csv(mpath(B, tag, "4_cell_review", f"{tag}_cell_review.csv"))
    att = C.one_day_flags(B, m, R)
    keep = np.flatnonzero(R.keep.values == 1)
    Sh = pd.read_csv(mpath(B, tag, "1_movie", f"{tag}_pre_mc_shifts.csv"))
    T = dict(np.load(mpath(B, tag, "5_analysis", "traces", f"{tag}_pre_traces.npz")))
    # 10-24 (figure review): 12.4 x 5.2 in = the deck's picture box, so the PNG is shown at ~100 % and every font is >= 10 pt;
    # taller trace panel (no overlapping traces), a y label and a dF/F scale bar per trace, labels clear of the lines
    fig = plt.figure(figsize=(FW, 5.2))
    xs, w, y0, h = [0.005, 0.255, 0.505, 0.755], 0.235, 0.47, 0.42
    mx = np.asarray(P["max_dff"], float)
    panels = [("1  motion-corrected mean image", norm01(np.asarray(P["mean_img"], float)), None),
              ("2  seed image: local correlation x PNR", norm01(np.asarray(P["Cn"], float) * np.asarray(P["PNR"], float), 1, 99.8, 0.6), None),
              (f"3  EXTRACT: {S.shape[2]} candidates", norm01(mx, 1, 99.8, 0.7), "all"),
              (f"4  review on the max dF/F: {len(keep)} kept", norm01(mx, 1, 99.8, 0.7), "kept")]
    for k, (ttl, img, ol) in enumerate(panels):
        a = fig.add_axes([xs[k], y0, w, h])
        a.imshow(img, cmap="gray"); a.set_axis_off()
        a.set_title(ttl, fontsize=11, loc="left")
        if ol == "all":
            outlines(a, S, range(S.shape[2]), "#00d0ff", 0.5)
        elif ol == "kept":
            outlines(a, S, [j for j in keep if not att.iloc[j]], "#2ecc40", 0.8)
            outlines(a, S, [j for j in keep if att.iloc[j]], C.ONE_COL, 0.8)
        if k < 3:
            fig.patches.append(FancyArrowPatch((xs[k] + w - 0.002, y0 + h / 2), (xs[k + 1] + 0.004, y0 + h / 2),
                               transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14, color="0.25", lw=1.4))
    fig.legend(handles=[Line2D([], [], color="#00d0ff", lw=1.5, label="EXTRACT candidate"),
                        Line2D([], [], color="#2ecc40", lw=1.5, label=C.GREEN_SHORT),
                        Line2D([], [], color=C.ONE_COL, lw=1.5, label=C.ORANGE_SHORT)],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=10.5)
    # motion
    a = fig.add_axes([0.05, 0.1, 0.4, 0.24])
    t = Sh.isx_frame.values / 30.0
    a.plot(t, Sh.x_shift_px, lw=0.5, color="#1f77b4", label="x")
    a.plot(t, Sh.y_shift_px, lw=0.5, color="#d62728", label="y")
    a.set_xlim(0, t.max()); a.set_xlabel("time in the baseline session (s)", fontsize=10.5)
    a.set_ylabel("shift (px)", fontsize=10.5)
    a.tick_params(labelsize=10)
    a.set_title(f"5  motion (rigid): median |shift| {np.median(np.hypot(Sh.x_shift_px, Sh.y_shift_px)):.1f} px, "
                f"max {np.max(np.hypot(Sh.x_shift_px, Sh.y_shift_px)):.1f} px\n    frame-template r {Sh.corr_before.median():.2f} -> "
                f"{Sh.corr_after.median():.2f} (before -> after correction)", fontsize=10.5, loc="left")
    a.legend(fontsize=10, frameon=False, ncol=2, loc="upper left", handlelength=1.2)
    clean(a)
    # example traces: the 6 manually selected neurons with the highest SNR (clear transients), 120 s
    # around the first block start, 0.2-s running mean for display
    ids = T["cell_ids"].astype(int)
    man = [int(R.id.iloc[j]) for j in keep if not att.iloc[j]]
    Ev = pd.read_csv(mpath(B, tag, "5_analysis", "traces", f"{tag}_pre_events.csv"))
    w0, w1 = 150.0, 270.0
    ev = Ev[Ev.cell.isin(man) & (Ev.onset_s >= w0) & (Ev.onset_s < w1)]
    amp = ev.groupby("cell").amp.sum() if "amp" in ev else ev.groupby("cell").size()
    pick = [int(c) for c in amp.sort_values(ascending=False).index[:6]]
    a = fig.add_axes([0.6, 0.09, 0.33, 0.3])
    tt = T["t_s"]; mm = (tt >= w0) & (tt <= w1)
    H_ = 0.8                                                     # trace height in a slot of 1: a clear gap between traces
    for i, c in enumerate(pick):
        j = int(np.where(ids == c)[0][0])
        d = pd.Series(T["dff"][mm, j]).rolling(6, center=True, min_periods=1).mean().values
        lo_, hi_ = np.nanpercentile(d, 10), np.nanmax(d)
        d = (d - lo_) / (hi_ - lo_ + 1e-9)
        yb = len(pick) - i
        a.plot(tt[mm], d * H_ + yb, lw=0.7, color="k")
        a.text(w0 - 2, yb + H_ * 0.35, f"#{c}", ha="right", va="center", fontsize=10)          # left of the trace start
        a.plot([w1 + 2, w1 + 2], [yb, yb + H_], color="#c0392b", lw=2, clip_on=False, solid_capstyle="butt")
        a.text(w1 + 4, yb + H_ / 2, f"{100 * (hi_ - lo_):.1f} %", ha="left", va="center", fontsize=10, color="#c0392b")
    a.axvspan(180, 270, color="#f5b041", alpha=0.18, lw=0)
    a.set_xlim(150, 270); a.set_yticks([]); a.set_ylim(0.9, len(pick) + 0.9)
    a.set_ylabel("neuron (dF/F, scaled\nper neuron)", fontsize=10.5, labelpad=34)
    a.set_xlabel("time in the baseline session (s)", fontsize=10.5); a.tick_params(labelsize=10)
    a.set_title("6  the 6 most active selected neurons here\n    red bar = its dF/F range; orange = stimulus block 1", fontsize=10.5,
                loc="left")
    for sp in ("top", "right", "left"):
        a.spines[sp].set_visible(False)
    fig.savefig(out, dpi=160); plt.close(fig)
    nc = int((R.keep == 1).sum())
    return (f"{m} ({ARM[m]}, {C.SEX[m]}), " + ("best field of view" if C.DAY == 1 else "example field of view (the same as on Day 1)")
            + ". 1-2: rigid motion correction (NoRMCorre), then a spatial high-pass and the "
            "correlation x peak-to-noise image that seeds EXTRACT. 3: EXTRACT (1-photon, seeded, no automatic elimination). "
            f"4: every candidate judged on the whole-day max dF/F projection: {nc} of {S.shape[2]} kept.")


def fig_neurons(B, out):
    fig = plt.figure(figsize=(FW, FH))
    pos = [(0.0, 0.5), (0.31, 0.5), (0.0, 0.0), (0.31, 0.0)]
    rows = []
    for k, (m, arm, sex) in enumerate(MICE):
        tag = f"{m}{C.T}"
        S = np.asarray(sio.loadmat(mpath(B, tag, "3_extract", f"{tag}_extract.mat"), variable_names=["spatial_weights"])["spatial_weights"], np.float32)
        R = pd.read_csv(mpath(B, tag, "4_cell_review", f"{tag}_cell_review.csv"))
        bg = np.asarray(sio.loadmat(mpath(B, tag, "3_extract", f"{tag}_projections.mat"), variable_names=["max_dff"])["max_dff"], float)
        att = C.one_day_flags(B, m, R)
        keep = np.flatnonzero(R.keep.values == 1)
        a = fig.add_axes([pos[k][0], pos[k][1], 0.30, 0.43])
        a.imshow(norm01(bg, 1, 99.8, 0.7), cmap="gray"); a.set_axis_off()
        outlines(a, S, [j for j in keep if not att.iloc[j]], "#2ecc40", 0.7)
        outlines(a, S, [j for j in keep if att.iloc[j]], C.ONE_COL, 0.7)
        nm, nr = int(sum(1 for j in keep if not att.iloc[j])), int(sum(1 for j in keep if att.iloc[j]))
        a.set_title(f"{m} ({C.LABEL}: {arm}, {sex}): {len(keep)} neurons", fontsize=10.5, color=ARMC[arm], pad=3)
        sh = []
        for ph in ("pre", "post"):
            X = pd.read_csv(mpath(B, tag, "1_movie", f"{tag}_{ph}_mc_shifts.csv"))
            sh.append(float(np.max(np.hypot(X.x_shift_px, X.y_shift_px))))
        rows.append(dict(mouse=m, arm=arm, cand=S.shape[2], manual=nm, rule=nr, kept=len(keep), sh=sh))
    a = fig.add_axes([0.70, 0.30, 0.29, 0.60])
    y = np.arange(len(rows))[::-1]
    for yy, r in zip(y, rows):
        a.barh(yy, r["cand"], color="0.88", edgecolor="0.6", height=0.62)
        a.barh(yy, r["manual"], color="#2ecc40", height=0.62)
        a.barh(yy, r["rule"], left=r["manual"], color=C.ONE_COL, height=0.62)
        a.text(r["cand"] + 4, yy, f"{r['kept']} of {r['cand']}", va="center", fontsize=10)
    # 10-25 (figure fix round 2): crossover wording ('Day 1: SBI-553'), every font >= 10 pt (the PNG is shown at 100 %)
    a.set_yticks(y, [f"{r['mouse']}\n{C.LABEL}: {r['arm']}" for r in rows], fontsize=10)
    for lab, r in zip(a.get_yticklabels(), rows):
        lab.set_color(ARMC[r["arm"]])
    a.set_xlabel("neurons (count)", fontsize=10.5); a.set_xlim(0, max(r["cand"] for r in rows) * 1.32)
    a.tick_params(axis="x", labelsize=10)
    a.set_title("candidates and kept neurons", loc="left", fontsize=11.5)
    a.legend(handles=[Patch(fc="0.88", ec="0.6", label="EXTRACT candidates"),
                      Patch(color="#2ecc40", label="kept, on both days"),
                      Patch(color=C.ONE_COL, label=f"kept, {C.ONE_LAB}")], fontsize=10, frameon=False, loc="lower right",
             bbox_to_anchor=(1.0, -0.48))
    clean(a)
    fig.savefig(out, dpi=160); plt.close(fig)
    return rows, ("Whole-day max dF/F projection (one detection per day, so the same neurons are followed in both sessions). Green = "
                  + f"the neuron exists on both days, {'magenta' if C.DAY == 1 else 'cyan'} = {C.ONE_LAB} (your confirmed tracking)"
                  + ". n = 2 mice per group. Max motion shift (px), "
                  "baseline / injection: " + ", ".join(f"{r['mouse']} {r['sh'][0]:.0f} / {r['sh'][1]:.0f}" for r in rows) + ".")


# -------------------------------------------------------------------------------------- NEURONS
def sessions_noise_z(B):
    A.Z_NORM = "noise"
    out = {}
    for m, _, _ in MICE:
        for ph in ("pre", "post"):
            out[(m, ph)] = A.load_session(mpath(B), m, ph, os.path.join(B, SCORE))
    return out


def fig_whole_heatmap(SS, out, m=C.EX):
    s0, s1 = SS[(m, "pre")], SS[(m, "post")]
    ids = s0["T"]["cell_ids"].astype(int)
    fig = plt.figure(figsize=(FW, FH))
    binned = {}
    for ph, s in (("pre", s0), ("post", s1)):
        T = s["T"]; t = T["t_s"]; Z = T["z"]
        nb = int(np.ceil(t.max()))
        k = np.clip(t.astype(int), 0, nb - 1)
        M = np.full((nb, Z.shape[1]), np.nan)
        df = pd.DataFrame(Z); df["b"] = k
        g = df.groupby("b").mean()
        M[g.index.values] = g.values
        M[: int(A.SKIP_S)] = np.nan
        binned[ph] = M
    blk = [(180, 360), (390, 570), (600, 780), (810, 990)]
    stim0 = s0["B"]["stim_order"]
    score = np.nanmean(np.vstack([np.nanmean(binned["pre"][b0:b1], 0) for b0, b1 in blk]), 0)
    order = np.argsort(-score)
    for r, (ph, s, lab) in enumerate((("pre", s0, "A  baseline session"), ("post", s1, "B  after the injection"))):
        y0 = 0.53 - r * 0.47
        ax = fig.add_axes([0.06, y0, 0.86, 0.33])
        M = binned[ph][:, order].T
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=-1, vmax=6, interpolation="nearest",
                       extent=(0, M.shape[1], len(ids), 0))
        ax.set_ylabel("neuron", fontsize=9); ax.set_yticks([])
        ax.tick_params(labelsize=8.5)
        if r == 1:
            ax.set_xlabel("time in session (s)")
        else:
            ax.set_xticklabels([])
        st = s["B"]["stim_order"]
        tk = fig.add_axes([0.06, y0 + 0.335, 0.86, 0.05], sharex=ax)
        tk.set_ylim(0, 1); tk.set_axis_off()
        for i, (b0, b1) in enumerate(blk):
            tk.text((b0 + b1) / 2, 1.0, st[i], ha="center", va="bottom", fontsize=8.5, color=SCOL[st[i]])
            for f in s["B"]["ev"][st[i]]:
                tk.plot([f / 30.0] * 2, [0.05, 0.85], color=SCOL[st[i]], lw=0.8)
        fig.text(0.06, y0 + 0.40, lab, fontsize=11, va="bottom")
    cax = fig.add_axes([0.935, 0.08, 0.012, 0.78])
    cb = fig.colorbar(im, cax=cax); cb.set_label("z (dF/F / noise SD)", fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.savefig(out, dpi=160); plt.close(fig)
    return (f"{m} ({ARM[m]}): all {len(ids)} neurons, 1-s bins, same neuron order in both sessions (sorted by baseline activity in "
            "the stimulus blocks). Ticks = deliveries, coloured by stimulus; the order of the blocks differs between the sessions. "
            "z = dF/F divided by the neuron's noise SD (one scale for both sessions).")


def event_responses(SS):
    """(mouse, phase, event) -> time x cells trial-mean z minus the [-2, 0] s mean; and the time axis"""
    out, n_tr, tt = {}, {}, None
    for (m, ph), s in SS.items():
        for ev in EVS:
            X, tt0, used = A.peri(s, s["B"]["ev"][ev])
            if X is None or len(used) < A.MIN_TRIALS:
                continue
            tt = tt0
            pre = (tt >= A.PRE_W[0]) & (tt < A.PRE_W[1])
            mu = np.nanmean(X, 0)
            out[(m, ph, ev)] = mu - np.nanmean(mu[pre], 0, keepdims=True)
            n_tr[(m, ph, ev)] = len(used)
    return out, n_tr, tt


def fig_event_heatmap(B, SS, out):
    RESP, NTR, tt = event_responses(SS)
    ET = pd.read_csv(C.A(B, "ca_day1", "ca_d1_cells_event_tests.csv")); ET["mouse"] = ET.mouse.astype(str)
    show = (tt >= -2) & (tt <= 5)
    resp_w = (tt >= 0) & (tt < 2)
    fig = plt.figure(figsize=(FW, FH))
    left, right, top, bot = 0.085, 0.915, 0.86, 0.11
    ncol = len(EVS) * 2
    gap = 0.008
    pw = (right - left - gap * len(EVS) - 3 * gap * (len(EVS) - 1)) / ncol
    ncell = {m: SS[(m, "pre")]["T"]["cell_ids"].size for m, _, _ in MICE}
    tot = sum(ncell.values())
    orders = {}
    for m, _, _ in MICE:
        sc = np.nanmean(np.vstack([np.nanmean(RESP[(m, "pre", e)][resp_w], 0) for e in STIM if (m, "pre", e) in RESP]), 0)
        orders[m] = np.argsort(-sc)
    x = left
    im = None
    for ei, ev in enumerate(EVS):
        for pi, ph in enumerate(("pre", "post")):
            ax = fig.add_axes([x, bot, pw, top - bot])
            rows, ups = [], []
            for m, _, _ in MICE:
                M = RESP.get((m, ph, ev))
                ids = SS[(m, ph)]["T"]["cell_ids"].astype(int)
                if M is None:
                    M = np.full((len(tt), len(ids)), np.nan)
                Mo = A.sm(M)[show][:, orders[m]]
                rows.append(Mo.T)
                upset = set(ET[(ET.mouse == m) & (ET.phase == ph) & (ET.event == ev) & (ET.cls_shift == "UP")
                               & (ET.q_shift <= A.Q_RESP)].cell.astype(int))
                ups.append(np.array([c in upset for c in ids[orders[m]]]))
            H = np.vstack(rows); U = np.concatenate(ups)
            im = ax.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest",
                           extent=(tt[show][0], tt[show][-1], tot, 0))
            ax.axvline(0, color="k", lw=0.6)
            yy = np.flatnonzero(U) + 0.5
            ax.plot(np.full(len(yy), tt[show][-1] - 0.25), yy, ">", color="k", ms=2.2, clip_on=False)
            c0 = 0
            for m, _, _ in MICE:
                c0 += ncell[m]
                ax.axhline(c0, color="k", lw=0.7)
            ax.set_yticks([]); ax.set_xticks([0, 4]); ax.tick_params(labelsize=7.5, length=2)
            ax.set_title("base" if ph == "pre" else "inj.", fontsize=8.5, pad=2)
            if ei == 0 and pi == 0:
                c0 = 0
                for m, arm, _ in MICE:
                    ax.text(tt[show][0] - 0.6, c0 + ncell[m] / 2, f"{m}\n{arm}\n{ncell[m]}", ha="right", va="center",
                            fontsize=8, color=ARMC[arm])
                    c0 += ncell[m]
            if pi == 0:
                fig.text(x + pw + gap / 2, top + 0.045, ev if ev != "movement onset" else "movement onset",
                         ha="center", fontsize=9.5, color=SCOL[ev], fontweight="bold")
            x += pw + (gap if pi == 0 else 3 * gap)
    fig.text((left + right) / 2, 0.025, "time from the event (s)", ha="center", fontsize=9.5)
    cax = fig.add_axes([0.945, bot, 0.01, top - bot])
    cb = fig.colorbar(im, cax=cax); cb.set_label("response (z, minus [-2, 0] s)", fontsize=8.5); cb.ax.tick_params(labelsize=7.5)
    fig.savefig(out, dpi=170); plt.close(fig)
    nup = ET[(ET.cls_shift == "UP") & (ET.q_shift <= A.Q_RESP) & ET.event.isin(EVS)]
    # the largest population-wide rises (mean over the neurons of one mouse, [0, 2] s)
    pm = sorted(((float(np.nanmean(M[resp_w])), m, ph, ev) for (m, ph, ev), M in RESP.items()), reverse=True)
    fig_event_heatmap.tops = pm[:3]
    return nup, ("Each row = one neuron (all 4 mice, n = 2 mice per group), same order in every panel (by the baseline response to "
                 "the 4 stimuli). Colour = trial-mean response; base = baseline, inj. = after injection. > = activity rose after the "
                 "event more than at shifted times (time-shift test, BH q <= 0.20, exploratory).")


def fig_resp_counts(B, out):
    R8 = pd.read_csv(C.A(B, "ca_responsive_change", "R8_responsive_calibrated.csv")); R8["mouse"] = R8.mouse.astype(str)
    R7 = pd.read_csv(C.A(B, "ca_responsive_change", "R7_population_response_test.csv")); R7["mouse"] = R7.mouse.astype(str)
    fig, ax = plt.subplots(1, 2, figsize=(FW, FH))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.2, wspace=0.18)
    off = {m: (i - 1.5) * 0.17 for i, (m, _, _) in enumerate(MICE)}
    for k, (D, col, ylab, ttl) in enumerate(((R8, "pct", "neurons responding (%)", "A  single neurons: strict test (max-T)"),
                                             (R7, "ratio", "population response / chance", "B  whole population (shift test)"))):
        a = ax[k]
        for j, ev in enumerate(EVS):
            for m, arm, _ in MICE:
                v = {}
                for ph in ("pre", "post"):
                    r = D[(D.mouse == m) & (D.phase == ph) & (D.event == ev)]
                    if not len(r):
                        continue
                    r = r.iloc[0]
                    v[ph] = 100 * r.maxT_up / r.n_cells if k == 0 else r.energy / r.energy_null_median
                    sig = (k == 1 and r.q_energy <= 0.05)
                    xx = j + off[m] + (-0.04 if ph == "pre" else 0.04)
                    a.plot(xx, v[ph], "o", ms=5.5, mfc="white" if ph == "pre" else ARMC[arm], mec=ARMC[arm], mew=1.3, zorder=3)
                    if sig:
                        a.text(xx, v[ph] + (0.12 if k == 1 else 1.0), "*", ha="center", fontsize=11, zorder=4)
                if len(v) == 2:
                    a.plot([j + off[m] - 0.04, j + off[m] + 0.04], [v["pre"], v["post"]], color=ARMC[arm], lw=1, zorder=2)
        if k == 1:
            a.axhline(1, color="0.4", ls="--", lw=1)
        a.set_xticks(range(len(EVS)), [e if e != "movement onset" else "movement\nonset" for e in EVS], fontsize=9.5)
        for lab, e in zip(a.get_xticklabels(), EVS):
            lab.set_color(SCOL[e])
        a.set_ylabel(ylab); a.set_title(ttl, loc="left"); a.set_ylim(bottom=0)
        clean(a)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="baseline"),
                        Line2D([], [], marker="o", ls="", mfc="k", mec="k", label="after injection"),
                        Patch(color=ARMC["SBI-553"], label=C.GROUP_LABEL["SBI-553"]), Patch(color=ARMC["vehicle"], label=C.GROUP_LABEL["vehicle"]),
                        Line2D([], [], color="0.4", ls="--", label="chance (B)"), Line2D([], [], color="k", ls="", marker="$*$", ms=8, label="* = that point vs chance (B, q <= 0.05)")],
               loc="upper center", ncol=6, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(out, dpi=160); plt.close(fig)
    up = R8[R8.event.isin(STIM)]
    mv = R8[R8.event == "movement onset"]
    return (100 * (up.maxT_up / up.n_cells).max(), 100 * (mv.maxT_up / mv.n_cells).max(),
            "One dot = one mouse and session; 4 dots per event (offset by mouse; n = 2 mice per group). A: neurons whose response "
            "beats the largest response of any neuron at shifted event times (Westfall-Young max-T, family-wise error 5 %; corrects "
            "for correlated neurons). B: RMS over neurons of the mean response vs the same events shifted in their block, BH over 56 tests.")


# ------------------------------------------------------------------------------------ BEHAVIOUR
BEH_ORDER = ["withdrawal", "flinch", "attending", "lick/bite", "guarding", "escape/rearing", "general movement", "grooming"]
BEH_CAM = {"withdrawal": "bottom", "flinch": "bottom", "attending": "side", "lick/bite": "side", "escape/rearing": "side",
           "general movement": "bottom", "grooming": "side"}


def example_frame(mp4, clip=1):
    """the middle 'on' frame (title drawn in red) of example clip `clip` of a behavior_example_clips.py video"""
    import cv2
    c = cv2.VideoCapture(mp4)
    red, frames, k, gap = [], [], 1, False
    while True:
        ok, fr = c.read()
        if not ok:
            break
        white = fr.mean() > 252                              # the white gap between clips
        if white:
            gap = True
            if k == clip and frames:
                break
            continue
        if gap:
            k, gap = k + 1, False
        if k != clip:
            continue
        frames.append(fr)
        t = fr[4:26, 0:420].astype(int)
        red.append(int(((t[:, :, 2] > 150) & (t[:, :, 1] < 80) & (t[:, :, 0] < 80)).sum()))
    if not frames:
        return None
    on = [i for i, v in enumerate(red) if v > 15]
    return frames[on[len(on) // 2] if on else len(frames) // 2]


def fig_behaviour_examples(B, out):
    ex = os.path.join(B, "analysis", "behavior_examples")          # the Day-1 example clips (same scoring on every day)
    tiles = []
    for b in BEH_ORDER:
        f = os.path.join(ex, f"behaviour_examples_{b.replace('/', '_').replace(' ', '_')}.mp4")
        if os.path.isfile(f):
            fr = example_frame(f, clip=2 if b == "flinch" else 1)     # flinch clip 1 = the same moment as withdrawal clip 1
            if fr is not None:
                im = fr[44:404, :540] if BEH_CAM[b] == "side" else fr[44:404, 540:]
                tiles.append((b, im[:, :, ::-1]))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from behavior_example_clips import DEF
    short = {"withdrawal": "paw pulled away right after contact (tap)", "flinch": "fast shake / flick of the paw (tap)",
             "attending": "looks at / sniffs the paw, no mouth contact (hold)", "lick/bite": "mouth contact with the paw (hold)",
             "guarding": "paw held up > 2 s (hold)", "escape/rearing": "stands up, tries to climb out (hold)",
             "general movement": "walks / turns (automatic, from video)", "grooming": "strokes face / licks flank (automatic)"}
    fig = plt.figure(figsize=(FW, FH))
    nc = 4
    w, h = 0.235, 0.36
    for i, (b, im) in enumerate(tiles):
        r, c = divmod(i, nc)
        ax = fig.add_axes([0.01 + c * (w + 0.013), 0.53 - r * 0.48, w, h])
        ax.imshow(im); ax.set_axis_off()
        kind = "REFLEXIVE" if b in ("withdrawal", "flinch") else "AUTOMATIC" if b in ("general movement", "grooming") else "AFFECTIVE"
        ax.set_title(f"{b}  ({kind.lower()})", fontsize=10, loc="left", fontweight="bold", pad=3)
        ax.text(0.0, -0.04, "\n".join(S1.wrap_lines(f"{short[b]}  [{BEH_CAM[b]} camera]", 8.5, w * FW - 0.1)),
                transform=ax.transAxes, fontsize=8.5, va="top")
    if len(tiles) < 8:
        r, c = divmod(len(tiles), nc)
        ax = fig.add_axes([0.01 + c * (w + 0.013), 0.53 - r * 0.48, w, h]); ax.set_axis_off()
        ax.text(0.02, 0.95, "How each session is scored", fontsize=10.5, fontweight="bold", va="top")
        ax.text(0.02, 0.78, "deliveries: key at contact (1-4 = this\nsession's stimulus order)\n"
                "reflexes: one tap per event\naffective: key held while it lasts\n"
                "movement, grooming: automatic from\nthe video, checked by eye\nscorer blind to the treatment",
                fontsize=9, va="top", linespacing=1.35)
    fig.savefig(out, dpi=160); plt.close(fig)
    return ("Single frames from the example clips (analysis\\behavior_examples, 3 clips per behaviour). Both cameras are triggered by "
            "the Inscopix frames (30 Hz), so camera frame n = imaging frame n. Guarding had no clear example on Day 1."
            + ("" if C.DAY == 1 else " Frames are from Day 1; Day 2 was scored the same way."))


def behaviour_counts(B):
    """per session x stimulus: deliveries and events in the response windows (every event counted)"""
    rows = []
    for p in sorted(glob.glob(os.path.join(B, SCORE, "ScoringAB_*.mat"))):
        d = BA.load(p)
        dF = d["dF"]
        held_on = {nm: BA.spans(v)[0] for nm, v in d["held"].items()}
        for i, (f, k) in enumerate(zip(dF, d["dT"])):
            if not 1 <= k <= 4:
                continue
            nxt = dF[i + 1] if i + 1 < len(dF) else d["n"]
            hi = min(f + int(2 * 30), nxt - 1)
            r = dict(mouse=d["mouse"], phase=d["phase"], stim=d["stim"][k - 1],
                     withdrawal=int(np.sum((d["W"] >= f) & (d["W"] <= hi))), flinch=int(np.sum((d["F"] >= f) & (d["F"] <= hi))))
            for nm, on in held_on.items():
                r[nm] = int(np.sum((on >= f) & (on <= f + 3 * 30)))
            rows.append(r)
    D = pd.DataFrame(rows)
    G = D.groupby(["mouse", "phase", "stim"]).agg(n=("withdrawal", "size"), **{b: (b, "sum") for b in BEH_ORDER}).reset_index()
    return D, G


def rate_test(k1, n1, k2, n2):
    """exact Poisson rate comparison (conditional binomial): ratio (k2/n2)/(k1/n1), 95% CI, p"""
    if k1 + k2 == 0:
        return np.nan, np.nan, np.nan, np.nan
    p = binomtest(int(k2), int(k1 + k2), n2 / (n1 + n2)).pvalue
    lo = beta.ppf(0.025, k2, k1 + 1) if k2 > 0 else 0.0
    hi = beta.ppf(0.975, k2 + 1, k1) if k1 > 0 else np.inf
    f = n1 / n2
    rat = (k2 / n2) / (k1 / n1) if k1 > 0 else np.inf
    return rat, lo / (1 - lo) * f if lo < 1 else np.inf, hi / (1 - hi) * f if hi < 1 else np.inf, p


def fig_behaviour_forest(B, out):
    """events per delivery, baseline -> injection, one line per mouse; the behaviours with events in most
    sessions get a panel, the rare ones (lick/bite, guarding, grooming) are pooled in the last one"""
    D, G = behaviour_counts(B)
    T = G.groupby(["mouse", "phase"]).sum(numeric_only=True).reset_index()
    panels = [("withdrawal", "withdrawal"), ("flinch", "flinch"), ("attending", "attending"), ("lick/bite", "lick / bite"),
              ("escape/rearing", "escape / rearing"), ("general movement", "movement"), ("grooming", "grooming")]
    fig, ax = plt.subplots(1, len(panels), figsize=(FW, FH))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.8, bottom=0.17, wspace=0.62)
    res = []
    for bi, (b, ttl) in enumerate(panels):
        a = ax[bi]
        top, labs = 0, []
        for m, arm, _ in MICE:
            r0 = T[(T.mouse == m) & (T.phase == "baseline")]; r1 = T[(T.mouse == m) & (T.phase == "injection")]
            k1, n1, k2, n2 = int(r0[b].iloc[0]), int(r0.n.iloc[0]), int(r1[b].iloc[0]), int(r1.n.iloc[0])
            rat, lo, hi, p = rate_test(k1, n1, k2, n2)
            res.append(dict(behaviour=b, mouse=m, arm=arm, base_events=k1, base_deliveries=n1, inj_events=k2, inj_deliveries=n2,
                            base_per_delivery=k1 / n1, inj_per_delivery=k2 / n2, ratio=rat, ci_lo=lo, ci_hi=hi, p=p))
            v0, v1 = k1 / n1, k2 / n2
            top = max(top, v0, v1)
            a.plot([0, 1], [v0, v1], color=ARMC[arm], lw=1.2)
            a.plot(0, v0, "o", ms=5.5, mfc="white", mec=ARMC[arm], mew=1.3)
            a.plot(1, v1, "o", ms=5.5, color=ARMC[arm])
            st = PS.star(p) if np.isfinite(p) and p < 0.05 else ""   # that mouse: baseline vs after the injection
            labs.append((v1, f"{m} {st}", ARMC[arm]))
        last = -1e9
        for v1, t, col in sorted(labs):                       # pushed apart so they do not overlap
            yv = max(v1, last + 0.075 * max(top, 1e-9))
            a.text(1.08, yv, t, va="center", fontsize=7.5, color=col)
            last = yv
        a.set_xticks([0, 1], ["base", "inj."], fontsize=9); a.set_xlim(-0.25, 1.75)
        a.set_ylim(0, top * 1.15 if top > 0 else 1)
        a.set_title(ttl, fontsize=10, loc="left")
        if bi == 0:
            a.set_ylabel("events per delivery")
        clean(a)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="baseline"),
                        Line2D([], [], marker="o", ls="", mfc="k", mec="k", label="after injection"),
                        Patch(color=ARMC["SBI-553"], label=C.GROUP_LABEL["SBI-553"]), Patch(color=ARMC["vehicle"], label=C.GROUP_LABEL["vehicle"]),
                        Line2D([], [], color="k", ls="", marker="$*$", ms=8, label="* next to a mouse = its baseline vs after injection")],
               loc="upper center", ncol=5, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(out, dpi=160); plt.close(fig)
    Rz = pd.DataFrame(res)
    ng = int(G["guarding"].sum()) if "guarding" in G else 0
    return Rz, G, ("Events per delivery = all events in the response window / deliveries, 4 stimuli pooled (reflexes: delivery to 2 s or "
                   "the next delivery; held and automatic behaviours: bout starts within 3 s). n = 2 mice per group. Exact Poisson rate "
                   "test per mouse, uncorrected (* p<0.05, ** <0.01, *** <0.001). "
                   + (f"Guarding: no events on {C.LABEL}." if ng == 0 else f"Guarding: {ng} events, pooled in no panel."))


EVR = ["Light touch", "Mild touch", "Pin prick", "Heat", "withdrawal", "flinch", "escape/rearing onset", "movement onset"]
SHR = {"Light touch": "light touch", "Mild touch": "mild touch", "Pin prick": "pin prick", "Heat": "heat", "withdrawal": "withdrawal",
       "flinch": "flinch", "escape/rearing onset": "escape", "movement onset": "movement"}


def fig_responsive_change(B, out):
    """activated / inhibited neurons (selected on half of the baseline trials), their response on the other half and
    after the injection; per event, one pair of bars per mouse (ca_cell_event_matrix.py)"""
    Su = pd.read_csv(C.A(B, "ca_cell_event_matrix", "responsive_change_summary.csv")); Su["mouse"] = Su.mouse.astype(str)
    fig, ax = plt.subplots(2, len(EVR), figsize=(FW, FH))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.85, bottom=0.12, wspace=0.42, hspace=0.42)
    nsig = {"activated": [0, 0], "inhibited": [0, 0]}                    # [n tests, n p < 0.05]
    for r_i, k in enumerate(("activated", "inhibited")):
        for e_i, e in enumerate(EVR):
            a = ax[r_i, e_i]
            lo, hi = 0.0, 0.0
            mk_ = []
            for i, (m, arm, _) in enumerate(MICE):
                g = Su[(Su.mouse == m) & (Su.event == e) & (Su.group == k)]
                x = i * 1.0
                if not len(g):
                    a.text(x, 0, "-", ha="center", va="center", fontsize=7, color="0.6")
                    continue
                g = g.iloc[0]
                se0 = g.base_sem if np.isfinite(g.base_sem) else 0; se1 = g.inj_sem if np.isfinite(g.inj_sem) else 0
                a.bar(x - 0.21, g.base_mean, yerr=se0, color="white", edgecolor=ARMC[arm], width=0.38, lw=1, capsize=1.2,
                      error_kw=dict(lw=0.6))
                a.bar(x + 0.21, g.inj_mean, yerr=se1, color=ARMC[arm], width=0.38, capsize=1.2, error_kw=dict(lw=0.6))
                lo = min(lo, g.base_mean - se0, g.inj_mean - se1); hi = max(hi, g.base_mean + se0, g.inj_mean + se1)
                if np.isfinite(g.p_paired):
                    nsig[k][0] += 1
                    if g.p_paired < 0.05:
                        nsig[k][1] += 1
                        mk_.append((x, max(g.base_mean + se0, g.inj_mean + se1), g.p_paired))
            span = max(hi - lo, 0.2)
            a.set_ylim(lo - 0.25 * span, hi + 0.3 * span)
            for x_, y_, p_ in mk_:                  # bracket = that mouse's baseline bar vs its injection bar (paired Wilcoxon)
                PS.bracket(a, x_ - 0.21, x_ + 0.21, y_ + 0.05 * span, p_, "k", fs=9, show_ns=False, inset=0)
            for i, (m, arm, _) in enumerate(MICE):
                g = Su[(Su.mouse == m) & (Su.event == e) & (Su.group == k)]
                a.text(i, lo - 0.2 * span, f"{int(g.n_cells.iloc[0])}" if len(g) else "0", ha="center", va="bottom", fontsize=6,
                       color="0.35")
            a.axhline(0, color="0.4", lw=0.6)
            a.set_xticks(range(4), [m for m, _, _ in MICE], fontsize=6, rotation=90)
            for lab, (m, arm, _) in zip(a.get_xticklabels(), MICE):
                lab.set_color(ARMC[arm])
            a.tick_params(axis="y", labelsize=7)
            if r_i == 0:
                a.set_title(SHR[e], fontsize=9.5)
            if e_i == 0:
                a.set_ylabel(f"{k} neurons\nresponse (z)", fontsize=8.5)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    fig.legend(handles=[Patch(fc="white", ec="k", label="baseline (other half of the trials)"), Patch(color="k", label="after injection"),
                        Patch(color=ARMC["SBI-553"], label=C.GROUP_LABEL["SBI-553"]), Patch(color=ARMC["vehicle"], label=C.GROUP_LABEL["vehicle"])],
               loc="upper center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(out, dpi=160); plt.close(fig)
    # how often the held-out baseline response has the sign the neuron was selected for (a replication check)
    Sp = pd.read_csv(C.A(B, "ca_cell_event_matrix", "responsive_change_split_half.csv"))
    Sp = Sp[Sp.event.isin(EVR)]
    rep = {k: float(np.mean((g.base_heldout > 0) if k == "activated" else (g.base_heldout < 0))) for k, g in Sp.groupby("group")}
    return nsig, rep, ("Neurons selected on one half of the baseline trials (Wilcoxon signed-rank of the trial responses, p < 0.05; "
                       "activated = rose, inhibited = fell), measured on the other half (open) and on all injection trials (filled). "
                       "Mean +- SEM over neurons; numbers = neurons; bracket with * = that mouse's baseline vs injection bars (paired Wilcoxon p < 0.05). Response = z [0, 2] s minus z [-2, 0] s.")


def fig_behaviour_stim(B, G, out):
    fig, ax = plt.subplots(1, 4, figsize=(FW, FH), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.62]))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.2, wspace=0.32)
    G = G.copy()
    G["affective"] = G["attending"] + G["lick/bite"] + G["guarding"]
    off = {m: (i - 1.5) * 0.15 for i, (m, _, _) in enumerate(MICE)}
    for k, (b, ttl) in enumerate((("withdrawal", "A  withdrawal"), ("flinch", "B  flinch"),
                                   ("affective", "C  affective (attending + lick/bite + guarding)"))):
        a = ax[k]
        for j, s in enumerate(STIM):
            for m, arm, _ in MICE:
                v = []
                for ph in ("baseline", "injection"):
                    r = G[(G.mouse == m) & (G.phase == ph) & (G.stim == s)]
                    v.append(float(r[b].iloc[0] / r.n.iloc[0]) if len(r) and r.n.iloc[0] else np.nan)
                xx = j + off[m]
                a.plot([xx - 0.035, xx + 0.035], v, color=ARMC[arm], lw=1)
                a.plot(xx - 0.035, v[0], "o", ms=5, mfc="white", mec=ARMC[arm], mew=1.2)
                a.plot(xx + 0.035, v[1], "o", ms=5, mfc=ARMC[arm], mec=ARMC[arm])
        a.set_xticks(range(4), [s.replace(" ", "\n") for s in STIM], fontsize=9.5)
        a.set_ylabel("events per delivery" if k == 0 else ""); a.set_title(ttl, loc="left", fontsize=10.5)
        a.set_ylim(bottom=0)
        clean(a)
    # spontaneous movement in the first 3 min (no stimulus)
    H = pd.read_csv(C.A(B, "behavior_day1", "behav_d1_held.csv"))
    H["mouse"] = H.mouse.astype(str)
    a = ax[3]
    mv = {}
    for m, arm, _ in MICE:
        v = [float(H[(H.mouse == m) & (H.phase == ph) & (H.block == "baseline") & (H.behaviour == "general movement")].pct_time.iloc[0])
             for ph in ("baseline", "injection")]
        mv[m] = v
        a.plot([0, 1], v, "-o", color=ARMC[arm], mfc="white", lw=1.2)
        a.plot(1, v[1], "o", color=ARMC[arm])
    # labels at the injection end, pushed apart so they do not overlap
    lab = sorted(((mv[m][1], m) for m, _, _ in MICE))
    ymax = max(max(v) for v in mv.values())
    last = -1e9
    for v, m in lab:
        yv = max(v, last + 0.07 * ymax)
        a.text(1.07, yv, m, va="center", fontsize=8, color=ARMC[ARM[m]], clip_on=False)
        last = yv
    a.set_xticks([0, 1], ["baseline", "injection"]); a.set_xlim(-0.3, 1.45)
    a.set_ylabel("time moving (%)"); a.set_title("D  first 3 min, no stimulus", loc="left", fontsize=10.5)
    a.set_ylim(bottom=0)
    clean(a)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="baseline"),
                        Line2D([], [], marker="o", ls="", mfc="k", mec="k", label="after injection"),
                        Patch(color=ARMC["SBI-553"], label=C.GROUP_LABEL["SBI-553"]), Patch(color=ARMC["vehicle"], label=C.GROUP_LABEL["vehicle"])],
               loc="upper center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(out, dpi=160); plt.close(fig)
    nmin, nmax = int(G.n.min()), int(G.n.max())
    return mv, nmin, nmax, (f"Per mouse and stimulus (n = 2 mice per group; {nmin}-{nmax} deliveries per stimulus and session, so counts are always "
                        "divided by deliveries). Line joins baseline (open) and injection (filled) of one mouse. D: % of frames with general "
                        "movement before any stimulus (automatic, from the video).")


# ------------------------------------------------------------------------------------------ deck
HOWC = (0x1A, 0x6E, 0x8E)                     # the "how to read" line under the title


def add_slide(prs, section, title, fig_png, note, bottom, preview_png, howto=""):
    """One slide: section label, title, a 'how to read' line (what the figure shows), the figure, the
    graph note (tests, n, definitions) and one or two sentences of what we did and found.
    Layout (13.33 x 7.5 in): section 0.06, title 0.28-0.88, how-to 0.9-1.35, figure 1.4-5.55, note
    5.6-6.15, sentences 6.22-7.3. Every text is wrapped by measured width; too long stops the build."""
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from PIL import Image
    tpt = next((p for p in (26, 24, 22) if S1.text_width_in(title, p, bold=True) <= 12.3), None)
    if tpt is None:
        raise SystemExit(f"title too long for one line: {title!r}")
    hl = S1.wrap_lines(howto, 12.5, 12.3) if howto else []
    nl = S1.wrap_lines(note, 10.5, 12.3) if note else []
    bl = S1.wrap_lines(bottom, 14.5, 12.3)
    if len(hl) > 2 or len(nl) > 3 or len(bl) > 3:
        raise SystemExit(f"text too long (how-to {len(hl)}, note {len(nl)}, sentences {len(bl)} lines): {title!r}")
    s = prs.slides.add_slide(prs.slide_layouts[6])

    def box(x, y, w, h, text, pt, bold=False, rgb=(0x33, 0x33, 0x33)):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.02)
        p = tf.paragraphs[0]; p.text = text; p.font.size = Pt(pt); p.font.bold = bold; p.font.name = "Arial"
        p.font.color.rgb = RGBColor(*rgb)
    box(X0, 0.06, 6, 0.3, section, 11, True, (0x8A, 0x8F, 0x98))
    if C.PROV_MARK:
        box(8.3, 0.04, 4.6, 0.3, C.PROV_MARK, 10.5, True, (0xC0, 0x39, 0x2B))
    box(X0, 0.28, 12.4, 0.62, title, tpt, True, (0x1F, 0x2D, 0x3D))
    if howto:
        box(X0, 0.9, 12.4, 0.48, howto, 12.5, False, HOWC)
    im = Image.open(fig_png)
    boxw, boxh, top = 12.4, 4.15, 1.4
    w = boxw; h = w * im.size[1] / im.size[0]
    if h > boxh:
        h = boxh; w = h * im.size[0] / im.size[1]
    s.shapes.add_picture(fig_png, Inches(X0 + (boxw - w) / 2), Inches(top), Inches(w), Inches(h))
    if note:
        box(X0, 5.6, 12.4, 0.58, note, 10.5, False, (0x55, 0x55, 0x55))
    box(X0, 6.22, 12.4, 1.05, bottom, 14.5)
    fig = plt.figure(figsize=(S1.SW, S1.SH), dpi=100)
    fx = (X0 + 0.05) / S1.SW
    fig.text(fx, 1 - 0.2 / S1.SH, section, fontsize=11, fontweight="bold", color="#8a8f98", va="center")
    if C.PROV_MARK:
        fig.text((8.35) / S1.SW, 1 - 0.19 / S1.SH, C.PROV_MARK, fontsize=10.5, fontweight="bold", color="#c0392b", va="center")
    fig.text(fx, 1 - 0.58 / S1.SH, title, fontsize=tpt, fontweight="bold", color="#1f2d3d", va="center")
    if hl:
        fig.text(fx, 1 - 0.93 / S1.SH, "\n".join(hl), fontsize=12.5, color="#1a6e8e", va="top", linespacing=1.12)
    ax = fig.add_axes([(X0 + (boxw - w) / 2) / S1.SW, 1 - (top + h) / S1.SH, w / S1.SW, h / S1.SH])
    ax.imshow(im); ax.set_axis_off()
    if nl:
        fig.text(fx, 1 - 5.63 / S1.SH, "\n".join(nl), fontsize=10.5, color="#555555", va="top", linespacing=1.12)
    fig.text(fx, 1 - 6.25 / S1.SH, "\n".join(bl), fontsize=14.5, color="#333333", va="top", linespacing=1.12)
    fig.savefig(preview_png, dpi=100); plt.close(fig)


def summary_slide(prs, cards, preview_png, title, bottom, howto=None):
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    s = prs.slides.add_slide(prs.slide_layouts[6])

    def box(x, y, w, h, text, pt, bold=False, rgb=(0x33, 0x33, 0x33)):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.02)
        p = tf.paragraphs[0]; p.text = text; p.font.size = Pt(pt); p.font.bold = bold; p.font.name = "Arial"
        p.font.color.rgb = RGBColor(*rgb)
    box(X0, 0.06, 6, 0.3, "SUMMARY", 11, True, (0x8A, 0x8F, 0x98))
    if C.PROV_MARK:
        box(8.3, 0.04, 4.6, 0.3, C.PROV_MARK, 10.5, True, (0xC0, 0x39, 0x2B))
    box(X0, 0.3, 12.4, 0.65, title, 26, True, (0x1F, 0x2D, 0x3D))
    howto = howto or (f"How to read: four results of {C.LABEL}, one per card; the big number is the key value, the text says what it is "
                      "and where it comes from (the slides that follow show each one).")
    box(X0, 0.9, 12.4, 0.48, howto, 12.5, False, HOWC)
    fig = plt.figure(figsize=(S1.SW, S1.SH), dpi=100)
    fig.text((X0 + 0.05) / S1.SW, 1 - 0.2 / S1.SH, "SUMMARY", fontsize=11, fontweight="bold", color="#8a8f98", va="center")
    if C.PROV_MARK:
        fig.text(8.35 / S1.SW, 1 - 0.19 / S1.SH, C.PROV_MARK, fontsize=10.5, fontweight="bold", color="#c0392b", va="center")
    fig.text((X0 + 0.05) / S1.SW, 1 - 0.62 / S1.SH, title, fontsize=26, fontweight="bold", color="#1f2d3d", va="center")
    fig.text((X0 + 0.05) / S1.SW, 1 - 0.93 / S1.SH, "\n".join(S1.wrap_lines(howto, 12.5, 12.3)), fontsize=12.5, color="#1a6e8e",
             va="top", linespacing=1.12)
    cw, ch, gx, gy = 6.05, 2.1, 0.3, 0.25
    for i, (big, head, text) in enumerate(cards):
        r, c = divmod(i, 2)
        x, y = X0 + c * (cw + gx), 1.45 + r * (ch + gy)
        sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(cw), Inches(ch))
        sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor(0xF3, 0xF5, 0xF8); sh.line.fill.background()
        sh.adjustments[0] = 0.06
        bw = 1.75                                                   # column of the big number
        if S1.text_width_in(big, 30, True) > bw - 0.1:
            raise SystemExit(f"summary number too wide: {big!r}")
        hl = S1.wrap_lines(head, 14, cw - bw - 0.45)
        tl = S1.wrap_lines(text, 12, cw - bw - 0.45)
        hy = y + 0.18 + 0.24 * len(hl) + 0.12
        if hy + 0.21 * len(tl) > y + ch - 0.05:
            raise SystemExit(f"summary card text too long: {head!r}")
        box(x + 0.15, y + 0.15, bw, 0.7, big, 30, True, (0xC0, 0x39, 0x2B) if i == 0 else (0x1F, 0x2D, 0x3D))
        box(x + bw + 0.2, y + 0.15, cw - bw - 0.35, 0.3 + 0.24 * len(hl), head, 14, True, (0x1F, 0x2D, 0x3D))
        box(x + bw + 0.2, hy, cw - bw - 0.35, y + ch - hy - 0.05, text, 12, False, (0x44, 0x44, 0x44))
        fig.patches.append(plt.Rectangle((x / S1.SW, 1 - (y + ch) / S1.SH), cw / S1.SW, ch / S1.SH, transform=fig.transFigure,
                                         fc="#f3f5f8", ec="none"))
        fig.text((x + 0.2) / S1.SW, 1 - (y + 0.18) / S1.SH, big, fontsize=30, fontweight="bold", va="top",
                 color="#c0392b" if i == 0 else "#1f2d3d")
        fig.text((x + bw + 0.25) / S1.SW, 1 - (y + 0.18) / S1.SH, "\n".join(hl), fontsize=14, fontweight="bold", va="top",
                 color="#1f2d3d", linespacing=1.15)
        fig.text((x + bw + 0.25) / S1.SW, 1 - (hy + 0.03) / S1.SH, "\n".join(tl), fontsize=12, va="top", color="#444444",
                 linespacing=1.15)
    box(X0, 6.3, 12.4, 1.0, bottom, 15)
    fig.text((X0 + 0.05) / S1.SW, 1 - 6.33 / S1.SH, "\n".join(S1.wrap_lines(bottom, 15, 12.3)), fontsize=15, color="#333333",
             va="top", linespacing=1.15)
    fig.savefig(preview_png, dpi=100); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-frame", default="auto")
    a = ap.parse_args(argv)
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(S1.SW), Inches(S1.SH)
    od = os.path.splitext(a.out)[0] + "_figures"
    k = build_main(a.base, prs, od, a.video_frame)
    prs.save(a.out)
    print("->", a.out, "|", k, "slides | figures + previews in", od)


def build_main(B, prs, od, video_frame="auto", pvf=None, limits=True):
    """add the main slides to prs; pvf() -> the preview path of the next slide (default: od/preview_slideNN.png)"""
    os.makedirs(od, exist_ok=True)
    f = lambda nm: os.path.join(od, nm)
    W = S1.whole(B)
    nk = S1.kept_counts(B)
    ncell = {m: int(W[(W.mouse == m) & (W.phase == "pre")].cell.nunique()) for m, _, _ in MICE}
    bad = {m: (ncell[m], nk[m][0]) for m, _, _ in MICE if ncell[m] != nk[m][0]}
    if bad:
        raise SystemExit(f"analysis cells differ from the confirmed selection (analysis, selection): {bad}")
    # figures
    SS = sessions_noise_z(B)
    n_design = fig_design2(SS, nk, f("s02_design.png"))
    n_pipe = fig_pipeline(B, f(f"s03_pipeline_{C.EX}.png"))
    rows, n_neur = fig_neurons(B, f("s04_neurons_per_mouse.png"))
    n_heat = fig_whole_heatmap(SS, f(f"s05_whole_session_heatmap_{C.EX}.png"))
    nup, n_evh = fig_event_heatmap(B, SS, f("s07_event_heatmap.png"))
    up_stim_max, up_mv_max, n_cnt = fig_resp_counts(B, f("s08_responsive_counts.png"))
    n_act = S1.fig_activity(W, f("s09_activity_per_mouse.png"))
    n_arm = S1.fig_arm(B, f("s10_sbi_vs_vehicle.png"))
    n_bex = fig_behaviour_examples(B, f("s11_behaviour_examples.png"))
    Rz, G, n_bf = fig_behaviour_forest(B, f("s12_behaviour_per_delivery.png"))
    Rz.round(4).to_csv(f("s12_behaviour_per_delivery_tests.csv"), index=False)
    G.to_csv(f("s13_behaviour_counts_per_stimulus.csv"), index=False)
    mvp, nmin, nmax, n_bs = fig_behaviour_stim(B, G, f("s13_behaviour_per_stimulus.png"))
    n_cav = S1.fig_caveats(B, f("s14_caveats.png"))
    nsig, rep, n_rc = fig_responsive_change(B, f("s08b_responsive_change.png"))
    # numbers for the sentences
    rb = C.A(B, "ca_robustness")
    R1 = pd.read_csv(os.path.join(rb, "R1_robustness_per_mouse.csv")); R1["mouse"] = R1.mouse.astype(str)
    RL = pd.read_csv(os.path.join(rb, "R2_leave_one_mouse_out.csv"))
    RA = pd.read_csv(os.path.join(rb, "R1_robustness_arm.csv"))
    V = R1[(R1.window == "stimulus blocks") & (R1.variant == "main: OASIS transients, still frames")].set_index("mouse")
    ch = {m: 100 * (2 ** V.at[m, "log2_ratio_median"] - 1) for m, _, _ in MICE if m in V.index}
    V0 = R1[(R1.window == "no-stimulus start") & (R1.variant == "main: OASIS transients, still frames")].set_index("mouse")
    ch0 = {m: 100 * (2 ** V0.at[m, "log2_ratio_median"] - 1) for m, _, _ in MICE if m in V0.index}
    big = min(ch, key=ch.get)                                    # the mouse with the largest drop (Day 1: 5941)
    l5941 = RL[(RL.window == "stimulus blocks") & (RL.left_out == big)].sbi_minus_vehicle.iloc[0]
    all4 = RL[(RL.window == "stimulus blocks") & (RL.left_out == "none")].sbi_minus_vehicle.iloc[0]
    q = RA[RA.window == "stimulus blocks"]
    R7 = pd.read_csv(C.A(B, "ca_responsive_change", "R7_population_response_test.csv"))
    st = R7[~R7.event.isin(["movement onset", "flinch"])]; mv = R7[R7.event == "movement onset"]
    R4 = pd.read_csv(C.A(B, "ca_response_shape", "R4_down_speed_cells.csv"))
    R5 = pd.read_csv(C.A(B, "ca_responsive_change", "R5_split_half_reliability.csv"))
    tr = R5[R5.event.isin(STIM)].n_trials
    wd = Rz[Rz.behaviour == "withdrawal"].set_index("mouse")
    esc = Rz[Rz.behaviour == "escape/rearing"].set_index("mouse")
    ntot = sum(nk[m][0] for m, _, _ in MICE)
    Rc = pd.read_csv(C.A(B, "ca_cell_event_matrix", "responsive_change_summary.csv"))
    Rc = Rc[Rc.event.isin(EVR) & (Rc.p_paired < 0.05)]
    rcd = {g: (int((x.inj_mean.abs() < x.base_mean.abs()).sum()), int((x.inj_mean.abs() >= x.base_mean.abs()).sum()))
           for g, x in Rc.groupby("group")}
    for g in ("activated", "inhibited"):
        rcd.setdefault(g, (0, 0))
    Rr = pd.read_csv(FD_REVIEW(B, C.EX))
    n_kept_5941, n_cand_5941 = int((Rr.keep == 1).sum()), len(Rr)
    Sh5941 = pd.read_csv(mpath(B, f"{C.EX}{C.T}", "1_movie", f"{C.EX}{C.T}_pre_mc_shifts.csv"))
    sh_med_5941 = float(np.median(np.hypot(Sh5941.x_shift_px, Sh5941.y_shift_px)))
    sig_dir = {}
    for _, r in Rz[Rz.p < 0.05].iterrows():
        sig_dir.setdefault((r.behaviour, np.sign(r.inj_per_delivery - r.base_per_delivery)), []).append(str(r.mouse))
    gs = G.assign(r=G.withdrawal / G.n).pivot_table(index=["mouse", "phase"], columns="stim", values="r")
    n_grad = int((gs[["Pin prick", "Heat"]].max(1) > gs["Light touch"]).sum()); n_gs = len(gs)
    pv = (lambda k: pvf()) if pvf else (lambda k: f(f"preview_slide{k:02d}.png"))
    k = 1
    sbi = C.GROUP["SBI-553"]
    cards = [(f"{ch[big]:+.0f}%", (f"Activity in the stimulus blocks fell most in {big} ({ARM[big]})" if ch[big] < 0 else
                                   f"Activity in the stimulus blocks: smallest change in {big} ({ARM[big]})"),
              "Median change per neuron: " + ", ".join(f"{m} {ch[m]:+.0f}%" for m in ch) + ". SBI-553 minus vehicle "
              f"{all4:+.2f} log2; without {big}: {l5941:+.2f} log2."),
             (f"{ntot}", "neurons followed in both sessions",
              ", ".join(f"{m} {nk[m][0]}" for m, _, _ in MICE) + " (whole-day detection, every candidate checked on the max projection)."),
             (f"{int((mv.q_energy <= 0.05).sum())}/{len(mv)}", "sessions with a clear population response to movement onset",
              f"Stimuli and withdrawal: {int((st.q_energy <= 0.05).sum())}/{len(st)}. Strictly tested single neurons: at most "
              f"{up_stim_max:.0f}% per stimulus, up to {up_mv_max:.0f}% for movement onset."),
             (f"{mvp[sbi[0]][1]:.0f}, {mvp[sbi[1]][1]:.0f}%", f"time moving before any stimulus after SBI-553 ({', '.join(sbi)})",
              "Baseline -> injection: " + ", ".join(f"{m} {mvp[m][0]:.0f}->{mvp[m][1]:.0f}%" for m, _, _ in MICE)
              + ". Withdrawal per delivery: " + ", ".join(
                  f"{m} {wd.at[m, 'base_per_delivery']:.2f}->{wd.at[m, 'inj_per_delivery']:.2f}" for m, _, _ in MICE if m in wd.index) + ".")]
    summary_slide(prs, cards, pv(k), f"{C.LABEL} (SBI-553 vs vehicle, 2 + 2 mice): what we found so far",
                  "Mini1p Ca imaging and pain behaviour in 4 mice, each imaged at baseline and after SBI-553 or vehicle on the same "
                  "day. " + C.NEXT); k += 1
    PH = {"pre": "baseline", "post": "injection"}
    add_slide(prs, "IMAGING", "Design: one field of view, two sessions, four stimuli", f("s02_design.png"), n_design,
              "Each mouse was imaged in a baseline session and again after SBI-553 or vehicle on the same day, with the same "
              "neurons and the same four stimuli; the scorer and the cell selection were blind to the treatment.", pv(k),
              howto="How to read: A = the time line of one session (grey = no stimulus, orange = stimulus blocks); B = the four "
                    "mice, their group, how many neurons we followed and the order in which the four stimuli were given."); k += 1
    add_slide(prs, "IMAGING", f"From the raw movie to neurons ({'best case' if C.DAY == 1 else 'example'}, {C.EX})",
              f(f"s03_pipeline_{C.EX}.png"), n_pipe,
              "We corrected motion, detected candidate neurons with EXTRACT and kept only those that look like a soma on the max "
              f"projection. Result: {n_kept_5941} of {n_cand_5941} candidates kept; median motion shift {sh_med_5941:.1f} px.", pv(k),
              howto="How to read: panels 1-4 go from the movie to the list of neurons (arrows); 5 = how far the brain moved during "
                    "the session (small = good); 6 = the result, one activity trace per neuron."); k += 1
    add_slide(prs, "IMAGING", "Neurons per mouse", f("s04_neurons_per_mouse.png"), n_neur,
              f"{ntot} neurons in 4 mice (" + ", ".join(f"{r['mouse']} {r['kept']} of {r['cand']}" for r in rows) + "). "
              + f"{min(rows, key=lambda r: r['kept'])['mouse']} has the fewest neurons; "
              + f"{max(rows, key=lambda r: max(r['sh']))['mouse']} moved most (max shift "
              + f"{max(max(r['sh']) for r in rows):.0f} px).", pv(k),
              howto="How to read: each image = one mouse's field of view, every outline = one neuron we analyse (green = on both days, "
                    + ("magenta" if C.DAY == 1 else "cyan") + f" = {C.ONE_LAB}). Right: candidates proposed by EXTRACT and how many we kept."); k += 1
    add_slide(prs, "NEURONS", f"Every neuron over a whole session ({C.EX}, {ARM[C.EX]})", f(f"s05_whole_session_heatmap_{C.EX}.png"), n_heat,
              f"We binned every neuron's activity in 1-s steps over both sessions. Result: activity in the stimulus blocks "
              f"{'decreased' if ch[C.EX] < 0 else 'increased'} after the injection (median change per neuron {ch[C.EX]:+.0f}%).", pv(k),
              howto="How to read: each row = one neuron, time runs left to right over the whole session, bright = active. Coloured "
                    "ticks on top = stimulus deliveries. A and B show the same neurons in the same order."); k += 1
    vf = video_frame
    ET = pd.read_csv(C.A(B, "ca_day1", "ca_d1_cells_event_tests.csv")); ET = ET[ET.event == "Heat"]
    nup_all = ET.assign(up=ET.cls_shift == "UP").groupby([ET["mouse"].astype(str), "phase"]).up.sum()
    if vf == "auto":
        bm, bp = nup_all.idxmax()
        c = sorted(glob.glob(C.A(B, "ca_videos", f"{bm}{C.T}_{bp}_behaviour_ca_Heat_*s_frame.png")), key=os.path.getmtime)
        vf = c[-1] if c else ""
    if vf and os.path.isfile(vf):
        rid, target, nh, nall, note2 = S1.fig_session(B, vf, f("s06_example_session.png"))
        oth = nup_all.drop(nup_all.idxmax())
        add_slide(prs, "NEURONS", f"Example session: Ca activity in the {target.lower()} block", f("s06_example_session.png"), note2,
                  f"{nh} of {nall} neurons responded time-locked to {target.lower()} here, the most of the 8 sessions (others "
                  f"{int(oth.min())}-{int(oth.max())}). Videos for every stimulus and for movement: analysis\\{C.an('ca_videos')}.", pv(k),
                  howto="How to read: A = one frame of the Ca movie (bright = active; outlines = neurons); B = 8 of those neurons "
                        "over 40 s with the deliveries (vertical lines) and, below, what the mouse did at the same moment."); k += 1
    tops = fig_event_heatmap.tops
    add_slide(prs, "NEURONS", "Each neuron's response to each event", f("s07_event_heatmap.png"), n_evh,
              "We averaged each neuron's activity around every event. Result: largest mean response over all neurons of a mouse: "
              + ", ".join(f"{m} {PH[ph]} {ev.lower()} {v:+.2f} z" for v, m, ph, ev in tops)
              + f"; {len(nup)} neuron x event x session combinations passed the single-neuron test.", pv(k),
              howto="How to read: each row = one neuron (4 mice stacked), each small panel = its average activity from 2 s before "
                    "to 5 s after the event. Red after 0 = more active after the event, blue = less; base / inj. = the two sessions."); k += 1
    R8 = pd.read_csv(C.A(B, "ca_responsive_change", "R8_responsive_calibrated.csv"))
    s_mv = int((R8[R8.event == "movement onset"].maxT_up > 0).sum()); n_mv = int((R8.event == "movement onset").sum())
    s_st = int((R8[R8.event.isin(STIM)].maxT_up > 0).sum()); n_st = int(R8.event.isin(STIM).sum())
    add_slide(prs, "NEURONS", f"Single neurons respond to movement in {s_mv}/{n_mv} sessions, to a stimulus in {s_st}/{n_st}",
              f("s08_responsive_counts.png"), n_cnt,
              f"With the strict test at most {up_stim_max:.0f}% of the neurons responded to a stimulus, up to {up_mv_max:.0f}% to movement "
              f"onset. The population beat chance for movement onset in {int((mv.q_energy <= 0.05).sum())}/{len(mv)} sessions and for "
              f"the stimuli and withdrawal in {int((st.q_energy <= 0.05).sum())}/{len(st)}.", pv(k),
              howto="How to read: one dot = one mouse in one session (open = baseline, filled = after injection, a line joins them). "
                    "A = % of neurons that pass the strict single-neuron test; B = whole-population response, 1 = no response."); k += 1
    add_slide(prs, "NEURONS", "Responsive neurons, activated and inhibited, baseline vs after injection", f("s08b_responsive_change.png"), n_rc,
              "We compared the responsive neurons' response before and after the injection. Result (p < 0.05): activated groups "
              f"weaker in {rcd['activated'][0]}, stronger in {rcd['activated'][1]} of {nsig['activated'][0]}; inhibited groups weaker in "
              f"{rcd['inhibited'][0]}, stronger in {rcd['inhibited'][1]} of {nsig['inhibited'][0]}.", pv(k),
              howto="How to read: top = neurons that ROSE after the event at baseline, bottom = neurons that FELL; per event, two bars per "
                    "mouse: their response at baseline (open) and after the injection (filled). Closer to 0 = weaker response."); k += 1
    add_slide(prs, "NEURONS", (f"Stimulus-block activity after the injection: largest drop in {big} ({ARM[big]})" if ch[big] < 0
                               else "Stimulus-block activity after the injection, per mouse"),
              f("s09_activity_per_mouse.png"), n_act,
              "We counted Ca transients per minute of still frames for every neuron. Median change per neuron in the stimulus blocks: "
              + ", ".join(f"{m} {ch[m]:+.0f}%" for m in ch) + "; in the first 3 min without stimulus: "
              + ", ".join(f"{m} {ch0[m]:+.0f}%" for m in ch0) + ".", pv(k),
              howto="How to read: each dot = one neuron's activity (transients per minute while the mouse is still); a grey line joins "
                    "the same neuron in the two sessions; black bar = median. A = stimulus blocks, B = the first 3 min."); k += 1
    t4 = "SBI-553 vs vehicle: change per mouse and the group difference"
    add_slide(prs, "NEURONS", t4, f("s10_sbi_vs_vehicle.png"), n_arm,
              f"We repeated the comparison under {len(q)} analysis choices. Result: SBI-553 decreased more than vehicle in "
              f"{int((q.sbi_minus_vehicle < 0).sum())}/{len(q)}; group difference {all4:+.2f} log2 with all 4 mice, {l5941:+.2f} without {big}.", pv(k),
              howto="How to read: A = each mouse's change (below 0 = less activity after the injection; grey = range over analysis "
                    "choices); B = SBI-553 minus vehicle when one mouse is left out; C = the same estimated by a mixed model."); k += 1
    add_slide(prs, "BEHAVIOUR", "What we score: reflexive, affective and automatic behaviours", f("s11_behaviour_examples.png"), n_bex,
              "Two cameras (side, bottom) were scored by hand for deliveries, reflexes and affective behaviours, blind to the "
              "treatment; movement and grooming were detected automatically and checked by eye.", pv(k),
              howto="How to read: one video frame for each behaviour we score, with its definition and the camera it is scored on."); k += 1
    sig = Rz[(Rz.p < 0.05)]
    add_slide(prs, "BEHAVIOUR", "Behaviour per delivery, baseline vs injection, every mouse", f("s12_behaviour_per_delivery.png"), n_bf,
              "Every behaviour is divided by the number of deliveries. Result (p < 0.05): " + "; ".join(
                  f"{b} {'increased' if d > 0 else 'decreased'} in {', '.join(ms)}" for (b, d), ms in sig_dir.items()) + ".", pv(k),
              howto="How to read: one line = one mouse, from baseline (open dot) to after the injection (filled dot); y = how many "
                    "times the behaviour followed a delivery, divided by the number of deliveries (stars = Poisson test)."); k += 1
    # (behaviour sentence above lists the p < 0.05 changes)
    add_slide(prs, "BEHAVIOUR", "Reflexes and affective behaviour per stimulus", f("s13_behaviour_per_stimulus.png"), n_bs,
              f"Result: withdrawal per delivery was higher for pin prick or heat than for light touch in {n_grad}/{n_gs} mouse-sessions. "
              "Time moving before any stimulus (baseline -> injection): " + ", ".join(f"{m} {mvp[m][0]:.0f}->{mvp[m][1]:.0f}%" for m, _, _ in MICE) + ".", pv(k),
              howto="How to read: A-C = the same per-delivery measure split by stimulus, one short line per mouse (open = baseline, "
                    "filled = after injection); D = % of time the mouse moved before any stimulus was given."); k += 1
    dn = R4[R4.cls == "DOWN"]
    if not limits:
        return k
    add_slide(prs, "LIMITS", "Limits of the current data", f("s14_caveats.png"), n_cav,
              f"{100 * (dn.speed > 0.53).mean():.0f}% of the DOWN responses are faster than Ca allows, so we report UP responses only; "
              f"with {int(tr.min())}-{int(tr.max())} trials per stimulus and session the single-neuron response maps are not stable. "
              + ("More trials and Day 2 will address both." if C.DAY == 1 else "More trials will address both."), pv(k),
              howto="How to read: A = how fast responses start (grey zone = the speeds a calcium decrease can have); B = whether the "
                    "same neurons respond in two random halves of the trials (1 = always the same neurons, 0 = chance).")
    return k


if __name__ == "__main__":
    main()
