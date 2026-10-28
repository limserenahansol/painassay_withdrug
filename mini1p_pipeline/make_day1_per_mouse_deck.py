"""make_day1_per_mouse_deck.py - the per-mouse supplement of the Day-1 deck (the Ntsr1 mini2p deck's slides,
for every mouse). Eight slides per mouse, the same order for each:

  1  every neuron, labelled               max projection plain | with every analysed neuron outlined + numbered
  2  every neuron over the whole session  heatmap, baseline and after injection, same neuron order
  3  baseline: all neurons + scored events every trace (colour = response class) and the behaviour raster
  4  injection: all neurons + scored events
  5  where the responsive neurons are     one FOV map per event, baseline and injection (red UP, blue DOWN)
  6  UP, DOWN and other neurons           mean +- SEM response of each class, per event and session
  7  population response                  all neurons: mean response per event, baseline vs injection
  8  how many neurons go up or down       counts per event with the population test

Every slide: a title, a 'how to read' line, the graph note (tests, n) and one or two sentences of what
we did and found (numbers read from the analysis files). The build stops if the analysed cells differ
from the confirmed selection.

  python make_day1_per_mouse_deck.py --base C:/Users/hsollim/Documents/HEAL_mini1p_SBI553 --out <pptx>
"""
import argparse
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
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
import make_day1_ca_slides as S1                                    # noqa: E402
import make_day1_full_deck as FD                                    # noqa: E402
import deck_cfg as C                                                # noqa: E402  (CA_DAY: which day)
import pairstars as PS  # noqa: E402  (user rule 10-22: every star shows the pair it compares)
DC = C                                                              # (fig_fov has a local C)
import ca_day1_analysis as A                                        # noqa: E402

MICE, ARMC, STIM, FW, FH = S1.MICE, S1.ARMC, S1.STIM, S1.FW, S1.FH
ARM = {m: a for m, a, _ in MICE}
SEX = {m: s for m, _, s in MICE}
EV7 = STIM + ["withdrawal", "flinch", "movement onset"]
ECOL = {"Light touch": "#5dade2", "Mild touch": "#f5b041", "Pin prick": "#e74c3c", "Heat": "#8e44ad",
        "withdrawal": "#6c3483", "flinch": "#1e8449", "movement onset": "#17a2b8"}
PH = {"pre": "baseline", "post": "after injection"}
UPC, DNC, NOC = "#c0392b", "#2471a3", "0.55"


def pstr(p):
    return "n.a." if not np.isfinite(p) else ("p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}")


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def responses(SS):
    out, ntr, tt = {}, {}, None
    for (m, ph), s in SS.items():
        for ev in EV7:
            X, tt0, used = A.peri(s, s["B"]["ev"][ev])
            if X is None or len(used) < A.MIN_TRIALS:
                continue
            tt = tt0
            pre = (tt >= A.PRE_W[0]) & (tt < A.PRE_W[1])
            mu = np.nanmean(X, 0)
            out[(m, ph, ev)] = mu - np.nanmean(mu[pre], 0, keepdims=True)
            ntr[(m, ph, ev)] = len(used)
    return out, ntr, tt


def classes(ET, m, ph, ev, ids):
    E = ET[(ET.mouse == m) & (ET.phase == ph) & (ET.event == ev)].set_index("cell")
    return np.array([E.cls_shift.get(c, "no response") if c in E.index else "no response" for c in ids])


def contours_of(S, ids):
    out = {}
    for c in ids:
        f = S[:, :, c - 1]
        if f.max() <= 0:
            continue
        cs, _ = cv2.findContours((f >= 0.3 * f.max()).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out[c] = [cc.reshape(-1, 2) for cc in cs if len(cc) > 2]
    return out


def load_fov(B, m):
    tag = f"{m}{C.T}"
    S = np.asarray(sio.loadmat(FD.mpath(B, tag, "3_extract", f"{tag}_extract.mat"), variable_names=["spatial_weights"])["spatial_weights"], np.float32)
    R = pd.read_csv(FD.mpath(B, tag, "4_cell_review", f"{tag}_cell_review.csv"))
    bg = np.asarray(sio.loadmat(FD.mpath(B, tag, "3_extract", f"{tag}_projections.mat"), variable_names=["max_dff"])["max_dff"], float)
    return S, R, FD.norm01(bg, 1, 99.8, 0.7)


# ------------------------------------------------------------------------------------- figures
def fig_fov(B, m, out):
    S, R, img = load_fov(B, m)
    att = DC.one_day_flags(B, m, R)
    keep = R[R.keep == 1]
    C = contours_of(S, keep.id.astype(int).values)
    fig, ax = plt.subplots(1, 2, figsize=(FW, FH))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93 if DC.DAY == 1 else 0.88, bottom=0.01, wspace=0.03)
    for k in (0, 1):
        ax[k].imshow(img, cmap="gray"); ax[k].set_axis_off()
    ax[0].set_title("max dF/F projection of the whole day", fontsize=11)
    ax[1].set_title(f"{len(keep)} analysed neurons (numbers = Day-{DC.DAY} ids)\ngreen = on both days, "
                    + ("magenta" if DC.DAY == 1 else "cyan") + f" = {DC.ONE_LAB}", fontsize=10)
    for _, r in keep.iterrows():
        c = int(r.id)
        col = DC.ONE_COL if att.loc[_] else "#2ecc40"
        for cc in C.get(c, []):
            ax[1].plot(cc[:, 0], cc[:, 1], color=col, lw=0.9)
        ax[1].text(r.col + 5, r.row - 5, str(c), color=col, fontsize=6.5, fontweight="bold")
    fig.savefig(out, dpi=170); plt.close(fig)
    nm = int((att.loc[keep.index] == "").sum())
    return len(keep), nm, len(keep) - nm


def fig_traces(SS, ET, m, ph, out):
    s = SS[(m, ph)]
    T, Bh = s["T"], s["B"]
    ids = T["cell_ids"].astype(int)
    t = T["t_s"]
    Z = T["z"]
    up = np.zeros(len(ids), bool); dn = np.zeros(len(ids), bool)
    for ev in STIM:
        cl = classes(ET, m, ph, ev, ids)
        up |= cl == "UP"; dn |= cl == "DOWN"
    dn &= ~up
    k = 6                                                              # 5 Hz for display
    n5 = len(t) // k
    t5 = t[:n5 * k].reshape(n5, k).mean(1)
    Z5 = np.nanmean(Z[:n5 * k].reshape(n5, k, -1), 1)
    half = int(np.ceil(len(ids) / 2))
    fig = plt.figure(figsize=(FW, FH))
    for col, sel in enumerate((np.arange(half), np.arange(half, len(ids)))):
        x0 = 0.045 + col * 0.49
        ax = fig.add_axes([x0, 0.30, 0.445, 0.68])
        for i, j in enumerate(sel):
            z = Z5[:, j]
            lo, hi = np.nanmedian(z), np.nanpercentile(z, 99.5)
            y = (z - lo) / (hi - lo + 1e-9) * 0.85
            c = UPC if up[j] else DNC if dn[j] else NOC
            ax.plot(t5, y + (len(sel) - i), lw=0.35, color=c)
            ax.text(t5[0] - 8, len(sel) - i + 0.3, str(ids[j]), ha="right", va="center", fontsize=4.6, color=c)
        ax.set_xlim(0, t5[-1]); ax.set_ylim(0.3, len(sel) + 1.3)
        ax.set_yticks([]); ax.set_xticks([])
        for sp in ("top", "right", "left", "bottom"):
            ax.spines[sp].set_visible(False)
        for b0, b1 in [(180, 360), (390, 570), (600, 780), (810, 990)]:
            ax.axvspan(b0, b1, color="#f5b041", alpha=0.07, lw=0)
        # behaviour raster
        rb = fig.add_axes([x0, 0.05, 0.445, 0.23], sharex=ax)
        rows = STIM + ["withdrawal", "flinch", "movement"]
        for r_i, nm in enumerate(rows):
            yy = len(rows) - 1 - r_i
            if nm == "movement":
                mv = s["B"]["st"]["move"]
                on, off = FD.BA.spans(mv)
                for a0, a1 in zip(on, off):
                    rb.plot([a0 / 30.0, a1 / 30.0], [yy, yy], color="#17a2b8", lw=3, solid_capstyle="butt")
            else:
                f = Bh["ev"][nm]
                rb.plot(f / 30.0, np.full(len(f), yy), "|", color=ECOL[nm], ms=6, mew=1.0)
        rb.set_yticks(range(len(rows)), [f"{r} ({len(Bh['ev'][r])})" if r != "movement" else "movement" for r in rows][::-1], fontsize=7)
        rb.set_ylim(-0.7, len(rows) - 0.3)
        rb.set_xlabel("time in session (s)", fontsize=8.5); rb.tick_params(axis="x", labelsize=8)
        for sp in ("top", "right"):
            rb.spines[sp].set_visible(False)
    fig.savefig(out, dpi=170); plt.close(fig)
    return int(up.sum()), int(dn.sum()), int(len(ids) - up.sum() - dn.sum())


def fig_maps(B, SS, ET, m, out):
    S, R, img = load_fov(B, m)
    ids = SS[(m, "pre")]["T"]["cell_ids"].astype(int)
    C = contours_of(S, ids)
    pos = R.set_index("id")
    fig = plt.figure(figsize=(FW, FH))
    w = 0.134; h = w * FW / FH * img.shape[0] / img.shape[1]
    counts = {}
    for r_i, ph in enumerate(("pre", "post")):
        y0 = 0.52 - r_i * 0.47
        for e_i, ev in enumerate(EV7):
            ax = fig.add_axes([0.035 + e_i * (w + 0.005), y0, w, h])
            ax.imshow(img, cmap="gray", alpha=0.85); ax.set_axis_off()
            cl = classes(ET, m, ph, ev, ids)
            for c, k in zip(ids, cl):
                col = UPC if k == "UP" else DNC if k == "DOWN" else "#bbbbbb"
                lw = 1.0 if k in ("UP", "DOWN") else 0.35
                for cc in C.get(c, []):
                    ax.plot(cc[:, 0], cc[:, 1], color=col, lw=lw)
                if k in ("UP", "DOWN"):
                    ax.text(pos.at[c, "col"] + 5, pos.at[c, "row"] - 5, str(c), color="white", fontsize=4.8, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.12", fc=col, ec="none"))
            nu, nd = int((cl == "UP").sum()), int((cl == "DOWN").sum())
            counts[(ph, ev)] = (nu, nd)
            ax.text(0.02, 0.02, f"{nu} up, {nd} down", transform=ax.transAxes, fontsize=7, color="white", fontweight="bold", va="bottom")
            if r_i == 0:
                ax.set_title(ev, fontsize=9.5, color=ECOL[ev], fontweight="bold", pad=3)
        fig.text(0.012, y0 + h / 2, PH[ph], rotation=90, va="center", ha="center", fontsize=10)
    fig.savefig(out, dpi=170); plt.close(fig)
    return counts


def fig_groups(RESP, ET, SS, m, tt, out):
    fig, ax = plt.subplots(2, len(EV7), figsize=(FW, FH), sharex=True)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.9, bottom=0.12, wspace=0.42, hspace=0.38)
    best = {}
    show = (tt >= -3) & (tt <= 6)
    for r_i, ph in enumerate(("pre", "post")):
        ids = SS[(m, ph)]["T"]["cell_ids"].astype(int)
        for e_i, ev in enumerate(EV7):
            a = ax[r_i, e_i]
            M = RESP.get((m, ph, ev))
            if M is None:
                a.set_axis_off(); continue
            Ms = A.sm(M)
            cl = classes(ET, m, ph, ev, ids)
            for k, col in (("no response", NOC), ("DOWN", DNC), ("UP", UPC)):
                sel = cl == k
                if not sel.any():
                    continue
                mu = np.nanmean(Ms[:, sel], 1)
                se = np.nanstd(Ms[:, sel], 1) / np.sqrt(max(sel.sum(), 1))
                a.fill_between(tt[show], (mu - se)[show], (mu + se)[show], color=col, alpha=0.25, lw=0)
                a.plot(tt[show], mu[show], color=col, lw=1.1, label=f"{k if k != 'no response' else 'rest'} ({int(sel.sum())})")
                if k == "UP":
                    best[(ph, ev)] = float(np.nanmean(M[(tt >= 0) & (tt < 2)][:, sel]))
            a.axvspan(0, 2, color="0.9", zorder=0, lw=0)
            a.axvline(0, color="k", lw=0.6)
            a.axhline(0, color="0.6", lw=0.5)
            a.tick_params(labelsize=7)
            a.legend(fontsize=6, frameon=False, loc="upper left", handlelength=1)
            if r_i == 0:
                a.set_title(ev, fontsize=9.5, color=ECOL[ev], fontweight="bold")
            if e_i == 0:
                a.set_ylabel(f"{PH[ph]}\nz, minus [-2, 0] s", fontsize=8.5)
            if r_i == 1:
                a.set_xlabel("s from event", fontsize=8)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    fig.savefig(out, dpi=160); plt.close(fig)
    return best


def fig_population(RESP, SS, m, tt, out):
    fig, ax = plt.subplots(2, len(EV7), figsize=(FW, FH), gridspec_kw=dict(height_ratios=[1.25, 1]))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.12, wspace=0.45, hspace=0.45)
    show = (tt >= -3) & (tt <= 6); w = (tt >= 0) & (tt < 2)
    res = {}
    col1 = ARMC[ARM[m]]
    for e_i, ev in enumerate(EV7):
        a, b = ax[0, e_i], ax[1, e_i]
        r = {}
        for ph, col in (("pre", "0.45"), ("post", col1)):
            M = RESP.get((m, ph, ev))
            if M is None:
                continue
            Ms = A.sm(M)
            mu = np.nanmean(Ms, 1); se = np.nanstd(Ms, 1) / np.sqrt(Ms.shape[1])
            a.fill_between(tt[show], (mu - se)[show], (mu + se)[show], color=col, alpha=0.25, lw=0)
            a.plot(tt[show], mu[show], color=col, lw=1.1, label=PH[ph])
            r[ph] = np.nanmean(M[w], 0)
        a.axvspan(0, 2, color="0.92", zorder=0, lw=0); a.axvline(0, color="k", lw=0.6); a.axhline(0, color="0.6", lw=0.5)
        a.set_title(ev, fontsize=9.5, color=ECOL[ev], fontweight="bold"); a.tick_params(labelsize=7)
        a.set_xlabel("s from event", fontsize=8)
        if e_i == 0:
            a.set_ylabel("mean over neurons\n(z, minus [-2, 0] s)", fontsize=8.5)
            a.legend(fontsize=7, frameon=False, loc="upper left")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        pv = {}
        for i, (ph, col) in enumerate((("pre", "0.45"), ("post", col1))):
            if ph not in r:
                continue
            v = r[ph][np.isfinite(r[ph])]
            pv[ph] = wilcoxon(v).pvalue if len(v) >= 5 and np.any(v != 0) else np.nan
            b.bar(i, np.mean(v), yerr=np.std(v) / np.sqrt(len(v)), color=col, width=0.6, capsize=3)
            PS.vs_ref(b, i, 0.98, pv[ph], "vs 0", "k", fs=7.5, show_ns=False, axes_y=True, va="top")   # that bar vs 0
        pp = np.nan
        if "pre" in r and "post" in r:
            d = r["post"] - r["pre"]; d = d[np.isfinite(d)]
            pp = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan
        res[ev] = (pv.get("pre", np.nan), pv.get("post", np.nan), pp,
                   float(np.nanmean(r["pre"])) if "pre" in r else np.nan, float(np.nanmean(r["post"])) if "post" in r else np.nan)
        b.axhline(0, color="0.4", lw=0.6)
        b.set_xticks([0, 1], ["base", "inj."], fontsize=8); b.tick_params(axis="y", labelsize=7)
        b.set_title(f"base vs inj. {pstr(pp)}", fontsize=8)
        if e_i == 0:
            b.set_ylabel("response [0, 2] s\n(mean +- SEM)", fontsize=8.5)
        for sp in ("top", "right"):
            b.spines[sp].set_visible(False)
    fig.savefig(out, dpi=160); plt.close(fig)
    return res


def fig_counts(ET, R7, SS, m, out):
    fig, ax = plt.subplots(1, 2, figsize=(FW, FH), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.93, top=0.86, bottom=0.14, wspace=0.55)
    sig = {}
    for k, ph in enumerate(("pre", "post")):
        a = ax[k]
        ids = SS[(m, ph)]["T"]["cell_ids"].astype(int)
        n = len(ids)
        for y, ev in enumerate(EV7[::-1]):
            cl = classes(ET, m, ph, ev, ids)
            nd, nu = int((cl == "DOWN").sum()), int((cl == "UP").sum())
            a.barh(y, nd, color=DNC, height=0.62)
            a.barh(y, n - nd - nu, left=nd, color="0.86", height=0.62)
            a.barh(y, nu, left=n - nu, color="#e69f00", height=0.62)
            r = R7[(R7.mouse == m) & (R7.phase == ph) & (R7.event == ev)]
            if len(r):
                p, q = float(r.p_energy.iloc[0]), float(r.q_energy.iloc[0])
                sig[(ph, ev)] = q
                a.text(n * 1.03, y, f"{pstr(p)}{' *' if q <= 0.05 else ''}", va="center", fontsize=9)
        ntr = {ev: len(SS[(m, ph)]["B"]["ev"][ev]) for ev in EV7}
        a.set_yticks(range(len(EV7)), [f"{e} ({ntr[e]})" for e in EV7[::-1]], fontsize=9.5)
        a.set_xlim(0, n); a.set_xticks([0, n]); a.set_xlabel(f"neurons (n = {n})")
        a.set_title(f"{PH[ph]} session", fontsize=11, loc="left")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.legend(handles=[Patch(color=DNC, label="down"), Patch(color="0.86", label="unchanged"), Patch(color="#e69f00", label="up")],
               loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(out, dpi=160); plt.close(fig)
    return sig


EVM = STIM + ["withdrawal", "flinch", "attending onset", "lick/bite onset", "escape/rearing onset", "movement onset", "grooming onset"]
SHORTM = {"Light touch": "light touch", "Mild touch": "mild touch", "Pin prick": "pin prick", "Heat": "heat",
          "withdrawal": "withdrawal", "flinch": "flinch", "attending onset": "attending", "lick/bite onset": "lick/bite",
          "escape/rearing onset": "escape/rearing", "movement onset": "movement", "grooming onset": "grooming"}


def fig_matrix(TA, m, out):
    """neurons (columns, by number) x events (rows, baseline and injection): red UP, blue DOWN"""
    X = TA[TA["mouse"].astype(str) == m]
    ids = sorted(X.cell.unique())
    evs = [e for e in EVM if e in set(X.event)]
    rows, lab = [], []
    for e in evs:
        for ph in ("pre", "post"):
            Y = X[(X.phase == ph) & (X.event == e)].set_index("cell")
            v = []
            for c in ids:
                k = Y.cls_shift.get(c, None) if c in Y.index else None
                tested = c in Y.index and np.isfinite(Y.q_shift.get(c, np.nan))
                v.append(np.nan if not tested else 1 if k == "UP" else -1 if k == "DOWN" else 0)
            rows.append(v); lab.append(f"{SHORTM[e]}  {'base' if ph == 'pre' else 'inj.'}")
    M = np.array(rows, float)
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.13, 0.14, 0.80, 0.80])
    cmap = matplotlib.colors.ListedColormap([DNC, "#f2f2f2", UPC])
    ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=-1.5, vmax=1.5, interpolation="nearest")
    yy, xx = np.where(np.isnan(M))
    ax.scatter(xx, yy, marker="x", s=4, color="0.75", lw=0.4)
    ax.set_yticks(range(len(lab)), lab, fontsize=7)
    for i in range(2, len(lab), 2):
        ax.axhline(i - 0.5, color="k", lw=0.6)
    ax.set_xticks(range(len(ids)), [str(c) for c in ids], fontsize=4.8, rotation=90)
    ax.set_xlabel("neuron number (as on the field-of-view map)", fontsize=9)
    # how many events each neuron responds to (UP, either session)
    nup = np.nansum(M == 1, 0)
    ax2 = fig.add_axes([0.13, 0.945, 0.80, 0.045], sharex=ax)
    ax2.bar(range(len(ids)), nup, color=UPC, width=0.8); ax2.set_axis_off()
    ax2.text(-1, nup.max() if len(nup) else 1, "UP rows per neuron", ha="right", va="top", fontsize=7)
    ax3 = fig.add_axes([0.94, 0.14, 0.05, 0.80], sharey=ax)
    ax3.barh(range(len(lab)), np.nansum(M == 1, 1), color=UPC, height=0.8)
    ax3.barh(range(len(lab)), -np.nansum(M == -1, 1), color=DNC, height=0.8)
    ax3.axvline(0, color="k", lw=0.5); ax3.tick_params(labelsize=6); ax3.set_yticks([])
    ax3.set_xlabel("n DOWN | UP", fontsize=6.5)
    fig.savefig(out, dpi=170); plt.close(fig)
    multi = int(((M == 1).reshape(len(evs), 2, -1)[:, 0, :].sum(0) >= 2).sum())       # >= 2 events at baseline
    best = sorted(((int(np.nansum(M[2 * i] == 1)), SHORTM[e]) for i, e in enumerate(evs)), reverse=True)[:3]
    return len(ids), multi, best


def fig_resp_change(SP, m, out):
    """split-half: activated / inhibited neurons, baseline (held-out half) vs injection, per event; one dot per neuron"""
    X = SP[SP["mouse"].astype(str) == m]
    evs = [e for e in EVM if e in set(X.event)]
    fig, ax = plt.subplots(2, max(len(evs), 1), figsize=(FW, FH), sharey="row", squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.12, wspace=0.15, hspace=0.42)
    col = ARMC[ARM[m]]
    out_p = {}
    for r_i, k in enumerate(("activated", "inhibited")):
        for e_i, e in enumerate(evs):
            a = ax[r_i, e_i]
            g = X[(X.event == e) & (X.group == k)]
            if len(g):
                b0, b1 = g.base_heldout.values, g.inj.values
                a.bar(0, b0.mean(), color="white", edgecolor=col, width=0.7, lw=1.2)
                a.bar(1, b1.mean(), color=col, width=0.7)
                for u, v in zip(b0, b1):
                    a.plot([0, 1], [u, v], color="0.6", lw=0.4, zorder=3)
                p = wilcoxon(b1 - b0).pvalue if len(g) >= 5 and np.any(b1 != b0) else np.nan
                out_p[(k, e)] = (len(g), b0.mean(), b1.mean(), p)
                a.set_title(f"{SHORTM[e]}\nn = {len(g)}  {pstr(p) if np.isfinite(p) else ''}", fontsize=7.5)
            else:
                a.set_title(f"{SHORTM[e]}\nn = 0", fontsize=7.5)
            a.axhline(0, color="0.4", lw=0.6)
            a.set_xticks([0, 1], ["base", "inj."], fontsize=7); a.tick_params(axis="y", labelsize=7)
            if e_i == 0:
                a.set_ylabel(f"{k} neurons\nresponse (z)", fontsize=8.5)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    fig.savefig(out, dpi=160); plt.close(fig)
    return out_p


def fig_guide(out):
    rows = [["1", "every neuron, labelled", "the field of view and the number of every neuron"],
            ["2", "every neuron over the whole session", "heatmap of all neurons, baseline and after injection"],
            ["3", "baseline: all neurons + scored events", "every trace (red = responds to a stimulus) and the behaviour"],
            ["4", "after injection: all neurons + scored events", "the same for the second session"],
            ["5", "where the responsive neurons are", "one map per event: red = rises (UP), blue = falls (DOWN)"],
            ["6", "UP, DOWN and the other neurons", "average response of each group, per event and session"],
            ["7", "population response", "all neurons together, baseline vs after injection"],
            ["8", "how many neurons go up or down", "counts per event and the population test"],
            ["9", "which neuron responds to what", "neuron x event matrix (stimuli and behaviours), both sessions"],
            ["10", "responsive neurons after the injection", "activated and inhibited neurons: baseline vs injection response"]]
    fig = plt.figure(figsize=(FW, FH)); ax = fig.add_axes([0.03, 0.05, 0.94, 0.9]); ax.set_axis_off()
    T = ax.table(cellText=rows, colLabels=["slide", "title", "what it shows"], loc="upper left", cellLoc="left", colLoc="left",
                 colWidths=[0.07, 0.38, 0.55])
    T.auto_set_font_size(False); T.set_fontsize(11); T.scale(1, 2.1)
    for (r, c), cell in T.get_celld().items():
        cell.set_edgecolor("0.85")
        if r == 0:
            cell.set_text_props(fontweight="bold"); cell.set_facecolor("#f3f5f8")
    fig.savefig(out, dpi=160); plt.close(fig)


# ------------------------------------------------------------------------------------------ deck
HOW = {
    9: "How to read: each column = one neuron (number as on slide 1), each row = one event in one session; red = the neuron rose "
       "after that event, blue = it fell, light grey = no response. Bars: how many events per neuron (top), neurons per row (right).",
    10: "How to read: neurons that rose (top) or fell (bottom) after an event at baseline; open bar = their baseline response, "
        "filled = after the injection, one grey line per neuron. Closer to 0 = a weaker response after the injection.",
    1: "How to read: left = the whole-day max projection; right = the same with every analysed neuron outlined and numbered. "
       "The numbers are the same neurons on every slide of this mouse and in both sessions.",
    2: "How to read: each row = one neuron, time runs left to right over the whole session, bright = active. Coloured ticks on "
       "top = deliveries by stimulus. The two sessions show the same neurons in the same order.",
    3: "How to read: one line per neuron (number on the left; red = rose after at least one stimulus, blue = only fell, grey = "
       "no response); below, on the same time axis, every scored delivery, reflex and movement bout.",
    5: "How to read: one map per event (columns) and session (rows); a red outline = the neuron rose after the event (UP), blue "
       "= it fell (DOWN), thin grey = no response. Counts at the bottom of each map.",
    6: "How to read: the average response of the UP neurons (red), the DOWN neurons (blue) and the rest (grey), mean +- SEM "
       "over neurons; n in brackets. Grey band = the [0, 2] s window that the test uses.",
    7: "How to read: top = the average of ALL neurons around each event (grey = baseline, colour = after injection); bottom = "
       "the response in [0, 2] s per session (star over 'vs 0' = that bar differs from 0) and the paired comparison of the two sessions (title: base vs inj.).",
    8: "How to read: for every event, how many neurons fell (blue), did not change (grey) or rose (orange); in brackets the "
       "number of events. Right: the population test (* = q <= 0.05 over all 56 tests of the 4 mice).",
}
NOTE = {
    9: "Time-shift test per neuron and event (each session separately): response z [0, 2] s minus [-2, 0] s vs the same events "
       "shifted inside their block, BH q <= 0.20 over neurons (exploratory). x = fewer than 5 events, not tested.",
    10: "Selected on one half of the baseline trials (Wilcoxon signed-rank p < 0.05), measured on the other half (open bar) and on "
        "all injection trials (filled); grey lines = single neurons; title: n neurons and paired Wilcoxon p.",
    3: "Traces: dF/F divided by the neuron's noise SD (one scale for both sessions), 0.2-s bins, each scaled to its own "
       "range. Class from the time-shift test on the 4 stimuli (BH q <= 0.20, exploratory). Orange shading = stimulus blocks.",
    5: "Time-shift test per neuron: the event's response vs the same events shifted in time inside their block; BH q <= 0.20 "
       "over the neurons (exploratory). DOWN responses are mostly whole-field artefacts (see the main deck, Limits).",
    6: "Response = trial-mean z minus its [-2, 0] s mean, 0.2-s running mean for display. Classes as on slide 5.",
    7: "Response per neuron = mean of [0, 2] s of the trial-mean z (baseline-subtracted). Wilcoxon signed-rank over neurons: vs 0 "
       "in each session (* p<0.05, ** <0.01, *** <0.001) and baseline vs injection (paired, same neurons).",
    8: "Population test = RMS over neurons of the mean response vs the same events shifted in their block (2000 shifts), "
       "BH over the 56 tests of all mice. Single-neuron classes as on slide 5.",
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(S1.SW), Inches(S1.SH)
    od = os.path.splitext(a.out)[0] + "_figures"
    n = build_per_mouse(a.base, prs, od)
    prs.save(a.out)
    print("->", a.out, "|", n, "slides | figures + previews in", od)


def build_per_mouse(B, prs, od, pvf=None, section="PER-MOUSE SUPPLEMENT"):
    os.makedirs(od, exist_ok=True)
    f = lambda nm: os.path.join(od, nm)
    W = S1.whole(B)
    nk = S1.kept_counts(B)
    ncell = {m: int(W[(W.mouse == m) & (W.phase == "pre")].cell.nunique()) for m, _, _ in MICE}
    bad = {m: (ncell[m], nk[m][0]) for m, _, _ in MICE if ncell[m] != nk[m][0]}
    if bad:
        raise SystemExit(f"analysis cells differ from the confirmed selection (analysis, selection): {bad}")
    SS = FD.sessions_noise_z(B)
    TA = pd.read_csv(C.A(B, "ca_cell_event_matrix", "cell_event_tests_all.csv")); TA["mouse"] = TA.mouse.astype(str)
    SP = pd.read_csv(C.A(B, "ca_cell_event_matrix", "responsive_change_split_half.csv")); SP["mouse"] = SP.mouse.astype(str)
    RESP, NTR, tt = responses(SS)
    ET = pd.read_csv(C.A(B, "ca_day1", "ca_d1_cells_event_tests.csv")); ET["mouse"] = ET.mouse.astype(str)
    R7 = pd.read_csv(C.A(B, "ca_responsive_change", "R7_population_response_test.csv")); R7["mouse"] = R7.mouse.astype(str)
    R1 = pd.read_csv(C.A(B, "ca_robustness", "R1_robustness_per_mouse.csv")); R1["mouse"] = R1.mouse.astype(str)
    main_v = "main: OASIS transients, still frames"
    ch = {r.mouse: 100 * (2 ** r.log2_ratio_median - 1) for r in R1[(R1.window == "stimulus blocks") & (R1.variant == main_v)].itertuples()}
    ch0 = {r.mouse: 100 * (2 ** r.log2_ratio_median - 1) for r in R1[(R1.window == "no-stimulus start") & (R1.variant == main_v)].itertuples()}
    k = 1
    pv = pvf if pvf else (lambda: f(f"preview_slide{k:02d}.png"))
    fig_guide(f("guide.png"))
    FD.add_slide(prs, section, f"{C.LABEL} per mouse: the same ten slides for each of the four mice", f("guide.png"), "",
                 "The main deck shows the four mice together; here every mouse is shown on its own, in the layout of the CeA-Ntsr1 "
                 f"mini2p deck. Mice: {C.GROUPS_TXT}.", pv(),
                 howto="How to read: the table lists the ten slides that follow for every mouse, in the same order."); k += 1
    for m, arm, sex in MICE:
        sec = f"MOUSE {m} ({arm}, {sex})"
        n, nm, nr = fig_fov(B, m, f(f"{m}_1_fov.png"))
        FD.add_slide(prs, sec, f"{m}: every neuron, labelled", f(f"{m}_1_fov.png"), "",
                     f"{n} neurons: {nm} on both days (green) and {nr} on {C.ONE_LAB} ({'magenta' if C.DAY == 1 else 'cyan'}). "
                     + "The same field of view is used for both sessions.", pv(), howto=HOW[1]); k += 1
        note2 = FD.fig_whole_heatmap(SS, f(f"{m}_2_heatmap.png"), m=m)
        FD.add_slide(prs, sec, f"{m}: every neuron over the whole session", f(f"{m}_2_heatmap.png"), note2,
                     f"Median change per neuron after the injection: stimulus blocks {ch[m]:+.0f}%, first 3 min without stimulus "
                     f"{ch0[m]:+.0f}% (transients per minute of still frames).", pv(), howto=HOW[2]); k += 1
        for ph in ("pre", "post"):
            nu, nd, nn = fig_traces(SS, ET, m, ph, f(f"{m}_3_traces_{ph}.png"))
            nev = {e: len(SS[(m, ph)]["B"]["ev"][e]) for e in STIM}
            FD.add_slide(prs, sec, f"{m} {PH[ph]}: all neurons with the scored events", f(f"{m}_3_traces_{ph}.png"), NOTE[3],
                         f"{nu} neurons rose after at least one stimulus (red), {nd} only fell (blue) and {nn} did not respond (grey). "
                         "Deliveries: " + ", ".join(f"{e.lower()} {nev[e]}" for e in STIM) + ".", pv(), howto=HOW[3]); k += 1
        cnt = fig_maps(B, SS, ET, m, f(f"{m}_5_maps.png"))
        up_b = sorted(((cnt[("pre", e)][0], e) for e in EV7), reverse=True)
        up_i = sorted(((cnt[("post", e)][0], e) for e in EV7), reverse=True)
        FD.add_slide(prs, sec, f"{m}: where the responsive neurons are", f(f"{m}_5_maps.png"), NOTE[5],
                     "UP neurons per event, baseline: " + ", ".join(f"{e.lower()} {cnt[('pre', e)][0]}" for e in EV7)
                     + "; after injection: " + ", ".join(f"{e.lower()} {cnt[('post', e)][0]}" for e in EV7) + ".", pv(), howto=HOW[5]); k += 1
        best = fig_groups(RESP, ET, SS, m, tt, f(f"{m}_6_groups.png"))
        bb = sorted(((v, ph, e) for (ph, e), v in best.items()), reverse=True)[:2]
        FD.add_slide(prs, sec, f"{m}: average response of UP, DOWN and other neurons", f(f"{m}_6_groups.png"), NOTE[6],
                     ("Largest mean responses of the UP neurons: " + ", ".join(f"{e.lower()} ({PH[ph]}, {v:+.2f} z)" for v, ph, e in bb)
                      if bb else "No UP neurons in this mouse.") + ".", pv(), howto=HOW[6]); k += 1
        res = fig_population(RESP, SS, m, tt, f(f"{m}_7_population.png"))
        chg = [(e, r[2]) for e, r in res.items() if np.isfinite(r[2]) and r[2] < 0.05]
        rb = [(e, r[3]) for e, r in res.items() if np.isfinite(r[0]) and r[0] < 0.05 and r[3] > 0]
        FD.add_slide(prs, sec, f"{m}: population response, baseline vs injection", f(f"{m}_7_population.png"), NOTE[7],
                     ("All neurons together rose at baseline after: " + ", ".join(e.lower() for e, _ in rb) + ". " if rb else
                      "No event raised the whole population at baseline (p < 0.05). ")
                     + ("Baseline vs injection differed for: " + ", ".join(f"{e.lower()} ({pstr(p)})" for e, p in chg) + "."
                        if chg else "No event's population response differed between the sessions (p < 0.05)."), pv(), howto=HOW[7]); k += 1
        sig = fig_counts(ET, R7, SS, m, f(f"{m}_8_counts.png"))
        sb = [e.lower() for e in EV7 if sig.get(("pre", e), 1) <= 0.05]
        si = [e.lower() for e in EV7 if sig.get(("post", e), 1) <= 0.05]
        FD.add_slide(prs, sec, f"{m}: how many neurons go up or down, per event", f(f"{m}_8_counts.png"), NOTE[8],
                     "Population response beyond chance (q <= 0.05) at baseline: " + (", ".join(sb) if sb else "none")
                     + "; after the injection: " + (", ".join(si) if si else "none") + ".", pv(), howto=HOW[8]); k += 1
        nn, multi, best = fig_matrix(TA, m, f(f"{m}_9_matrix.png"))
        FD.add_slide(prs, sec, f"{m}: which neuron responds to which stimulus or behaviour", f(f"{m}_9_matrix.png"), NOTE[9],
                     f"At baseline {multi} of {nn} neurons rose after two or more different events. Most UP neurons at baseline: "
                     + ", ".join(f"{e} {c}" for c, e in best) + ".", pv(), howto=HOW[9]); k += 1
        rc = fig_resp_change(SP, m, f(f"{m}_10_responsive_change.png"))
        sigc = [f"{e and SHORTM[e]} ({g}, {b0:+.2f} -> {b1:+.2f} z)" for (g, e), (n_, b0, b1, p) in rc.items() if np.isfinite(p) and p < 0.05]
        FD.add_slide(prs, sec, f"{m}: responsive neurons, baseline vs after injection", f(f"{m}_10_responsive_change.png"), NOTE[10],
                     ("Responsive neurons changed after the injection for: " + "; ".join(sigc) + "." if sigc else
                      "No group of responsive neurons changed significantly after the injection (paired Wilcoxon p < 0.05).")
                     , pv(), howto=HOW[10]); k += 1
        print(f"{m}: 8 slides")
    return k - 1


if __name__ == "__main__":
    main()
