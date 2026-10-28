"""ca_space_stability.py - where the responsive neurons sit in the field of view, and whether a neuron keeps its response
across days, conditions and events (2026-10-23; PI follow-up of the slides 'Are the responsive neurons clustered in the
field of view?' and 'Stable across days, shared between events').

What and why
 1 CLUSTER_maps.png (+ _p2)  the slide on clustering showed only z values. Here every mouse x day x event where a group of
                             responsive neurons (UP+DOWN pooled, UP or DOWN) was clustered in P4_topography.csv (p < 0.05)
                             is drawn: the field of view of that session (max dF/F projection of the day, footprint outlines
                             at 30 % of peak; UP red, DOWN blue, others grey), the clustered group filled and its convex hull
                             dashed, next to the other session of the same day (the same neurons), sorted by p.
   CLUSTER_summary.png       number of p < 0.05 maps per condition / group / mouse vs the 5 % expected by chance
                             (binomial, one-sided) - is the clustering more than chance?
 2 STAB_classes.png          the 185 tracked neurons: % of the reference neurons (UP or DOWN in the first session) that are
                             the same class in the second session vs chance (neuron identities shuffled within mouse, 1000):
                             Day-1 baseline vs Day-2 baseline (no drug), after vehicle vs after SBI-553, and the class change
                             from the same-day baseline (rose / fell) on the vehicle day vs on the SBI-553 day.
   STAB_resp_corr.png        per event the response (resp) of each tracked neuron, Day-1 vs Day-2 baseline and after vehicle
                             vs after SBI-553 (Spearman rho, p from 1000 within-mouse shuffles; per-mouse colours).
   STAB_change_corr.png      the baseline-normalized change (injection - its own same-day baseline), vehicle day vs SBI-553 day,
                             and the rho of the three comparisons per event (user rule 10-23: baseline-normalized comparisons).
   STAB_overlap.png          event overlap (Jaccard %, responsive to both / to either) at baseline, after vehicle and after
                             SBI-553 (every neuron of each session) and SBI-553 - vehicle (Fisher exact per cell, exploratory).
 3 MAPS_<mouse>.png          supplement: 4 sessions (vehicle-day baseline, after vehicle, SBI-553-day baseline, after
                             SBI-553) x 7 events; responsive neurons coloured; tracked neurons filled, the others hollow.
Every figure in English (<name>.png) and Korean (<name>_kr.png); summary.json = the numbers and slide texts.

Inputs  analysis\\event_analysis\\cells_classes.csv (classes, resp), P4_topography.csv (clustering z, p; re-checked here:
        recomputed from cells_classes.csv with the same code and seed it matches exactly), analysis\\crossday\\<m>_pairs.csv
        (tracked neurons), E.centroids() (neuron positions, 4_cell_review\\*_cell_review.csv), 3_extract\\*_extract.mat
        (footprints) and *_projections.mat (max dF/F) of mini1p_out\\<date>\\<m>_d<d>.
Outputs analysis\\pi_extra\\space\\ (PNGs, stab_classes.csv, stab_resp_corr.csv, stab_overlap.csv, cluster_tests.csv,
        summary.json).
Run     python -I -X utf8 C:\\Users\\hsollim\\Documents\\HEAL_mini1p_SBI553\\mini1p_pipeline\\ca_space_stability.py
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial import ConvexHull
from scipy.stats import binomtest, fisher_exact, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401
import pairstars as PS  # noqa: E402  (every star shows the pair it compares)
import figfoot  # noqa: E402  (10-24: methods text -> slide notes, <png>.notes.txt)
import ca_event_analysis as E  # noqa: E402  (E.centroids(): neuron positions)

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
EA = os.path.join(B, "analysis", "event_analysis")
CD = os.path.join(B, "analysis", "crossday")
OUT = os.path.join(B, "analysis", "pi_extra", "space")
ROOT = {1: os.path.join(B, "mini1p_out", "20261019"), 2: os.path.join(B, "mini1p_out", "20261021")}
MO = ["5941", "5939", "5937", "1191"]
SEX = {"5941": "F", "1191": "F", "5939": "M", "5937": "M"}
SBI_DAY = {"5941": 1, "5939": 1, "5937": 2, "1191": 2}
CC = {"baseline": "#7f8c8d", "vehicle": "#2471a3", "SBI-553": "#c0392b"}
MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#34495e"}
MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
UPC, DNC, NOC = "#e74c3c", "#2e86c1", "#c8cdd2"
CATC = {"activated": UPC, "inhibited": DNC, "non-responsive": NOC}
OBS, CHN = "#2ecc40", "0.78"                                 # observed (green) / chance (grey), as on the old slide
EV7 = ["Light touch", "Mild touch", "Pin prick", "Heat", "withdrawal", "flinch", "movement onset"]
SH = {"Light touch": "light", "Mild touch": "mild", "Pin prick": "pin", "Heat": "heat", "withdrawal": "withdrawal",
      "flinch": "flinch", "movement onset": "movement", "escape/rearing onset": "escape", "grooming onset": "grooming",
      "attending onset": "attending"}
GRP = {"responsive": "UP+DOWN", "activated": "UP", "inhibited": "DOWN"}
GCOL = {"responsive": "#f1c40f", "activated": UPC, "inhibited": DNC}
GTXT = {"responsive": "#9a7d0a", "activated": "#c0392b", "inhibited": "#1f618d"}
V = {"activated": 1, "inhibited": -1, "non-responsive": 0}
NSH = 1000
SEED = 20261023
LG = "en"


def tr(en, kr):
    return kr if LG == "kr" else en


def font(lg):
    global LG
    LG = lg
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": ["Malgun Gothic", "DejaVu Sans"] if lg == "kr" else ["DejaVu Sans"],
                         "axes.unicode_minus": False})


def fn(name):
    return os.path.join(OUT, name + ("_kr" if LG == "kr" else "") + ".png")


def foot(fig, txt, fs=11, y=0.006):
    """10-24 (user: details in the notes, not on the slide): the methods text goes to <png>.notes.txt (figfoot -> slide
    notes); the figure keeps only the one-line star key when the text has one. Returns the number of lines drawn (1)."""
    figfoot.note(fig, txt, LG, fs=11.5)
    return 1


def ptxt(p, shuffle=False):
    if p is None or not np.isfinite(p):
        return "n.a."
    if shuffle and p <= 1.0 / (NSH + 1) + 1e-12:
        return f"p < {1.0 / (NSH + 1) + 5e-6:.3f}"                 # no shuffle reached the observed value
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def ptxt_fig(p, shuffle=False):
    """ptxt for figure titles: a p just under 0.05 (e.g. 0.046) is printed with 3 decimals, not as 'p = 0.05' under a black
    (significant) title"""
    t = ptxt(p, shuffle)
    if p is not None and np.isfinite(p) and p < 0.05 and t == "p = 0.05":
        t = f"p = {p:.3f}"
    return t


def cond_of(m, d, ph):
    return "baseline" if ph == "pre" else ("SBI-553" if SBI_DAY[m] == d else "vehicle")


def sess_lab(m, d, ph):
    c = cond_of(m, d, ph)
    if ph == "pre":
        return tr("baseline", "기준")
    return tr(f"after {c}", f"{c} 후")


def drug_day(m, d):
    return "SBI-553" if SBI_DAY[m] == d else "vehicle"


# ------------------------------------------------------------------------------------------------ data
def load():
    T = pd.read_csv(os.path.join(EA, "cells_classes.csv"))
    T["mouse"] = T.mouse.astype(str)
    Tp = pd.read_csv(os.path.join(EA, "P4_topography.csv"))
    Tp["mouse"] = Tp.mouse.astype(str)
    P = {m: pd.read_csv(os.path.join(CD, f"{m}_pairs.csv")) for m in MO}
    kept = T.groupby(["mouse", "day"]).cell.unique()
    cen = {(str(m), int(d)): v.loc[v.index.isin(kept[(str(m), int(d))])]      # the analysed (kept) neurons only
           for (m, d), v in E.centroids().items()}
    R = T.set_index(["mouse", "day", "phase", "event", "cell"])
    rows = []
    for m in MO:
        sd, vd = SBI_DAY[m], 3 - SBI_DAY[m]
        for p in P[m].itertuples():
            ids = {1: int(p.day1_id), 2: int(p.day2_id)}
            for ev in EV7:
                r = dict(mouse=m, track_class=p.track_class, event=ev, id1=ids[1], id2=ids[2])
                for c, (d, ph) in {"b1": (1, "pre"), "b2": (2, "pre"), "vb": (vd, "pre"), "vi": (vd, "post"),
                                   "sb": (sd, "pre"), "si": (sd, "post")}.items():
                    k = (m, d, ph, ev, ids[d])
                    ok = k in R.index
                    r[c] = float(R.resp.get(k, np.nan)) if ok else np.nan
                    r["c_" + c] = R.cls.get(k, "") if ok else ""
                rows.append(r)
    X = pd.DataFrame(rows)
    X["chg_v"] = X.vi - X.vb
    X["chg_s"] = X.si - X.sb
    return T, Tp, P, cen, X


def check_topography(T, Tp, cen):
    """recompute P4_topography with the code and seed of ca_event_analysis.plot4_topography; max |z difference|"""
    rng = np.random.default_rng(3)
    rows = []

    def nnd(P_):
        D = np.hypot(P_[:, None, 0] - P_[None, :, 0], P_[:, None, 1] - P_[None, :, 1])
        np.fill_diagonal(D, np.inf)
        return float(np.mean(D.min(1)))
    for (m, d, ph, ev), Xg in T.groupby(["mouse", "day", "phase", "event"]):
        C_ = cen[(m, int(d))].reindex(Xg.cell.values).to_numpy(float)
        okc = np.isfinite(C_).all(1)
        for grp, sel in (("responsive", Xg.cls.isin(["activated", "inhibited"]).values),
                         ("activated", (Xg.cls == "activated").values), ("inhibited", (Xg.cls == "inhibited").values)):
            r = sel & okc
            if r.sum() < 4 or r.sum() > okc.sum() - 3:
                continue
            obs = nnd(C_[r])
            null = np.array([nnd(C_[okc][rng.choice(okc.sum(), r.sum(), replace=False)]) for _ in range(500)])
            rows.append(dict(mouse=m, day=int(d), phase=ph, event=ev, group=grp, z_new=(null.mean() - obs) / max(null.std(), 1e-9)))
    N = pd.DataFrame(rows)
    M = Tp.merge(N, on=["mouse", "day", "phase", "event", "group"])
    return float(np.abs(M.z - M.z_new).max()), len(M), len(Tp)


_FOV = {}


def fov(m, d):
    """max dF/F projection (display-scaled) and the outlines (30 % of peak) of every kept neuron of one mouse-day"""
    if (m, d) in _FOV:
        return _FOV[(m, d)]
    tag = f"{m}_d{d}"
    r = os.path.join(ROOT[d], tag)
    S = np.asarray(sio.loadmat(os.path.join(r, "3_extract", f"{tag}_extract.mat"), variable_names=["spatial_weights"])
                   ["spatial_weights"], np.float32)
    Rv = pd.read_csv(os.path.join(r, "4_cell_review", f"{tag}_cell_review.csv"))
    bg = np.asarray(sio.loadmat(os.path.join(r, "3_extract", f"{tag}_projections.mat"), variable_names=["max_dff"])["max_dff"], float)
    lo, hi = np.percentile(bg, [1, 99.8])
    img = np.clip((bg - lo) / (hi - lo + 1e-9), 0, 1) ** 0.7
    C = {}
    for c in Rv.id[Rv.keep == 1].astype(int):
        f = S[:, :, c - 1]
        if f.max() <= 0:
            continue
        cs, _ = cv2.findContours((f >= 0.3 * f.max()).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        C[c] = [cc.reshape(-1, 2) for cc in cs if len(cc) > 2]
    del S
    _FOV[(m, d)] = (img, C)
    return _FOV[(m, d)]


def cls_of(T, m, d, ph, ev):
    X = T[(T.mouse == m) & (T.day == d) & (T.phase == ph) & (T.event == ev)]
    return X.set_index("cell").cls if len(X) else None


# ------------------------------------------------------------------------------------------------ 1 clustering
def cluster_combos(Tp):
    S = Tp[Tp.p < 0.05]
    out = []
    for (m, d, ev), g in S.groupby(["mouse", "day", "event"]):
        out.append(dict(mouse=m, day=int(d), event=ev, pmin=float(g.p.min()), groups=sorted(set(g.group), key=list(GRP).index),
                        sig=[(r.phase, r.group) for r in g.itertuples()]))
    return sorted(out, key=lambda r: (r["pmin"], r["mouse"], r["day"]))


def draw_fov(a, T, Tp, cen, m, d, ph, ev, groups, lim):
    """one field-of-view map; returns (clustered in this session?, the lines for the text column: (text, colour, bold))"""
    img, C = fov(m, d)
    a.imshow(img, cmap="gray", vmin=0, vmax=1, alpha=0.92)
    a.set_xlim(lim[0], lim[1])
    a.set_ylim(lim[3], lim[2])
    a.set_xticks([])
    a.set_yticks([])
    cl = cls_of(T, m, d, ph, ev)
    pos = cen[(m, d)]
    lines = []
    hit = False
    if cl is None:
        for c, cc_ in C.items():
            for cc in cc_:
                a.plot(cc[:, 0], cc[:, 1], color="#c8c8c8", lw=0.5)
        a.text(0.5, 0.5, tr(f"no {SH[ev]}\nafter {cond_of(m, d, ph)}\n(< 5 events)", f"{cond_of(m, d, ph)} 후\n{SH[ev]} 없음\n(< 5회)"),
               transform=a.transAxes, ha="center", va="center", fontsize=12, color="w",
               bbox=dict(boxstyle="round,pad=0.3", fc="0.15", ec="none", alpha=0.85))
        lines.append((tr("not analysed", "분석 없음"), "0.3", False))
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_color("0.75")
            sp.set_linewidth(0.8)
        return hit, lines
    members = {}
    for g in groups:
        sel = cl[cl.isin(["activated", "inhibited"])] if g == "responsive" else cl[cl == g]
        row = Tp[(Tp.mouse == m) & (Tp.day == d) & (Tp.phase == ph) & (Tp.event == ev) & (Tp.group == g)]
        if lines:
            lines.append(("", "k", False))
        if len(row):
            r = row.iloc[0]
            sig = r.p < 0.05
            hit |= bool(sig)
            lines += [(GRP[g], GTXT[g], True), (f"n = {int(r.n)}", "0.1", False),
                      (tr(f"clustering z = {r.z:.2f}", f"군집 z = {r.z:.2f}"), "0.1", False),
                      (f"p = {r.p:.3f}", "0.1", bool(sig))]
            if sig:
                members[g] = sel.index.astype(int).tolist()
        else:
            lines += [(GRP[g], GTXT[g], True), (f"n = {len(sel)}", "0.1", False),
                      (tr("not tested (< 4)", "검정 안 함 (< 4)"), "0.3", False)]
    filled = set(c for v in members.values() for c in v)
    for c, k in cl.items():
        col = {"activated": UPC, "inhibited": DNC}.get(k, "#c8c8c8")
        for cc in C.get(int(c), []):
            if int(c) in filled:
                a.fill(cc[:, 0], cc[:, 1], color=col, alpha=0.75, lw=0)
            a.plot(cc[:, 0], cc[:, 1], color=col, lw=1.3 if k in ("activated", "inhibited") else 0.5)
    for g, ids in members.items():
        P_ = pos.reindex(ids)[["col", "row"]].dropna().to_numpy(float)
        if len(P_) >= 3:
            try:
                h = ConvexHull(P_)
                v = np.r_[h.vertices, h.vertices[:1]]
                a.plot(P_[v, 0], P_[v, 1], ls=(0, (4, 2.5)), lw=1.8, color=GCOL[g],
                       path_effects=[pe.Stroke(linewidth=3.4, foreground="k"), pe.Normal()])
            except Exception:                                 # collinear points: no hull
                pass
    for sp in a.spines.values():
        sp.set_visible(True)
        sp.set_color("#f39c12" if hit else "0.75")
        sp.set_linewidth(3.2 if hit else 0.8)
    return hit, lines


def scalebar(a, lim):
    x1 = lim[1] - 40
    y = lim[3] - 28
    a.plot([x1 - 100, x1], [y, y], color="w", lw=2.5, solid_capstyle="butt")
    a.text(x1 - 50, y - 12, "100 px", color="w", fontsize=11, ha="center", va="bottom",
           path_effects=[pe.Stroke(linewidth=2, foreground="0.1"), pe.Normal()])


def fig_cluster_maps(T, Tp, cen, combos):
    """4 mouse x day x event pairs per page (2 x 2); each pair = baseline map | injection map, text column right of each map"""
    pages = [combos[i:i + 4] for i in range(0, len(combos), 4)]
    files = []
    nsig = int((Tp.p < 0.05).sum())
    for pg, cmb in enumerate(pages):
        fig = plt.figure(figsize=(15, 6.3))
        W, H = fig.get_figwidth(), fig.get_figheight()
        mh = 2.14                                            # map height (in; the footer went to the notes, 10-24)
        hdr = 0.47                                           # 2 header lines above each map (in)
        tops = [H - 0.42 - hdr, H - 0.42 - hdr - mh - 0.12 - hdr]    # top edge of the maps of row 1 / row 2 (in)
        for k, c in enumerate(cmb):
            row, col = divmod(k, 2)
            m, d, ev = c["mouse"], c["day"], c["event"]
            Cc = cen[(m, d)]
            ih, iw = fov(m, d)[0].shape
            lim = (max(Cc.col.min() - 40, -0.5), min(Cc.col.max() + 40, iw - 0.5), max(Cc.row.min() - 40, -0.5),
                   min(Cc.row.max() + 40, ih - 0.5))
            mw = min(mh * (lim[1] - lim[0]) / (lim[3] - lim[2]), 2.0)      # <= 2.0 in: room for 'clustering z = ...'
            mhh = mw * (lim[3] - lim[2]) / (lim[1] - lim[0])
            for j, ph in enumerate(("pre", "post")):
                x0 = 0.08 + (2 * col + j) * 3.7 + (0.1 if col == 1 else 0)
                a = fig.add_axes([x0 / W, (tops[row] - mhh) / H, mw / W, mhh / H])
                hit, lines = draw_fov(a, T, Tp, cen, m, d, ph, ev, c["groups"], lim)
                scalebar(a, lim)
                cnd = cond_of(m, d, ph)
                a.annotate(f"{m} ({SEX[m]}) · Day {d} · {SH[ev]}", xy=(0, 1), xycoords="axes fraction", xytext=(0, 19),
                           textcoords="offset points", ha="left", va="bottom", fontsize=12, fontweight="bold", color="0.1")
                a.annotate(sess_lab(m, d, ph) + (tr("  (clustered)", "  (군집)") if hit else ""), xy=(0, 1),
                           xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", ha="left", va="bottom",
                           fontsize=12, fontweight="bold", color=CC[cnd])
                for i, (t_, c_, b_) in enumerate(lines):
                    a.annotate(t_, xy=(1, 1), xycoords="axes fraction", xytext=(7, -1 - 14.8 * i), textcoords="offset points",
                               ha="left", va="top", fontsize=11, color=c_, fontweight="bold" if b_ else "normal")
        if len(cmb) < 4:                                     # the free slot: every test with p < 0.05 as a table
            row, col = divmod(len(cmb), 2)
            x0 = 0.08 + 2 * col * 3.7 + (0.1 if col == 1 else 0) + 0.25
            y = tops[row] + hdr + 0.1
            S_ = Tp[Tp.p < 0.05].sort_values("p")
            nrow = len(S_)
            rh, fst = 0.2, 12                                    # row height (in), font (pt): >= 9.9 pt on the slide
            fit = int((y - 0.27 - 0.62 - 0.17) // rh)            # rows that fit above the legend
            show = nrow if nrow <= fit else fit - 1              # too many: the first rows + a pointer to the notes
            cx = [0, 0.82, 2.62, 3.82, 5.24, 6.12]
            fig.text(x0 / W, y / H, (tr(f"All {nrow} tests with p < 0.05 (of {len(Tp)})",
                                        f"p < 0.05인 검정 전체 {nrow}개 ({len(Tp)}개 중)") if show == nrow else
                                     tr(f"The {show} lowest p of {nrow} tests with p < 0.05 (of {len(Tp)})",
                                        f"p < 0.05인 검정 {nrow}개 중 p가 가장 낮은 {show}개 ({len(Tp)}개 중)")),
                     fontsize=13, fontweight="bold", va="top")
            y -= 0.27
            for k_, h_ in enumerate(tr(["mouse", "session", "event", "group (n)", "clust. z", "p"],
                                       ["마우스", "세션", "사건", "그룹 (n)", "군집 z", "p"])):
                fig.text((x0 + cx[k_]) / W, y / H, h_, fontsize=fst, fontweight="bold", va="top", color="0.25")
            for r in list(S_.itertuples())[:show]:
                y -= rh
                ses = f"D{r.day} " + sess_lab(r.mouse, r.day, r.phase)
                for k_, v_ in enumerate([r.mouse, ses, SH[r.event], f"{GRP[r.group]} ({r.n})", f"{r.z:.2f}", f"{r.p:.3f}"]):
                    fig.text((x0 + cx[k_]) / W, y / H, v_, fontsize=fst, va="top",
                             color=CC[cond_of(r.mouse, r.day, r.phase)] if k_ == 1 else (GTXT[r.group] if k_ == 3 else "0.1"))
            if show < nrow:
                fig.text(x0 / W, (y - rh) / H, tr(f"... {nrow - show} more (all tests are listed in the notes)",
                                                  f"... {nrow - show}개 더 (모든 검정은 노트에)"),
                         fontsize=fst, va="top", color="0.35", style="italic")
        fig.add_artist(Line2D([0.5, 0.5], [0.11, 0.9], transform=fig.transFigure, color="0.8", lw=1))
        ttl = tr("Where the clustered responsive neurons are: every session x event with p < 0.05, baseline | after the "
                 "injection (same neurons)",
                 "군집된 반응 뉴런의 위치: p < 0.05인 모든 세션 x 사건, 기준 | 주사 후 (같은 뉴런)")
        if len(pages) > 1:
            ttl += f"  ({pg + 1}/{len(pages)})"
        fig.text(0.008, 0.988, ttl, fontsize=14, fontweight="bold", va="top")
        # legend in 2 rows (column-major: UP/DOWN | non-resp./filled | hull/frame | z/scale bar)
        hd = [Line2D([], [], color=UPC, lw=2, label="UP"),
              Line2D([], [], color=DNC, lw=2, label="DOWN"),
              Line2D([], [], color="#b0b0b0", lw=1.2, label=tr("non-responsive", "무반응")),
              Patch(color="0.45", label=tr("filled = the clustered group", "채움 = 군집된 그룹")),
              Line2D([], [], color=GCOL["responsive"], lw=1.8, ls=(0, (4, 2.5)),
                     path_effects=[pe.Stroke(linewidth=3.4, foreground="k"), pe.Normal()],
                     label=tr("dashed = its convex hull (yellow UP+DOWN, red UP, blue DOWN)",
                              "점선 = 그 그룹의 볼록 껍질 (노랑 UP+DOWN, 빨강 UP, 파랑 DOWN)")),
              Patch(fc="none", ec="#f39c12", lw=2.5, label=tr("orange frame = clustered, p < 0.05",
                                                             "주황 테두리 = 군집, p < 0.05")),
              Line2D([], [], color="none", label=tr("clustering z > 0 = closer than random",
                                                    "군집 z > 0 = 무작위보다 가까움")),
              Line2D([], [], color="0.25", lw=2.5, solid_capstyle="butt",
                     label=tr("scale bar = 100 px (imaging pixels)", "축척 막대 = 100 px (영상 픽셀)"))]
        fig.legend(handles=hd, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=4, frameon=False, fontsize=11,
                   handlelength=1.8, columnspacing=1.4, labelspacing=0.35)
        foot(fig, tr(f"Clustering z = (mean nearest-neighbour distance of 500 random draws of the same number of that session's "
                     f"neurons - observed) / SD of the draws; p = share of draws <= observed (one-sided; P4_topography). "
                     f"Background = max dF/F projection of the day, outlines = footprints at 30 % of peak. Sorted by p. "
                     f"{nsig} of {len(Tp)} tests had p < 0.05 ({0.05 * len(Tp):.1f} expected by chance; CLUSTER_summary).",
                     f"군집 z = (그 세션 뉴런 중 같은 수를 무작위로 500번 뽑은 평균 최근접 이웃 거리 - 관측값) / 무작위 SD; p = 관측값 이하인 "
                     f"무작위 비율 (단측; P4_topography). 배경 = 그날의 max dF/F 투영, 윤곽 = footprint 30 % 선. p 순서. "
                     f"{len(Tp)}개 검정 중 {nsig}개 p < 0.05 (우연 기대 {0.05 * len(Tp):.1f}개; CLUSTER_summary)."), fs=10.5)
        name = "CLUSTER_maps" + ("" if pg == 0 else f"_p{pg + 1}")
        fig.savefig(fn(name), dpi=150)
        plt.close(fig)
        files.append(name)
    return files


def cluster_tests(Tp):
    """number of p < 0.05 maps vs 5 % expected (binomial, one-sided) per condition / group / mouse"""
    rows = []
    for kind, col, levels in (("condition", "condition", ["baseline", "vehicle", "SBI-553"]),
                              ("group", "group", ["responsive", "activated", "inhibited"]), ("mouse", "mouse", MO)):
        for lv in levels + ["all"]:
            Z = Tp if lv == "all" else Tp[Tp[col] == lv]
            n, k = len(Z), int((Z.p < 0.05).sum())
            p = binomtest(k, n, 0.05, alternative="greater").pvalue if n else np.nan
            rows.append(dict(kind=kind, level=lv, n_tests=n, n_sig=k, expected=0.05 * n, p_binom=p))
            if kind != "condition" and lv == "all":
                rows.pop()
    return pd.DataFrame(rows)


def fig_cluster_summary(Tp, CT):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6.3), gridspec_kw=dict(width_ratios=[4, 3, 4]))
    labs = {"condition": {"baseline": tr("baseline", "기준"), "vehicle": tr("after\nvehicle", "vehicle\n후"),
                          "SBI-553": tr("after\nSBI-553", "SBI-553\n후"), "all": tr("all", "전체")},
            "group": {"responsive": "UP+DOWN", "activated": "UP", "inhibited": "DOWN"},
            "mouse": {m: f"{m} ({SEX[m]})" for m in MO}}
    ttl = {"condition": tr("A  by condition", "A  조건별"), "group": tr("B  by group", "B  그룹별"),
           "mouse": tr("C  by mouse", "C  마우스별")}
    ymax = max(CT.n_sig.max(), CT.expected.max()) * 1.45 + 1
    for a, kind in zip(ax, ("condition", "group", "mouse")):
        Z = CT[CT.kind == kind].reset_index(drop=True)
        for i, r in Z.iterrows():
            colr = CC.get(r.level, MC.get(r.level, "#34495e" if r.level != "all" else "0.3"))
            if kind == "group":
                colr = {"responsive": "#b7950b", "activated": UPC, "inhibited": DNC}[r.level]
            a.bar(i - 0.2, r.n_sig, width=0.38, color=colr)
            a.bar(i + 0.2, r.expected, width=0.38, color=CHN)
            a.text(i - 0.2, r.n_sig + 0.15, f"{r.n_sig}/{r.n_tests}", ha="center", va="bottom", fontsize=11.5)
            a.text(i + 0.2, r.expected + 0.15, f"{r.expected:.1f}", ha="center", va="bottom", fontsize=11.5, color="0.35")
            PS.bracket(a, i - 0.2, i + 0.2, max(r.n_sig, r.expected) + 0.12 * ymax, r.p_binom, "k", fs=13, h=0.02 * ymax, inset=0)
        a.set_xticks(range(len(Z)), [labs[kind][lv] for lv in Z.level])
        a.set_ylim(0, ymax)
        a.set_xlim(-0.6, len(Z) - 0.4)
        a.set_title(ttl[kind], loc="left", fontsize=13, fontweight="bold")
        # the same y label on every panel (10-24 audit: B and C had none)
        a.set_ylabel(tr("maps (session x event x group)\nwith clustering p < 0.05", "군집 p < 0.05인 지도 수\n(세션 x 사건 x 그룹)"))
    ax[0].legend(handles=[Patch(color="0.45", label=tr("observed (coloured)", "관측 (색)")),
                          Patch(color=CHN, label=tr("expected by chance = 5 % of the tests", "우연 기대 = 검정의 5 %"))],
                 frameon=False, loc="upper left", fontsize=11)
    tot = CT[(CT.kind == "condition") & (CT.level == "all")].iloc[0]
    fig.text(0.008, 0.985, tr(f"Is the clustering more than chance? {tot.n_sig} of {tot.n_tests} maps p < 0.05 vs {tot.expected:.1f} "
                              f"expected (binomial {ptxt(tot.p_binom)})",
                              f"군집이 우연 이상인가? {tot.n_tests}개 지도 중 {tot.n_sig}개 p < 0.05, 기대 {tot.expected:.1f}개 "
                              f"(이항 {ptxt(tot.p_binom)})"), ha="left", va="top", fontsize=14, fontweight="bold")
    for a in ax:
        a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout(rect=(0, 0.05, 1, 0.93), w_pad=2.0)
    foot(fig, tr("Each map = one session x event x group (UP+DOWN pooled, UP, DOWN; groups with 4 neurons up to all but 3 are "
                 "tested), clustering test of P4_topography (500 random draws, one-sided). Bracket = observed vs expected count "
                 "(binomial test, one-sided, rate 0.05). The tests are not independent (UP+DOWN contains UP and DOWN; events share "
                 "neurons), so the binomial p is optimistic. " + PS.KEY,
                 "지도 하나 = 세션 x 사건 x 그룹 (UP+DOWN 합침, UP, DOWN; 뉴런 4개 이상, 전체-3개 이하인 그룹만 검정), "
                 "P4_topography 군집 검정 (무작위 500번, 단측). 괄호 = 관측 vs 기대 개수 (이항 검정, 단측, 비율 0.05). 검정들은 "
                 "독립이 아님 (UP+DOWN은 UP과 DOWN을 포함; 사건끼리 뉴런 공유) - 이항 p는 낙관적. "
                 "* p < 0.05, ** p < 0.01, *** p < 0.001, ns = 유의하지 않음"))
    fig.savefig(fn("CLUSTER_summary"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ 2 stability
def _perm_idx(mice, rng, n):
    """n within-mouse permutations of the row indices: array n x rows"""
    rows = len(mice)
    out = np.tile(np.arange(rows), (n, 1))
    for m in np.unique(mice):
        i = np.flatnonzero(mice == m)
        out[:, i] = i[np.argsort(rng.random((n, len(i))), 1)]
    return out


def stab_classes(X, tracking="A+B"):
    rng = np.random.default_rng(SEED)
    rows = []
    for kind in ("base", "inj", "chg"):
        for ev in EV7:
            Z = X[X.event == ev]
            if tracking == "A":
                Z = Z[Z.track_class == "A"]
            if kind == "base":
                ra, rb = Z.c_b1.map(V).values, Z.c_b2.map(V).values
            elif kind == "inj":
                ra, rb = Z.c_vi.map(V).values, Z.c_si.map(V).values
            else:
                ra = np.sign(Z.c_vi.map(V).values - Z.c_vb.map(V).values)
                rb = np.sign(Z.c_si.map(V).values - Z.c_sb.map(V).values)
            mice = Z.mouse.values
            ref = ra != 0
            same = (ra == rb) & ref
            obs = 100 * same.sum() / max(ref.sum(), 1)
            pi = _perm_idx(mice, rng, NSH)
            null = 100 * ((ra[None, :] == rb[pi]) & ref[None, :]).sum(1) / max(ref.sum(), 1)
            p = (1 + np.sum(null >= obs - 1e-9)) / (NSH + 1)
            r = dict(tracking=tracking, kind=kind, event=ev, n_tracked=len(Z), n_ref=int(ref.sum()), n_same=int(same.sum()),
                     pct=obs, chance=float(null.mean()), chance_lo=float(np.percentile(null, 2.5)),
                     chance_hi=float(np.percentile(null, 97.5)), p=p)
            for m in MO:
                k = mice == m
                r[f"nref_{m}"] = int(ref[k].sum())
                r[f"pct_{m}"] = 100 * same[k].sum() / ref[k].sum() if ref[k].sum() else np.nan
            # reverse direction (reference = the second session), for the notes
            refb = rb != 0
            r["pct_reverse"] = 100 * ((ra == rb) & refb).sum() / max(refb.sum(), 1)
            r["n_ref_reverse"] = int(refb.sum())
            rows.append(r)
    return pd.DataFrame(rows)


def fig_stab_classes(SC):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6.3), sharey=True)
    ttl = {"base": tr("A  Day-1 baseline -> Day-2 baseline\n(no drug; reference = UP / DOWN on Day 1)",
                      "A  Day-1 기준 -> Day-2 기준\n(약물 없음; 기준 = Day 1의 UP / DOWN)"),
           "inj": tr("B  after vehicle -> after SBI-553\n(reference = UP / DOWN after vehicle)",
                     "B  vehicle 후 -> SBI-553 후\n(기준 = vehicle 후 UP / DOWN)"),
           "chg": tr("C  change from the same-day baseline:\nvehicle day -> SBI-553 day (rose / fell)",
                     "C  같은 날 기준 대비 변화:\nvehicle 날 -> SBI-553 날 (상승 / 하강)")}
    for a, kind in zip(ax, ("base", "inj", "chg")):
        Z = SC[(SC.kind == kind) & (SC.tracking == "A+B")].reset_index(drop=True)
        for i, r in Z.iterrows():
            a.bar(i - 0.2, r.pct, width=0.38, color=OBS)
            a.bar(i + 0.2, r.chance, width=0.38, color=CHN)
            a.text(i, -6, f"{r.n_same}/{r.n_ref}", ha="center", va="center", fontsize=11, color="0.15")
            mx = r.pct
            for j, m in enumerate(MO):
                if r[f"nref_{m}"] >= 3:
                    a.scatter(i - 0.2 + (j - 1.5) * 0.075, r[f"pct_{m}"], marker=MK[m], s=30, color=MC[m], edgecolor="w",
                              lw=0.6, zorder=5, clip_on=False)
                    mx = max(mx, r[f"pct_{m}"])
            PS.bracket(a, i - 0.2, i + 0.2, max(mx, r.chance) + 5, r.p, "k", fs=13, h=1.3, inset=0)
        a.axhline(0, color="k", lw=0.8)
        a.spines["bottom"].set_visible(False)
        a.spines["left"].set_bounds(0, 100)
        a.tick_params(axis="x", length=0)
        a.set_xticks(range(len(Z)), [SH[e] for e in Z.event], rotation=35, ha="right")
        a.set_ylim(-11, 100)
        a.set_yticks([0, 20, 40, 60, 80, 100])
        a.set_xlim(-0.6, len(Z) - 0.4)
        a.set_title(ttl[kind], loc="left", fontsize=12.5)
        a.text(-0.5, -6, "k/n", ha="right", va="center", fontsize=11, color="0.15", style="italic")
    ax[0].set_ylabel(tr("% of the reference neurons\nwith the same class again", "기준 뉴런 중 다시\n같은 class인 비율 (%)"))
    ax[2].set_ylabel(tr("% with the same change again", "같은 변화가 다시 나온 비율 (%)"), fontsize=12)
    ax[2].yaxis.set_tick_params(labelleft=True)
    hd = [Patch(color=OBS, label=tr("observed (k / n below)", "관측 (아래 k / n)")),
          Patch(color=CHN, label=tr("chance (identity shuffled within mouse)", "우연 (마우스 안 뉴런 셔플)"))]
    hd += [Line2D([], [], color="none", label=tr("markers = one mouse:", "표식 = 마우스 하나:"))]
    hd += [Line2D([], [], marker=MK[m], color=MC[m], ls="none", ms=7, label=f"{m} ({SEX[m]})") for m in MO]
    fig.legend(handles=hd, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=7, frameon=False, fontsize=11.5,
               columnspacing=1.0, handlelength=1.5, handletextpad=0.4)
    n_tr = SC[(SC.tracking == "A+B")].n_tracked.iloc[0]
    fig.text(0.008, 0.985, tr(f"Do the tracked neurons keep their class? {n_tr} tracked neurons (same cell on both days), 4 mice",
                              f"추적된 뉴런은 class를 유지하는가? 추적 뉴런 {n_tr}개 (두 날 같은 세포), 마우스 4마리"),
             ha="left", va="top", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95), w_pad=1.2)
    foot(fig, tr("Class = UP (activated) / DOWN (inhibited) / non-responsive (time-shift test, BH q <= 0.20 per session). A, B: % of "
                 "the neurons UP or DOWN in the first session that have the same class in the second. C: per neuron the class "
                 "change from its own same-day baseline (rose: e.g. none -> UP, DOWN -> none; fell: the reverse); % of the neurons "
                 "that rose / fell on the vehicle day that did the same on the SBI-553 day. Bracket = observed vs chance (p = share "
                 "of 1000 shuffles >= observed, one-sided). Markers = one mouse (>= 3 reference neurons), descriptive. Drug order "
                 "= day (5941, 5939 SBI-553 on Day 1; 5937, 1191 on Day 2). " + PS.KEY,
                 "Class = UP (활성) / DOWN (억제) / 무반응 (time-shift 검정, 세션별 BH q <= 0.20). A, B: 첫 세션에서 UP 또는 DOWN인 "
                 "뉴런 중 둘째 세션에서도 같은 class인 비율. C: 뉴런마다 같은 날 기준 대비 class 변화 (상승: 예 무반응 -> UP, "
                 "DOWN -> 무반응; 하강: 반대); vehicle 날 상승 / 하강한 뉴런 중 SBI-553 날 같은 변화 비율. 괄호 = 관측 vs 우연 "
                 "(p = 셔플 1000번 중 관측 이상 비율, 단측). 표식 = 마우스 하나 (기준 뉴런 >= 3), 기술적. 약물 순서 = 날 (5941, 5939는 "
                 "Day 1에 SBI-553; 5937, 1191은 Day 2). * p < 0.05, ** p < 0.01, *** p < 0.001, ns = 유의하지 않음"), fs=10.5)
    fig.savefig(fn("STAB_classes"), dpi=150)
    plt.close(fig)


def corr_table(X):
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for kind, a_, b_ in (("base", "b1", "b2"), ("inj", "vi", "si"), ("chg", "chg_v", "chg_s")):
        for ev in EV7:
            Z = X[(X.event == ev)].dropna(subset=[a_, b_])
            x, y, mice = Z[a_].values, Z[b_].values, Z.mouse.values
            rho, p_sp = spearmanr(x, y)
            rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
            pi = _perm_idx(mice, rng, NSH)
            rys = ry[pi]
            nul = np.array([np.corrcoef(rx, r_)[0, 1] for r_ in rys])
            p_sh = (1 + np.sum(np.abs(nul) >= abs(rho) - 1e-12)) / (NSH + 1)
            r = dict(kind=kind, event=ev, n=len(Z), rho=float(rho), p_spearman=float(p_sp), p_shuffle=float(p_sh))
            for m in MO:
                k = mice == m
                r[f"rho_{m}"] = float(spearmanr(x[k], y[k])[0]) if k.sum() >= 5 else np.nan
                r[f"n_{m}"] = int(k.sum())
            rows.append(r)
    return pd.DataFrame(rows)


def _scatter(a, Z, xa, ya, lim):
    for m in MO:
        k = Z.mouse == m
        a.scatter(np.clip(Z[xa][k], -lim, lim), np.clip(Z[ya][k], -lim, lim), s=13, marker=MK[m], color=MC[m], alpha=0.75,
                  lw=0, zorder=3)
    a.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.8, ls="--", zorder=1)
    a.axhline(0, color="0.85", lw=0.7, zorder=0)
    a.axvline(0, color="0.85", lw=0.7, zorder=0)
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_aspect("equal")
    v = round(lim * 0.6, 1)
    t, tl = [-v, 0, v], [f"{-v:g}", "0", f"{v:g}"]           # 3 ticks, '0' short: no labels running together
    a.set_xticks(t, tl)
    a.set_yticks(t, tl)
    a.tick_params(labelsize=12.5, length=3, pad=2)


def _lim(v):
    v = np.abs(np.asarray(v, float))
    return float(np.ceil(np.nanpercentile(v, 99.0) * 1.1 * 10) / 10)


def fig_stab_resp_corr(X, CR):
    fig = plt.figure(figsize=(15, 6.3))
    gs = fig.add_gridspec(2, 7, left=0.085, right=0.995, top=0.85, bottom=0.165, hspace=0.6, wspace=0.42)
    rows = (("base", "b1", "b2", tr("A  baselines (no drug)", "A  기준끼리 (약물 없음)"),
             tr("Day-1 baseline response (z)", "Day-1 기준 반응 (z)"), tr("Day-2 baseline\nresponse (z)", "Day-2 기준\n반응 (z)")),
            ("inj", "vi", "si", tr("B  after the injections", "B  주사 후끼리"),
             tr("response after vehicle (z)", "vehicle 후 반응 (z)"), tr("after SBI-553\nresponse (z)", "SBI-553 후\n반응 (z)")))
    for i, (kind, xa, ya, rt, xl, yl) in enumerate(rows):
        for j, ev in enumerate(EV7):
            a = fig.add_subplot(gs[i, j])
            Z = X[X.event == ev].dropna(subset=[xa, ya])
            lim = _lim(np.r_[X[X.event == ev][["b1", "b2", "vi", "si"]].values.ravel()])
            _scatter(a, Z, xa, ya, lim)
            r = CR[(CR.kind == kind) & (CR.event == ev)].iloc[0]
            a.set_title(f"{SH[ev]}\nrho {r.rho:.2f}, {ptxt_fig(r.p_shuffle, True)}", fontsize=11.5,
                        fontweight="normal", color="k" if r.p_shuffle < 0.05 else "0.5")
            if j == 0:
                a.set_ylabel(yl, fontsize=12)
            if j == 3:
                a.set_xlabel(xl, fontsize=12)
            if j == 0:
                bb = a.get_position()
                fig.text(0.006, (bb.y0 + bb.y1) / 2, rt, rotation=90, fontsize=12.5, fontweight="bold", ha="left", va="center",
                         color="0.15")
    hd = [Line2D([], [], color="none", label=tr("one dot = one tracked neuron:", "점 하나 = 추적 뉴런 하나:"))]
    hd += [Line2D([], [], marker=MK[m], color=MC[m], ls="none", ms=7, label=f"{m} ({SEX[m]})") for m in MO]
    hd += [Line2D([], [], color="0.6", ls="--", label=tr("identity line (y = x)", "y = x 선")),
           Line2D([], [], color="none", label=tr("grey title = p >= 0.05", "회색 제목 = p >= 0.05"))]
    leg = fig.legend(handles=hd, loc="lower center", bbox_to_anchor=(0.54, 0.0), ncol=7, frameon=False, fontsize=11.5,
                     columnspacing=1.1, handlelength=1.5, handletextpad=0.4)
    leg.get_texts()[-1].set_color("0.5")
    n = int(CR.n.iloc[0])
    fig.text(0.008, 0.985, tr(f"Does a tracked neuron respond the same in two sessions? Response of each of the {n} tracked neurons",
                              f"추적 뉴런은 두 세션에서 같은 반응을 보이는가? 추적 뉴런 {n}개 각각의 반응"),
             ha="left", va="top", fontsize=14, fontweight="bold")
    foot(fig, tr("Response = mean z in the response window minus the baseline window (trial mean, best window of each event). One "
                 "dot = one tracked neuron. Spearman rho over the neurons of the 4 mice; p from 1000 shuffles of neuron identity "
                 "within mouse (two-sided); grey title = p >= 0.05. Dots beyond the axis are drawn at its edge.",
                 "반응 = 반응 창 평균 z - 기준 창 평균 z (시행 평균, 사건별 최적 창). 점 하나 = 추적 뉴런 하나. 4마리 뉴런 전체의 "
                 "Spearman rho; p = 마우스 안 뉴런 셔플 1000번 (양측); 회색 제목 = p >= 0.05. 축 밖의 점은 가장자리에 그림."), fs=10.5)
    fig.savefig(fn("STAB_resp_corr"), dpi=150)
    plt.close(fig)


def fig_stab_change_corr(X, CR):
    fig = plt.figure(figsize=(15, 6.3))
    gs = fig.add_gridspec(2, 4, left=0.075, right=0.66, top=0.85, bottom=0.145, hspace=0.6, wspace=0.45)
    for j, ev in enumerate(EV7):
        a = fig.add_subplot(gs[j // 4, j % 4])
        Z = X[X.event == ev].dropna(subset=["chg_v", "chg_s"])
        lim = _lim(np.r_[Z.chg_v.values, Z.chg_s.values])
        _scatter(a, Z, "chg_v", "chg_s", lim)
        r = CR[(CR.kind == "chg") & (CR.event == ev)].iloc[0]
        a.set_title(f"{SH[ev]}\nrho {r.rho:.2f}, {ptxt_fig(r.p_shuffle, True)}", fontsize=11.5,
                    color="k" if r.p_shuffle < 0.05 else "0.5")
        if j % 4 == 0:
            a.set_ylabel(tr("SBI-553 day:\nafter - baseline (z)", "SBI-553 날:\n후 - 기준 (z)"), fontsize=12)
    # one x label for the whole scatter grid (10-24 audit: it sat under the bottom-middle panel only)
    fig.text((0.075 + 0.66) / 2, 0.015, tr("vehicle day: after - baseline (z)   (x axis of all 7 panels)",
                                           "vehicle 날: 후 - 기준 (z)   (7개 패널 모두의 x축)"),
             ha="center", va="bottom", fontsize=12.5)
    la = fig.add_subplot(gs[1, 3])
    la.set_axis_off()
    hd = [Line2D([], [], marker=MK[m], color=MC[m], ls="none", ms=7, label=f"{m} ({SEX[m]})") for m in MO]
    hd += [Line2D([], [], color="0.6", ls="--", label=tr("y = x", "y = x 선")),
           Line2D([], [], color="none", label=tr("grey title = p >= 0.05", "회색 제목 = p >= 0.05"))]
    lg_ = la.legend(handles=hd, loc="center", frameon=False, fontsize=11.5,
                    title=tr("one dot = one\ntracked neuron", "점 하나 =\n추적 뉴런 하나"), title_fontsize=11.5)
    lg_.get_texts()[-1].set_color("0.5")
    b = fig.add_axes([0.778, 0.205, 0.212, 0.455])
    kinds = [("base", CC["baseline"], tr("Day-1 vs Day-2 baseline", "Day-1 vs Day-2 기준")),
             ("inj", "#7d3c98", tr("after vehicle vs after SBI-553", "vehicle 후 vs SBI-553 후")),
             ("chg", "0.15", tr("change: vehicle day vs SBI-553 day", "변화: vehicle 날 vs SBI-553 날"))]
    for k_, (kind, col, lab) in enumerate(kinds):
        Z = CR[CR.kind == kind].set_index("event").reindex(EV7)
        y = np.arange(len(EV7)) + (k_ - 1) * 0.27
        b.barh(y, Z.rho, height=0.26, color=col, label=lab, alpha=0.85)
        for m in MO:
            b.scatter(Z[f"rho_{m}"], y, marker=MK[m], s=44, color=MC[m], edgecolor="k", lw=0.6, zorder=4)
    b.axvline(0, color="0.5", lw=0.8)
    b.set_yticks(range(len(EV7)), [SH[e] for e in EV7])
    b.set_ylim(len(EV7) - 0.5, -0.5)
    b.set_xlabel(tr("Spearman rho (tracked neurons)", "Spearman rho (추적 뉴런)"))
    b.legend(frameon=False, fontsize=11.5, loc="lower left", bbox_to_anchor=(-0.42, 1.01), ncol=1,
             title=tr("bars = rho per event, 4 mice pooled:", "막대 = 사건별 rho, 4마리 합침:"), title_fontsize=11.5,
             alignment="left")
    # the right panel's own mouse key (its markers = one mouse's rho, not one neuron)
    hm = [Line2D([], [], marker=MK[m], color=MC[m], mec="k", mew=0.6, ls="none", ms=8, label=m) for m in MO]
    fig.legend(handles=hm, loc="lower center", bbox_to_anchor=(0.845, 0.0), ncol=4, frameon=False, fontsize=11.5,
               columnspacing=0.9, handletextpad=0.25, title=tr("markers = rho of one mouse", "표식 = 마우스 하나의 rho"),
               title_fontsize=11.5)
    n = int(CR.n.iloc[0])
    fig.text(0.008, 0.985, tr(f"Baseline-normalized: does the change after the injection repeat in the same neuron? ({n} tracked neurons)",
                              f"기준 정규화: 주사 후 변화가 같은 뉴런에서 반복되는가? (추적 뉴런 {n}개)"),
             ha="left", va="top", fontsize=14, fontweight="bold")
    foot(fig, tr("Change = response after the injection minus the response at the baseline of the same day, per tracked neuron (vehicle "
                 "day vs SBI-553 day). Spearman rho; p from 1000 shuffles of neuron identity within mouse (two-sided); grey title = "
                 "p >= 0.05. Right: rho of the three comparisons per event. Drug order = day (5941, 5939 SBI-553 on Day 1; 5937, "
                 "1191 on Day 2).",
                 "변화 = 주사 후 반응 - 같은 날 기준 반응, 추적 뉴런마다 (vehicle 날 vs SBI-553 날). Spearman rho; p = 마우스 안 "
                 "뉴런 셔플 1000번 (양측); 회색 제목 = p >= 0.05. 오른쪽: 사건별 세 비교의 rho. 약물 순서 = 날 (5941, 5939는 Day 1에 "
                 "SBI-553; 5937, 1191은 Day 2)."), fs=10.5)
    fig.savefig(fn("STAB_change_corr"), dpi=150)
    plt.close(fig)


def overlap(T, cond):
    X = T[(T.phase == "pre")] if cond == "baseline" else T[T.condition == cond]
    X = X[X.event.isin(EV7)]
    R = X.assign(r=X.cls.isin(["activated", "inhibited"])).pivot_table(index=["mouse", "day", "phase", "cell"], columns="event",
                                                                        values="r", aggfunc="first")
    K = len(EV7)
    J = np.full((K, K), np.nan)
    both = np.zeros((K, K), int)
    eith = np.zeros((K, K), int)
    for i, e1 in enumerate(EV7):
        for j, e2 in enumerate(EV7):
            ok = R[e1].notna() & R[e2].notna()
            a_, b_ = R[e1][ok].astype(bool), R[e2][ok].astype(bool)
            both[i, j], eith[i, j] = int((a_ & b_).sum()), int((a_ | b_).sum())
            J[i, j] = 100 * both[i, j] / eith[i, j] if eith[i, j] else np.nan
    return J, both, eith, len(R)


def fig_stab_overlap(T):
    res = {c: overlap(T, c) for c in ("baseline", "vehicle", "SBI-553")}
    K = len(EV7)
    D = res["SBI-553"][0] - res["vehicle"][0]
    Pf = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(K):
            if i != j:
                bs, es = res["SBI-553"][1][i, j], res["SBI-553"][2][i, j]
                bv, ev_ = res["vehicle"][1][i, j], res["vehicle"][2][i, j]
                if es and ev_:
                    Pf[i, j] = fisher_exact([[bs, es - bs], [bv, ev_ - bv]])[1]
    fig = plt.figure(figsize=(15, 6.3))
    W, H = 15, 6.3
    s = 2.72                                                    # heatmap size (in)
    x0s = [1.05, 3.87, 6.69]
    x4 = 11.32                                                  # the difference map: a wide gap after the first colour bar
    y0 = 2.05
    lab = [SH[e] if e != "movement onset" else "movement" for e in EV7]
    im = None
    ttl = {"baseline": tr("baseline", "기준"), "vehicle": tr("after vehicle", "vehicle 후"), "SBI-553": tr("after SBI-553", "SBI-553 후")}
    for k, c in enumerate(("baseline", "vehicle", "SBI-553")):
        a = fig.add_axes([x0s[k] / W, y0 / H, s / W, s / H])
        J = res[c][0].copy()
        np.fill_diagonal(J, np.nan)
        im = a.imshow(J, cmap="viridis", vmin=0, vmax=60)
        for i in range(K):
            for j in range(K):
                if i != j and np.isfinite(J[i, j]):
                    a.text(j, i, f"{J[i, j]:.0f}", ha="center", va="center", fontsize=11.5, color="w" if J[i, j] < 35 else "k")
        a.set_xticks(range(K), lab, rotation=50, ha="right", fontsize=11.5)
        a.set_yticks(range(K), lab if k == 0 else [""] * K, fontsize=11.5)
        a.set_facecolor("0.93")
        a.set_title(f"{ttl[c]}\n({res[c][3]} {tr('neurons', '뉴런')})", color=CC[c], fontsize=12.5, fontweight="bold")
        for sp in a.spines.values():
            sp.set_visible(False)
    cax = fig.add_axes([(x0s[2] + s + 0.12) / W, y0 / H, 0.13 / W, s / H])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 20, 40, 60], extend="max")          # > 60 % drawn as 60
    cb.ax.tick_params(labelsize=11.5)
    cb.set_label(tr("overlap (%)", "겹침 (%)"), fontsize=12)
    # what the overlap is, and the blank diagonal, as a key under the 3 overlap maps
    fig.legend(handles=[Line2D([], [], color="none", label=tr("overlap = neurons responsive to both events / to either (%)",
                                                              "겹침 = 두 사건 모두 반응 / 둘 중 하나 반응 뉴런 (%)")),
                        Patch(fc="0.93", ec="0.75", label=tr("grey diagonal = the same event (left blank)",
                                                             "회색 대각선 = 같은 사건 (비움)"))],
               loc="lower center", bbox_to_anchor=((x0s[0] + x0s[2] + s) / 2 / W, 0.005), ncol=1, frameon=False,
               fontsize=11.5, handlelength=1.4, labelspacing=0.3)
    a = fig.add_axes([x4 / W, y0 / H, s / W, s / H])
    Dm = D.copy()
    np.fill_diagonal(Dm, np.nan)
    vm = 30
    im2 = a.imshow(Dm, cmap="RdBu_r", vmin=-vm, vmax=vm)
    for i in range(K):
        for j in range(K):
            if i != j and np.isfinite(Dm[i, j]):
                dt = f"{Dm[i, j]:+.0f}"
                a.text(j, i, "0" if dt in ("-0", "+0") else dt, ha="center", va="center", fontsize=11.5,
                       color="k" if abs(Dm[i, j]) < 20 else "w")
                if np.isfinite(Pf[i, j]) and Pf[i, j] < 0.05:
                    a.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec="k", lw=1.8))
    a.set_xticks(range(K), lab, rotation=50, ha="right", fontsize=11.5)
    a.set_yticks(range(K), lab, fontsize=11.5)
    a.set_facecolor("0.93")
    for sp in a.spines.values():
        sp.set_visible(False)
    a.set_title(tr("after SBI-553 - after vehicle", "SBI-553 후 - vehicle 후"), fontsize=12.5, fontweight="bold")
    cax2 = fig.add_axes([(x4 + s + 0.12) / W, y0 / H, 0.13 / W, s / H])
    cb2 = fig.colorbar(im2, cax=cax2, ticks=[-30, -15, 0, 15, 30], extend="both")
    cb2.ax.tick_params(labelsize=11.5)
    cb2.set_label(tr("difference (percentage points)", "차이 (%p)"), fontsize=12)
    fig.legend(handles=[Patch(fc="none", ec="k", lw=1.8, label=tr("black box = p < 0.05 (Fisher exact)",
                                                                   "검은 상자 = p < 0.05 (Fisher exact)"))],
               loc="lower center", bbox_to_anchor=((x4 + s / 2) / W, 0.005), frameon=False, fontsize=11.5, handlelength=1.4)
    fig.text(0.008, 0.985, tr("Do the same neurons respond to several events? Event overlap at baseline, after vehicle and "
                              "after SBI-553\n(every neuron of each session)",
                              "같은 뉴런이 여러 사건에 반응하는가? 기준, vehicle 후, SBI-553 후의 사건 간 겹침\n(각 세션의 모든 뉴런)"),
             fontsize=14, fontweight="bold", va="top")
    nbox = int(np.sum(np.isfinite(Pf) & (Pf < 0.05)) // 2)
    foot(fig, tr(f"Overlap (Jaccard) = neurons responsive (UP or DOWN) to both events / to either, %, neuron-sessions pooled over the "
                 f"4 mice (baseline = the 8 baseline sessions, both days). Diagonal left blank. Right: after SBI-553 minus after "
                 f"vehicle; black box = Fisher exact p < 0.05 (both vs either-only, SBI-553 vs vehicle neurons; different neurons "
                 f"and days, exploratory, no correction): {nbox} of {K * (K - 1) // 2} pairs.",
                 f"겹침 (Jaccard) = 두 사건 모두에 반응 (UP 또는 DOWN)한 뉴런 / 둘 중 하나에 반응한 뉴런, %, 4마리 뉴런-세션 합침 "
                 f"(기준 = 두 날의 기준 세션 8개). 대각선은 비움. 오른쪽: SBI-553 후 - vehicle 후; 검은 상자 = Fisher exact p < 0.05 "
                 f"(둘 다 vs 하나만, SBI-553 vs vehicle 뉴런; 다른 뉴런과 날, 탐색적, 보정 없음): {K * (K - 1) // 2}쌍 중 {nbox}쌍."),
         fs=10.5)
    fig.savefig(fn("STAB_overlap"), dpi=150)
    plt.close(fig)
    rows = []
    for c in ("baseline", "vehicle", "SBI-553"):
        for i, e1 in enumerate(EV7):
            for j, e2 in enumerate(EV7):
                if i < j:
                    rows.append(dict(condition=c, event1=e1, event2=e2, both=int(res[c][1][i, j]), either=int(res[c][2][i, j]),
                                     jaccard=float(res[c][0][i, j]), p_fisher_sbi_vs_veh=float(Pf[i, j]) if c == "SBI-553" else np.nan))
    return pd.DataFrame(rows), {c: res[c][3] for c in res}


# ------------------------------------------------------------------------------------------------ 3 supplement maps
def fig_maps(T, cen, P, m):
    sd, vd = SBI_DAY[m], 3 - SBI_DAY[m]
    keys = [(vd, "pre"), (vd, "post"), (sd, "pre"), (sd, "post")]
    trk = {1: set(P[m].day1_id.astype(int)), 2: set(P[m].day2_id.astype(int))}
    allc = pd.concat([cen[(m, 1)], cen[(m, 2)]])
    lim = (allc.col.min() - 35, allc.col.max() + 35, allc.row.min() - 35, allc.row.max() + 35)
    fig = plt.figure(figsize=(15, 6.3))
    W, H = 15, 6.3
    mh = 1.2                                                    # (the footer went to the notes, 10-24)
    asp = (lim[1] - lim[0]) / (lim[3] - lim[2])
    mw = min(mh * asp, 1.4)                                     # wide fields: keep room for the counts
    mhe = mw / asp
    gapx = (W - 2.45 - 7 * mw) / 7                              # room right of each map for the UP / DOWN counts
    counts = {}
    for i, (d, ph) in enumerate(keys):
        C_ = cen[(m, d)]
        y = H - 0.75 - (i + 1) * mh - i * 0.1
        cnd = cond_of(m, d, ph)
        lab = (tr(f"{drug_day(m, d)} day\n(Day {d})\nbaseline", f"{drug_day(m, d)} 날\n(Day {d})\n기준") if ph == "pre"
               else tr(f"after {cnd}\n(Day {d})", f"{cnd} 후\n(Day {d})"))
        fig.text(2.2 / W, (y + mh / 2) / H, lab, ha="right", va="center", fontsize=11.5, fontweight="bold", color=CC[cnd],
                 linespacing=1.05)
        for k, ev in enumerate(EV7):
            a = fig.add_axes([(2.45 + k * (mw + gapx)) / W, (y + (mh - mhe) / 2) / H, mw / W, mhe / H])
            cl = cls_of(T, m, d, ph, ev)
            P_ = C_.reindex(cl.index)
            nu = nd = 0
            for g in ("non-responsive", "inhibited", "activated"):
                sel = (cl == g).values
                tk = np.array([int(c) in trk[d] for c in cl.index])
                sz = 9 if g == "non-responsive" else 20
                a.scatter(P_.col.values[sel & tk], P_.row.values[sel & tk], s=sz, color=CATC[g], lw=0.3,
                          edgecolor="k" if g != "non-responsive" else "none", zorder=3 if g != "non-responsive" else 2)
                a.scatter(P_.col.values[sel & ~tk], P_.row.values[sel & ~tk], s=sz, facecolor="w", edgecolor=CATC[g],
                          lw=1.1 if g != "non-responsive" else 0.8, zorder=3 if g != "non-responsive" else 2)
                if g == "activated":
                    nu = int(sel.sum())
                if g == "inhibited":
                    nd = int(sel.sum())
            counts[(d, ph, ev)] = (nu, nd, len(cl))
            a.set_xlim(lim[0], lim[1])
            a.set_ylim(lim[3], lim[2])
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(True)
                sp.set_color("0.8")
            a.text(1.05, 0.62, str(nu), transform=a.transAxes, ha="left", va="center", fontsize=11.5, fontweight="bold",
                   color="#c0392b")
            a.text(1.05, 0.36, str(nd), transform=a.transAxes, ha="left", va="center", fontsize=11.5, fontweight="bold",
                   color="#1f618d")
            if i == 0:
                a.set_title(SH[ev], fontsize=12.5, fontweight="bold")
    n1, n2 = len(cen[(m, 1)]), len(cen[(m, 2)])
    ntr = len(P[m])
    fig.text(0.008, 0.985, tr(f"{m} ({SEX[m]}): where the responsive neurons are in the 4 sessions (each dot = one neuron in the "
                              f"field of view)", f"{m} ({SEX[m]}): 4개 세션에서 반응 뉴런의 위치 (점 하나 = 시야 속 뉴런 하나)"),
             fontsize=14, fontweight="bold", va="top")
    hd = [Line2D([], [], marker="o", color=UPC, mec="k", mew=0.3, ls="none", ms=6.5, label=tr("UP", "UP")),
          Line2D([], [], marker="o", color=DNC, mec="k", mew=0.3, ls="none", ms=6.5, label=tr("DOWN", "DOWN")),
          Line2D([], [], marker="o", color=NOC, ls="none", ms=4.5, label=tr("non-responsive", "무반응")),
          Line2D([], [], marker="o", color="0.35", ls="none", ms=6.5, label=tr(f"filled = tracked on both days ({ntr})",
                                                                                f"채움 = 두 날 추적 ({ntr})")),
          Line2D([], [], marker="o", mfc="w", mec="0.35", mew=1.1, ls="none", ms=6.5,
                 label=tr("hollow = one day only", "속 빈 점 = 한 날만"))]
    hd += [Line2D([], [], color="none", label=tr("numbers right of each map: UP (red) / DOWN (blue) neurons",
                                                 "지도 오른쪽 숫자: UP (빨강) / DOWN (파랑) 뉴런 수"))]
    fig.legend(handles=hd, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=6, frameon=False, fontsize=11.5,
               columnspacing=1.2, handletextpad=0.4)
    foot(fig, tr(f"Numbers right of each map = number of UP (red) / DOWN (blue) neurons. Neurons: Day 1 {n1}, Day 2 {n2} (the two sessions of a day = the same "
                 f"neurons). Positions in the coordinates of each day (the days differ by a few pixels; no warp). Classes: time-shift "
                 f"test, BH q <= 0.20 per session.",
                 f"각 지도 오른쪽 숫자 = UP (빨강) / DOWN (파랑) 뉴런 수. 뉴런: Day 1 {n1}, Day 2 {n2} (한 날의 두 세션 = 같은 뉴런). 위치는 각 날의 "
                 f"좌표 (두 날은 몇 픽셀 차이; 변형 없음). Class: time-shift 검정, 세션별 BH q <= 0.20."), fs=10.5)
    fig.savefig(fn(f"MAPS_{m}"), dpi=150)
    plt.close(fig)
    return counts


# ------------------------------------------------------------------------------------------------ main
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="figures to draw (cluster summary classes corr change overlap maps)")
    ap.add_argument("--lang", nargs="*", default=["en", "kr"])
    ap.add_argument("--no-check", action="store_true", help="skip the P4_topography recomputation")
    a = ap.parse_args()
    want = (lambda k: True) if not a.only else (lambda k: k in a.only)
    os.makedirs(OUT, exist_ok=True)
    T, Tp, P, cen, X = load()
    dz = np.nan
    if not a.no_check:
        dz, nm, nt = check_topography(T, Tp, cen)
        print(f"P4_topography recomputed: max |dz| = {dz:.2e} over {nm}/{nt} rows", flush=True)
    combos = cluster_combos(Tp)
    CT = cluster_tests(Tp)
    SC = pd.concat([stab_classes(X, "A+B"), stab_classes(X, "A")], ignore_index=True)
    CR = corr_table(X)
    cl_files, OV, nov, mcounts = [], None, None, {}
    for lg in a.lang:
        font(lg)
        if want("cluster"):
            cl_files = fig_cluster_maps(T, Tp, cen, combos)
        if want("summary"):
            fig_cluster_summary(Tp, CT)
        if want("classes"):
            fig_stab_classes(SC)
        if want("corr"):
            fig_stab_resp_corr(X, CR)
        if want("change"):
            fig_stab_change_corr(X, CR)
        if want("overlap"):
            OV, nov = fig_stab_overlap(T)
        if want("maps"):
            mcounts = {m: fig_maps(T, cen, P, m) for m in MO}
        print(lg, "done", flush=True)
    if a.only:
        return
    CT.to_csv(os.path.join(OUT, "cluster_tests.csv"), index=False)
    SC.to_csv(os.path.join(OUT, "stab_classes.csv"), index=False)
    CR.to_csv(os.path.join(OUT, "stab_resp_corr.csv"), index=False)
    OV.to_csv(os.path.join(OUT, "stab_overlap.csv"), index=False)
    write_summary(T, Tp, P, combos, CT, SC, CR, OV, nov, cl_files, mcounts, dz)


def _chk(d):
    """slide-text length limits of the brief (key <= 150, bottom <= 230 characters)"""
    for k in ("key_en", "key_kr"):
        if len(d[k]) > 150:
            print(f"  WARNING {k} {len(d[k])} chars: {d[k]}")
    for k in ("bottom_en", "bottom_kr"):
        if len(d[k]) > 230:
            print(f"  WARNING {k} {len(d[k])} chars: {d[k]}")
    return d


def write_summary(T, Tp, P, combos, CT, SC, CR, OV, nov, cl_files, mcounts, dz):
    F = {}
    lb = 100 * T[(T.phase == "pre") & (T.event == "Light touch")].cls.isin(["activated", "inhibited"]).mean()
    nt, ns = len(Tp), int((Tp.p < 0.05).sum())
    tot = CT[(CT.kind == "condition") & (CT.level == "all")].iloc[0]
    cm = CT[CT.kind == "mouse"].set_index("level")
    cc = CT[CT.kind == "condition"].set_index("level")
    cg = CT[CT.kind == "group"].set_index("level")
    sig = Tp[Tp.p < 0.05].sort_values("p")
    sig_list = [dict(mouse=r.mouse, day=int(r.day), phase=r.phase, condition=r.condition, event=r.event, group=GRP[r.group],
                     n=int(r.n), nnd_px=round(float(r.nnd), 1), nnd_random_px=round(float(r.nnd_null), 1), z=round(float(r.z), 2),
                     p=round(float(r.p), 4)) for r in sig.itertuples()]
    pages = [combos[i:i + 4] for i in range(0, len(combos), 4)]
    pc = lambda c: [f"{x['mouse']} D{x['day']} {SH[x['event']]}" for x in c]           # noqa: E731
    mice_sig = {m: int(cm.at[m, "n_sig"]) for m in MO}
    base_notes_en = (f"Clustering test of P4_topography (ca_event_analysis.plot4_topography; recomputed here from cells_classes.csv "
                     f"with the same code and seed: max |z difference| {dz:.1e}): per session x event and group (UP+DOWN pooled, UP, "
                     f"DOWN; tested when 4 <= n <= all - 3) the mean nearest-neighbour distance (px) of the group vs 500 random draws "
                     f"of the same number of that session's neurons; z = (random mean - observed) / random SD; p one-sided. "
                     f"Tests with p < 0.05: " + "; ".join(f"{s['mouse']} D{s['day']} {s['condition']} {SH[s['event']]} {s['group']} "
                                                          f"n={s['n']} NN {s['nnd_px']} vs {s['nnd_random_px']} px z={s['z']} p={s['p']}"
                                                          for s in sig_list) + ". Pixel size not known here (scale bar in px).")
    base_notes_kr = (f"P4_topography 군집 검정 (cells_classes.csv로 같은 코드와 seed로 다시 계산: z 차이 최대 {dz:.1e}): 세션 x 사건, "
                     f"그룹 (UP+DOWN 합침, UP, DOWN; 4 <= n <= 전체 - 3일 때 검정)마다 그룹의 평균 최근접 이웃 거리 (px) vs 그 세션 "
                     f"뉴런에서 같은 수를 무작위로 500번 뽑은 거리; z = (무작위 평균 - 관측) / 무작위 SD; p 단측. p < 0.05 검정: "
                     + "; ".join(f"{s['mouse']} D{s['day']} {s['condition']} {SH[s['event']]} {s['group']} n={s['n']} NN "
                                 f"{s['nnd_px']} vs {s['nnd_random_px']} px z={s['z']} p={s['p']}" for s in sig_list)
                     + ". 픽셀 크기는 여기서 모름 (scale bar는 px).")
    for i, c in enumerate(pages):
        name = "CLUSTER_maps.png" if i == 0 else f"CLUSTER_maps_p{i + 1}.png"
        F[name] = _chk(dict(
            numbers=dict(n_tests=nt, n_sig=ns, expected=round(0.05 * nt, 2), sig_per_mouse=mice_sig, pairs_on_page=pc(c),
                         sig_tests=sig_list),
            title_en="Where are the clustered responsive neurons?" + ("" if i == 0 else f" ({i + 1}/{len(pages)})"),
            title_kr="군집된 반응 뉴런은 어디에 있나?" + ("" if i == 0 else f" ({i + 1}/{len(pages)})"),
            key_en=(f"Only {ns} of {nt} maps had clustered responsive neurons (p < 0.05; {0.05 * nt:.1f} expected by chance), "
                    f"all in 5941 ({mice_sig['5941']}) and 5939 ({mice_sig['5939']})"),
            key_kr=(f"군집된 반응 뉴런은 {nt}개 지도 중 {ns}개뿐 (p < 0.05; 우연 기대 {0.05 * nt:.1f}개), 모두 5941 "
                    f"({mice_sig['5941']}개)과 5939 ({mice_sig['5939']}개)"),
            bottom_en=(f"Field of view (max dF/F projection, footprint outlines) of every session x event with a clustered group, "
                       f"next to the other session of that day; nearest-neighbour distance vs 500 random draws; {nt} tests."),
            bottom_kr=(f"군집 그룹이 있는 모든 세션 x 사건의 시야 (max dF/F 투영, footprint 윤곽)와 그날 다른 세션; 최근접 이웃 거리 vs "
                       f"무작위 500번; 검정 {nt}개."),
            notes_en=base_notes_en + " Maps sorted by p; " + ", ".join(pc(c)) + " on this page.",
            notes_kr=base_notes_kr + " p 순서; 이 페이지: " + ", ".join(pc(c)) + ".",
            section="Q3 neurons", priority="main" if i == 0 else "backup"))
    F["CLUSTER_summary.png"] = _chk(dict(
        numbers=CT.round(4).to_dict("records"),
        title_en="Is the clustering more than chance?", title_kr="군집이 우연 이상인가?",
        key_en=(f"No: {ns}/{nt} maps p < 0.05 vs {0.05 * nt:.1f} expected (binomial p = {tot.p_binom:.2f}); baseline "
                f"{cc.at['baseline', 'n_sig']}, vehicle {cc.at['vehicle', 'n_sig']}, SBI-553 {cc.at['SBI-553', 'n_sig']}"),
        key_kr=(f"아니다: {nt}개 중 {ns}개 p < 0.05, 기대 {0.05 * nt:.1f}개 (이항 p = {tot.p_binom:.2f}); 기준 "
                f"{cc.at['baseline', 'n_sig']}, vehicle {cc.at['vehicle', 'n_sig']}, SBI-553 {cc.at['SBI-553', 'n_sig']}개"),
        bottom_en=(f"Number of session x event x group maps with clustering p < 0.05 vs 5 % of the tests, by condition, group and "
                   f"mouse; binomial test, one-sided; {nt} tests (not independent)."),
        bottom_kr=(f"군집 p < 0.05인 세션 x 사건 x 그룹 지도 수 vs 검정의 5 %, 조건 / 그룹 / 마우스별; 이항 검정, 단측; 검정 {nt}개 "
                   f"(독립 아님)."),
        notes_en=(f"Per condition: baseline {cc.at['baseline', 'n_sig']}/{cc.at['baseline', 'n_tests']} (p = "
                  f"{cc.at['baseline', 'p_binom']:.2f}), after vehicle {cc.at['vehicle', 'n_sig']}/{cc.at['vehicle', 'n_tests']} (p = "
                  f"{cc.at['vehicle', 'p_binom']:.2f}), after SBI-553 {cc.at['SBI-553', 'n_sig']}/{cc.at['SBI-553', 'n_tests']} (p = "
                  f"{cc.at['SBI-553', 'p_binom']:.2f}). Per group: UP+DOWN {cg.at['responsive', 'n_sig']}/{cg.at['responsive', 'n_tests']}, "
                  f"UP {cg.at['activated', 'n_sig']}/{cg.at['activated', 'n_tests']}, DOWN {cg.at['inhibited', 'n_sig']}/"
                  f"{cg.at['inhibited', 'n_tests']}. Per mouse: " + ", ".join(f"{m} {cm.at[m, 'n_sig']}/{cm.at[m, 'n_tests']} (p = "
                                                                              f"{cm.at[m, 'p_binom']:.3f})" for m in MO)
                  + ". Only 5941 exceeds 5 % (it has clustered maps at baseline and after SBI-553 on Day 1); the tests share "
                    "neurons (UP+DOWN contains UP and DOWN, events overlap), so the binomial p is optimistic."),
        notes_kr=(f"조건별: 기준 {cc.at['baseline', 'n_sig']}/{cc.at['baseline', 'n_tests']} (p = {cc.at['baseline', 'p_binom']:.2f}), "
                  f"vehicle 후 {cc.at['vehicle', 'n_sig']}/{cc.at['vehicle', 'n_tests']} (p = {cc.at['vehicle', 'p_binom']:.2f}), "
                  f"SBI-553 후 {cc.at['SBI-553', 'n_sig']}/{cc.at['SBI-553', 'n_tests']} (p = {cc.at['SBI-553', 'p_binom']:.2f}). "
                  f"그룹별: UP+DOWN {cg.at['responsive', 'n_sig']}/{cg.at['responsive', 'n_tests']}, UP {cg.at['activated', 'n_sig']}/"
                  f"{cg.at['activated', 'n_tests']}, DOWN {cg.at['inhibited', 'n_sig']}/{cg.at['inhibited', 'n_tests']}. 마우스별: "
                  + ", ".join(f"{m} {cm.at[m, 'n_sig']}/{cm.at[m, 'n_tests']} (p = {cm.at[m, 'p_binom']:.3f})" for m in MO)
                  + ". 5941만 5 %를 넘음 (Day 1 기준과 SBI-553 후 모두 군집 지도); 검정들이 뉴런을 공유하므로 (UP+DOWN은 UP과 DOWN을 "
                    "포함, 사건끼리 겹침) 이항 p는 낙관적."),
        section="Q3 neurons", priority="backup"))
    # ---- stability
    S = SC[SC.tracking == "A+B"].set_index(["kind", "event"])
    SA = SC[SC.tracking == "A"].set_index(["kind", "event"])
    nb = int((S.loc["base"].p < 0.05).sum())
    ni = int((S.loc["inj"].p < 0.05).sum())
    nc = int((S.loc["chg"].p < 0.05).sum())
    pin_b, pin_i = S.loc[("base", "Pin prick")], S.loc[("inj", "Pin prick")]
    mild_i = S.loc[("inj", "Mild touch")]

    def rowtxt(kind, Sx, lg):
        return "; ".join(f"{SH[e]} {Sx.loc[(kind, e)].n_same}/{Sx.loc[(kind, e)].n_ref} = {Sx.loc[(kind, e)].pct:.0f}% vs "
                         f"{Sx.loc[(kind, e)].chance:.0f}% ({ptxt(Sx.loc[(kind, e)].p, True)})" for e in EV7)
    F["STAB_classes.png"] = _chk(dict(
        numbers=SC.round(4).to_dict("records"),
        title_en="Do the tracked neurons keep their class across days and drugs?",
        title_kr="추적 뉴런은 날과 약물이 바뀌어도 class를 유지하는가?",
        key_en=(f"Baseline classes recur across days ({nb}/7 events > chance; pin {pin_b.pct:.0f}% vs {pin_b.chance:.0f}%), but "
                f"after vehicle vs SBI-553 pin is at chance ({pin_i.pct:.0f}% vs {pin_i.chance:.0f}%)"),
        key_kr=(f"기준 class는 날을 넘어 반복 (7개 중 {nb}개 사건 > 우연; pin {pin_b.pct:.0f}% vs {pin_b.chance:.0f}%), 그러나 "
                f"vehicle 후 vs SBI-553 후 pin은 우연 수준 ({pin_i.pct:.0f}% vs {pin_i.chance:.0f}%)"),
        bottom_en=(f"{int(S.n_tracked.iloc[0])} tracked neurons (4 mice): % of UP / DOWN neurons with the same class in the second "
                   f"session (C: same change from baseline) vs neuron identities shuffled within mouse (1000); one-sided."),
        bottom_kr=(f"추적 뉴런 {int(S.n_tracked.iloc[0])}개 (4마리): UP / DOWN 뉴런 중 둘째 세션에서도 같은 class인 비율 (C: 기준 대비 같은 "
                   f"변화) vs 마우스 안 뉴런 셔플 (1000번); 단측."),
        notes_en=(f"A Day-1 vs Day-2 baseline: {rowtxt('base', S, 'en')}. B after vehicle vs after SBI-553: {rowtxt('inj', S, 'en')}. "
                  f"C change from the same-day baseline (rose / fell), vehicle day vs SBI-553 day: {rowtxt('chg', S, 'en')}. "
                  f"Above chance: A {nb}/7, B {ni}/7, C {nc}/7 events. Mild after the injections is also at chance "
                  f"({mild_i.pct:.0f}% vs {mild_i.chance:.0f}%). Reverse direction (reference = second session): A pin "
                  f"{S.loc[('base', 'Pin prick')].pct_reverse:.0f}% of {S.loc[('base', 'Pin prick')].n_ref_reverse}, B pin "
                  f"{S.loc[('inj', 'Pin prick')].pct_reverse:.0f}% of {S.loc[('inj', 'Pin prick')].n_ref_reverse}. Sensitivity, track "
                  f"class A only ({int(SA.n_tracked.iloc[0])} neurons): A {int((SA.loc['base'].p < 0.05).sum())}/7, B "
                  f"{int((SA.loc['inj'].p < 0.05).sum())}/7, C {int((SA.loc['chg'].p < 0.05).sum())}/7 events above chance; pin A "
                  f"{SA.loc[('base', 'Pin prick')].pct:.0f}% vs {SA.loc[('base', 'Pin prick')].chance:.0f}%, B "
                  f"{SA.loc[('inj', 'Pin prick')].pct:.0f}% vs {SA.loc[('inj', 'Pin prick')].chance:.0f}%. Drug order = day "
                  f"(5941, 5939 SBI-553 on Day 1; 5937, 1191 on Day 2): B and C also compare Day 1 with Day 2. Per-mouse markers "
                  f"are descriptive (4 mice)."),
        notes_kr=(f"A Day-1 vs Day-2 기준: {rowtxt('base', S, 'kr')}. B vehicle 후 vs SBI-553 후: {rowtxt('inj', S, 'kr')}. "
                  f"C 같은 날 기준 대비 변화 (상승 / 하강), vehicle 날 vs SBI-553 날: {rowtxt('chg', S, 'kr')}. 우연 이상: A {nb}/7, "
                  f"B {ni}/7, C {nc}/7 사건. 주사 후 mild도 우연 수준 ({mild_i.pct:.0f}% vs {mild_i.chance:.0f}%). 반대 방향 (기준 = "
                  f"둘째 세션): A pin {S.loc[('base', 'Pin prick')].pct_reverse:.0f}% / {S.loc[('base', 'Pin prick')].n_ref_reverse}, "
                  f"B pin {S.loc[('inj', 'Pin prick')].pct_reverse:.0f}% / {S.loc[('inj', 'Pin prick')].n_ref_reverse}. 민감도, track "
                  f"class A만 ({int(SA.n_tracked.iloc[0])}개): A {int((SA.loc['base'].p < 0.05).sum())}/7, B "
                  f"{int((SA.loc['inj'].p < 0.05).sum())}/7, C {int((SA.loc['chg'].p < 0.05).sum())}/7 사건 우연 이상; pin A "
                  f"{SA.loc[('base', 'Pin prick')].pct:.0f}% vs {SA.loc[('base', 'Pin prick')].chance:.0f}%, B "
                  f"{SA.loc[('inj', 'Pin prick')].pct:.0f}% vs {SA.loc[('inj', 'Pin prick')].chance:.0f}%. 약물 순서 = 날 (5941, "
                  f"5939는 Day 1에 SBI-553; 5937, 1191은 Day 2): B와 C도 Day 1과 Day 2를 비교함. 마우스 표식은 기술적 (4마리)."),
        section="Q3 neurons", priority="main"))
    R = CR.set_index(["kind", "event"])
    rb = R.loc["base"]
    ri = R.loc["inj"]
    rcg = R.loc["chg"]
    sb = rb[rb.p_shuffle < 0.05]
    si = ri[ri.p_shuffle < 0.05]

    def rtxt(Rx):
        return "; ".join(f"{SH[e]} rho {Rx.loc[e].rho:.2f} ({ptxt(Rx.loc[e].p_shuffle, True)}; mice " +
                         ", ".join(f"{m} {Rx.loc[e][f'rho_{m}']:.2f}" for m in MO) + ")" for e in EV7)
    F["STAB_resp_corr.png"] = _chk(dict(
        numbers=CR[CR.kind != "chg"].round(4).to_dict("records"),
        title_en="Does a tracked neuron respond the same in two sessions?",
        title_kr="추적 뉴런은 두 세션에서 같은 반응을 보이는가?",
        key_en=(f"Yes: rho {sb.rho.min():.2f}-{sb.rho.max():.2f} for {len(sb)}/7 events between the baselines and "
                f"{si.rho.min():.2f}-{si.rho.max():.2f} for {len(si)}/7 after vehicle vs after SBI-553"),
        key_kr=(f"그렇다: 기준끼리 7개 중 {len(sb)}개 사건 rho {sb.rho.min():.2f}-{sb.rho.max():.2f}, vehicle 후 vs SBI-553 후 "
                f"{len(si)}개 rho {si.rho.min():.2f}-{si.rho.max():.2f}"),
        bottom_en=(f"Response (best window, trial mean z) of each of the {int(CR.n.iloc[0])} tracked neurons in two sessions; "
                   f"Spearman rho over 4 mice, p from 1000 shuffles of neuron identity within mouse (two-sided)."),
        bottom_kr=(f"추적 뉴런 {int(CR.n.iloc[0])}개 각각의 두 세션 반응 (최적 창, 시행 평균 z); 4마리 Spearman rho, p = 마우스 안 "
                   f"뉴런 셔플 1000번 (양측)."),
        notes_en=(f"Baselines (Day 1 vs Day 2): {rtxt(rb)}. After vehicle vs after SBI-553: {rtxt(ri)}. Light touch at baseline is "
                  f"not reproducible (only {lb:.0f}% of the baseline neurons respond to light). The baseline rho is higher than the injection rho for "
                  f"{int((rb.rho > ri.rho).sum())}/7 events (descriptive). Drug order = day."),
        notes_kr=(f"기준끼리 (Day 1 vs Day 2): {rtxt(rb)}. vehicle 후 vs SBI-553 후: {rtxt(ri)}. 기준의 light touch는 재현 안 됨 "
                  f"(기준 뉴런의 {lb:.0f}%만 light에 반응). 기준 rho가 주사 후 rho보다 큰 사건 {int((rb.rho > ri.rho).sum())}/7 (기술적). 약물 순서 = 날."),
        section="Q3 neurons", priority="main"))
    F["STAB_change_corr.png"] = _chk(dict(
        numbers=CR[CR.kind == "chg"].round(4).to_dict("records"),
        title_en="Baseline-normalized: does the change after the injection repeat in the same neuron?",
        title_kr="기준 정규화: 주사 후 변화가 같은 뉴런에서 반복되는가?",
        key_en=(f"No: the change from the same-day baseline is unrelated between the vehicle and SBI-553 days (rho "
                f"{rcg.rho.min():.2f} to {rcg.rho.max():.2f}, {int((rcg.p_shuffle < 0.05).sum())}/7 p < 0.05)"),
        key_kr=(f"아니다: 같은 날 기준 대비 변화는 vehicle 날과 SBI-553 날 사이에 관계 없음 (rho {rcg.rho.min():.2f} ~ "
                f"{rcg.rho.max():.2f}, 7개 중 {int((rcg.p_shuffle < 0.05).sum())}개 p < 0.05)"),
        bottom_en=(f"Per tracked neuron (n = {int(CR.n.iloc[0])}): (after - baseline) on the vehicle day vs on the SBI-553 day; "
                   f"Spearman rho, p from 1000 within-mouse shuffles; right: rho of the three comparisons."),
        bottom_kr=(f"추적 뉴런마다 (n = {int(CR.n.iloc[0])}): vehicle 날 (후 - 기준) vs SBI-553 날 (후 - 기준); Spearman rho, p = "
                   f"마우스 안 셔플 1000번; 오른쪽: 세 비교의 rho."),
        notes_en=(f"Change correlation: {rtxt(rcg)}. A neuron's response is stable (STAB_resp_corr) but its change after an "
                  f"injection does not repeat on the other day, so the after-injection changes are not a fixed property of the "
                  f"neuron (they look like session-to-session variation). Drug order = day."),
        notes_kr=(f"변화 상관: {rtxt(rcg)}. 뉴런의 반응은 안정적이지만 (STAB_resp_corr) 주사 후 변화는 다른 날 반복되지 않음 - 주사 후 "
                  f"변화는 뉴런의 고정된 성질이 아님 (세션 간 변동처럼 보임). 약물 순서 = 날."),
        section="Q3 neurons", priority="backup"))
    O = OV.copy()
    mj = O.groupby("condition").jaccard.mean()
    Ov = O[O.condition == "vehicle"].set_index(["event1", "event2"]).jaccard
    Osb = O[O.condition == "SBI-553"].set_index(["event1", "event2"])
    dd = (Osb.jaccard - Ov).sort_values()
    nbx = int((Osb.p_fisher_sbi_vs_veh < 0.05).sum())
    lo2, hi2 = dd.index[0], dd.index[-1]
    T_ = T.assign(r=T.cls.isin(["activated", "inhibited"]))
    pr = (100 * T_.groupby(["event", "condition"]).r.mean()).round(0)
    prt = lambda e: f"{SH[e]} {pr[(e, 'vehicle')]:.0f}% vs {pr[(e, 'SBI-553')]:.0f}%"           # noqa: E731
    F["STAB_overlap.png"] = _chk(dict(
        numbers=dict(mean_jaccard={k: round(float(v), 1) for k, v in mj.items()}, n_neurons=nov,
                     pairs=O.round(4).to_dict("records")),
        title_en="Do the same neurons respond to several events?", title_kr="같은 뉴런이 여러 사건에 반응하는가?",
        key_en=(f"Mean overlap {mj['baseline']:.0f}% baseline, {mj['vehicle']:.0f}% after vehicle, {mj['SBI-553']:.0f}% after "
                f"SBI-553; after SBI-553 pin shares fewer neurons ({SH[lo2[0]]}-{SH[lo2[1]]} {dd.iloc[0]:+.0f} pp)"),
        key_kr=(f"평균 겹침 기준 {mj['baseline']:.0f}%, vehicle 후 {mj['vehicle']:.0f}%, SBI-553 후 {mj['SBI-553']:.0f}%; SBI-553 후 "
                f"pin은 덜 겹침 ({SH[lo2[0]]}-{SH[lo2[1]]} {dd.iloc[0]:+.0f}%p)"),
        bottom_en=(f"Jaccard overlap of the responsive (UP or DOWN) neurons between events, every neuron of each session "
                   f"(baseline {nov['baseline']}, vehicle {nov['vehicle']}, SBI-553 {nov['SBI-553']}); Fisher exact per pair."),
        bottom_kr=(f"사건 간 반응 (UP 또는 DOWN) 뉴런의 Jaccard 겹침, 각 세션의 모든 뉴런 (기준 {nov['baseline']}, vehicle "
                   f"{nov['vehicle']}, SBI-553 {nov['SBI-553']}); 쌍마다 Fisher exact."),
        notes_en=(f"SBI-553 - vehicle per pair (pp): " + "; ".join(f"{SH[a]}-{SH[b]} {v:+.0f}" for (a, b), v in dd.items())
                  + f". Fisher exact p < 0.05 in {nbx}/21 pairs (different neurons and days, neurons pooled, exploratory, no "
                    f"correction). The largest rise: {SH[hi2[0]]}-{SH[hi2[1]]} {dd.iloc[-1]:+.0f} pp. Jaccard depends on how many "
                    f"neurons respond to each event (% responsive after vehicle vs after SBI-553: {prt('Heat')}, "
                    f"{prt('Pin prick')}, {prt('flinch')}, {prt('movement onset')}), so part of the difference follows the % "
                    f"responsive. Drug order = day."),
        notes_kr=(f"쌍별 SBI-553 - vehicle (%p): " + "; ".join(f"{SH[a]}-{SH[b]} {v:+.0f}" for (a, b), v in dd.items())
                  + f". Fisher exact p < 0.05: 21쌍 중 {nbx}쌍 (다른 뉴런과 날, 뉴런 합침, 탐색적, 보정 없음). 가장 큰 증가: "
                    f"{SH[hi2[0]]}-{SH[hi2[1]]} {dd.iloc[-1]:+.0f}%p. Jaccard는 각 사건에 반응하는 뉴런 수에 따라 달라짐 "
                    f"(vehicle 후 vs SBI-553 후 반응 비율: {prt('Heat')}, {prt('Pin prick')}, {prt('flinch')}, "
                    f"{prt('movement onset')}) - 차이의 일부는 반응 비율을 따름. 약물 순서 = 날."),
        section="Q3 neurons", priority="backup"))
    for m in MO:
        n1, n2 = int(T[(T.mouse == m) & (T.day == 1)].cell.nunique()), int(T[(T.mouse == m) & (T.day == 2)].cell.nunique())
        ntr = len(P[m])
        cnt = {f"D{d} {ph} {SH[ev]}": dict(up=v[0], down=v[1], n=v[2]) for (d, ph, ev), v in mcounts.get(m, {}).items()}
        F[f"MAPS_{m}.png"] = _chk(dict(
            numbers=dict(n_day1=n1, n_day2=n2, n_tracked=ntr, sbi_day=SBI_DAY[m], counts=cnt,
                         clustering_sig=int(cm.at[m, "n_sig"]), clustering_tests=int(cm.at[m, "n_tests"])),
            title_en=f"{m} ({SEX[m]}): where the responsive neurons are", title_kr=f"{m} ({SEX[m]}): 반응 뉴런의 위치",
            key_en=(f"{m}: {n1} / {n2} neurons (Day 1 / Day 2), {ntr} tracked; {cm.at[m, 'n_sig']} of {cm.at[m, 'n_tests']} "
                    f"clustering tests p < 0.05"),
            key_kr=(f"{m}: 뉴런 {n1} / {n2}개 (Day 1 / Day 2), 추적 {ntr}개; 군집 검정 {cm.at[m, 'n_tests']}개 중 "
                    f"{cm.at[m, 'n_sig']}개 p < 0.05"),
            bottom_en=(f"Field-of-view positions of every neuron in the 4 sessions x 7 events; UP red, DOWN blue (time-shift test, "
                       f"BH q <= 0.20); filled = tracked on both days. SBI-553 on Day {SBI_DAY[m]}."),
            bottom_kr=(f"4개 세션 x 7개 사건에서 모든 뉴런의 시야 위치; UP 빨강, DOWN 파랑 (time-shift 검정, BH q <= 0.20); 채움 = "
                       f"두 날 추적. SBI-553은 Day {SBI_DAY[m]}."),
            notes_en=("Rows: vehicle-day baseline, after vehicle, SBI-553-day baseline, after SBI-553 (not chronological for "
                      "mice with SBI-553 on Day 1). Positions in each day's own coordinates (the days differ by a few pixels; no "
                      "warp). Counts right of each map = UP / DOWN."),
            notes_kr=("행: vehicle 날 기준, vehicle 후, SBI-553 날 기준, SBI-553 후 (Day 1에 SBI-553을 받은 마우스는 시간 순서가 "
                      "아님). 위치는 각 날의 좌표 (몇 픽셀 차이; 변형 없음). 지도 오른쪽 숫자 = UP / DOWN."),
            section="supplement", priority="backup"))
    out = dict(script=os.path.abspath(__file__), created="2026-10-23", topography_check_max_abs_dz=dz,
               n_tracked=int(sum(len(P[m]) for m in MO)), tracked_per_mouse={m: len(P[m]) for m in MO}, figures=F)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("summary.json written:", len(F), "figures", flush=True)


if __name__ == "__main__":
    main()
