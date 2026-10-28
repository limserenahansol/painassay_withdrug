"""ca_flow_classes.py - FLOW (alluvial) charts of the neuron classes, baseline -> after the injection, vehicle day vs SBI-553 day
(2026-10-23).

WHY: the PI could not read the bar chart 'Do the same neurons keep responding after the injection?' (A = % of the baseline-
responsive neurons still responsive, B = % of the baseline non-responsive neurons responsive after the injection). A flow chart
shows, for every neuron of the day, its class at baseline (UP = activated, NR = non-responsive, DOWN = inhibited) and its class
after the injection, so each group can be followed by eye; the baseline-normalized version (each baseline class = 100 %) makes
the vehicle day and the SBI-553 day directly comparable although their baselines differ.

WHAT (every event with data: 4 stimuli, then withdrawal, flinch, movement onset; escape / grooming / attending are drawn only
when both drug days have a baseline AND an injection session with >= 5 events - on 10-23 none of the three qualifies):
  FLOW_<tag>.png          two alluvial panels, vehicle day (blue accent) | SBI-553 day (red accent); left column = class at the
                          baseline session, right column = class after the injection (the same neurons), band heights = neurons,
                          bands coloured by the baseline class, count on every band >= 5 neurons, n per block; under each panel
                          the within-day McNemar test of the UP and DOWN calls (the same neurons, baseline vs after).
  FLOW_<tag>_pct.png      the same flow with each baseline class scaled to 100 % (bands = % of that baseline class) + a panel of
                          the key transitions (stay UP, NR -> UP, UP -> NR, NR -> DOWN): vehicle day vs SBI-553 day, Fisher
                          exact on the pooled neurons (exploratory), per-mouse markers.
  FLOW_summary_stim.png   all stimuli side by side: stay responsive (same class), newly UP among NR, newly DOWN among NR;
  FLOW_summary_beh.png    the same for withdrawal, flinch, movement onset. Bars = pooled neurons, markers = mice, Fisher brackets.
  FLOW_tracked_<tag>.png  the 185 tracked neurons only (analysis\\crossday\\<m>_pairs.csv, A + B): vb -> vi | sb -> si (the same
                          neurons in all four columns) and class after vehicle -> class after SBI-553 (McNemar on the paired UP
                          calls).
  Every figure in English and Korean (<name>_kr.png). Tables flow_counts.csv (every mouse-day and pooled 3 x 3 transition
  counts), flow_tests.csv, tracked_classes.csv; summary.json (numbers + proposed slide texts per figure).

INPUTS: analysis\\event_analysis\\cells_classes.csv (timing-corrected classes: within-block time-shift test, BH q <= 0.20 per
session, best window per event: stimuli 0-0.5 s after vs the 0.5 s before, behaviours 0-0.5 s after vs the 1 s before),
analysis\\crossday\\<m>_pairs.csv. Drug order (crossover, confounded with day): Day 1 SBI-553 = 5941, 5939; Day 2 SBI-553 =
5937, 1191.

OUTPUT: analysis\\pi_extra\\flow\\

RUN:  python -I -X utf8 C:\\Users\\hsollim\\Documents\\HEAL_mini1p_SBI553\\mini1p_pipeline\\ca_flow_classes.py   (~1 min)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (retrying savefig)
import pairstars as PS  # noqa: E402  (every star shows the pair it compares)
import winwords as WW  # noqa: E402  (the response window of each event)
import figfoot  # noqa: E402  (10-24: the methods text under a figure goes to the slide notes, <png>.notes.txt)

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
EA = os.path.join(B, "analysis", os.environ.get("CA_EVENT_OUT", "event_analysis"))
CD = os.path.join(B, "analysis", "crossday")
OUT = os.path.join(B, "analysis", "pi_extra", "flow")
MO = ["5941", "5939", "5937", "1191"]
SBI_DAY = {"5941": 1, "5939": 1, "5937": 2, "1191": 2}
DRUGS = ["vehicle", "SBI-553"]
CC = {"baseline": "#7f8c8d", "vehicle": "#2471a3", "SBI-553": "#c0392b"}
MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#34495e"}
MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
CLS = ["UP", "NR", "DOWN"]
COL = {"UP": "#e74c3c", "NR": "#c8cdd2", "DOWN": "#2e86c1"}
EDG = {"UP": "#a93226", "NR": "#7b8288", "DOWN": "#1b4f72"}
ALPHA = {"UP": 0.42, "NR": 0.62, "DOWN": 0.42}
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]
BEH = ["withdrawal", "flinch", "movement onset"]
EXTRA = ["escape/rearing onset", "grooming onset", "attending onset"]
MAIN_EV = ["Pin prick", "Heat"]
BAND_MIN = 5                     # a band carries its count when it has >= 5 neurons (10-24: was 3; thin-band labels collided)
MOUSE_MIN = 3                    # a mouse marker is drawn when that mouse has >= 3 neurons in the denominator
FS = 12.5                        # 15 x 6.3 in figures go into a 12.4 x 5.2 in box (x 0.825): >= 11 pt here = >= 9 pt on the slide
SCALE_N = 50                     # scale bar of the count flows (neurons)
DRUG_ORDER_EN = "Drug order (crossover, confounded with day): Day 1 SBI-553 = 5941, 5939; Day 2 SBI-553 = 5937, 1191."
DRUG_ORDER_KR = "약물 순서 (crossover, 날과 교락): Day 1 SBI-553 = 5941, 5939; Day 2 SBI-553 = 5937, 1191."
KEY_KR = "* p < 0.05, ** p < 0.01, *** p < 0.001, ns = 유의하지 않음"
plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13, "axes.labelsize": 12.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False})


def L(lg, en, kr):
    return kr if lg == "kr" else en


def font(lg):
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"] if lg == "kr" else ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def tag_of(ev):
    return ev.replace(" ", "_").replace("/", "_")


def ptxt(p):
    if p is None or not np.isfinite(p):
        return "n.a."
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def wtxt(ev, lg):
    pre, gap, a, b = WW.of(ev)
    return L(lg, f"{a:g}-{b:g} s after vs the {pre:g} s before", f"후 {a:g}-{b:g} s vs 직전 {pre:g} s")


def fisher(k1, n1, k2, n2):
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def mcnemar(b_, c_):
    n = int(b_ + c_)
    return float(binomtest(int(b_), n, 0.5).pvalue) if n else np.nan


def pct(k, n):
    return 100.0 * k / n if n > 0 else np.nan


# ------------------------------------------------------------------------------------------------ text layout helpers
def vwidth(s):
    return sum(1.8 if ord(ch) > 0x1100 else 1.0 for ch in s)


def wrap(txt, maxw):
    lines, cur = [], ""
    for w in txt.split(" "):
        cand = f"{cur} {w}" if cur else w
        if vwidth(cand) > maxw and cur:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


KEY_FS = 11.5                    # the one-line star key that figfoot draws at the bottom


def foot_lines(fig, txt, fs=11):
    """10-24 (user: "all details put in note not main slides"): the methods text is no longer wrapped under the figure; it goes
    whole to the slide notes (figfoot). Kept as a one-item list for the callers."""
    return [" ".join(str(txt).split())]


def foot(fig, lines, lang=None, star=True):
    """the methods text -> <png>.notes.txt (slide notes); in the figure only the one-line star key (star=False: none, for figures
    without stars)"""
    figfoot.note(fig, " ".join(lines), lang, fs=KEY_FS, y=0.008, star=star)


def foot_h(fig, n=1, fs=KEY_FS):
    """figure fraction kept free at the bottom for the one-line star key (n is ignored: the key is always one line)"""
    return 0.012 + fs * 1.25 / 72 / fig.get_figheight()


def key_block(fig, handles, note, y, xc=0.5, fs=12, maxw=0.97, ncol=None, gap_pt=16, xmax=0.997, **kw):
    """the legend of `handles` plus a short text `note` (what an encoding means, e.g. the band colour), top at y (figure
    fraction), centred at xc (but not past xmax): on one row when both fit in maxw, else the note on its own row under the
    legend. Returns the lowest figure y used."""
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, y), ncol=ncol or len(handles), frameon=False,
                     fontsize=fs, borderaxespad=0, borderpad=0.15, columnspacing=1.3, handletextpad=0.5, **kw)
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bl = leg.get_window_extent(r).transformed(inv)
    if not note:
        leg.set_bbox_to_anchor((min(xc - bl.width / 2, xmax - bl.width), y), transform=fig.transFigure)
        return y - bl.height
    t = fig.text(0, y, note, ha="left", va="center", fontsize=fs, color="0.15")
    bt = t.get_window_extent(r).transformed(inv)
    g = gap_pt / 72 / fig.get_figwidth()
    if bl.width + g + bt.width <= maxw:
        x0 = xc - (bl.width + g + bt.width) / 2
        leg.set_bbox_to_anchor((x0, y), transform=fig.transFigure)
        t.set_position((x0 + bl.width + g, y - bl.height / 2))
        return y - bl.height
    leg.set_bbox_to_anchor((xc - bl.width / 2, y), transform=fig.transFigure)
    t.set_position((xc, y - bl.height - 0.004))
    t.set_ha("center")
    t.set_va("top")
    return y - bl.height - 0.004 - bt.height


def pt2y(ax, pt):
    h_in = ax.get_position().height * ax.figure.get_figheight()
    lo, hi = ax.get_ylim()
    return pt * (hi - lo) / (h_in * 72)


def repel(ys, sep, lo, hi):
    """ys = wanted heights top -> bottom; push apart to >= sep, keep inside [lo, hi] when possible"""
    y = list(ys)
    for k in range(1, len(y)):
        if y[k - 1] - y[k] < sep:
            y[k] = y[k - 1] - sep
    if y and y[-1] < lo:
        d = lo - y[-1]
        y = [v + d for v in y]
    if y and y[0] > hi:                                          # no room: centre the stack on the room
        mid = (lo + hi) / 2
        span = sep * (len(y) - 1)
        y = [mid + span / 2 - k * sep for k in range(len(y))]
    return y


def repel_min(ys, sep, lo, hi):
    """ys = wanted heights top -> bottom: push down to >= sep apart (each label moves only as far as needed), then up from the
    floor lo, then down from the ceiling hi"""
    y = list(ys)
    for k in range(1, len(y)):
        y[k] = min(y[k], y[k - 1] - sep)
    if y:
        y[-1] = max(y[-1], lo)
        for k in range(len(y) - 2, -1, -1):
            y[k] = max(y[k], y[k + 1] + sep)
        y[0] = min(y[0], hi)
        for k in range(1, len(y)):
            y[k] = min(y[k], y[k - 1] - sep)
    return y


# ------------------------------------------------------------------------------------------------ data
def load():
    T = pd.read_csv(os.path.join(EA, "cells_classes.csv"))
    T["mouse"] = T.mouse.astype(str)
    T["drug"] = np.where([SBI_DAY[m] == d for m, d in zip(T.mouse, T.day)], "SBI-553", "vehicle")
    T["c"] = T.cls.map({"activated": "UP", "inhibited": "DOWN", "non-responsive": "NR"})
    if T.c.isna().any():
        raise SystemExit(f"unknown classes: {T.cls[T.c.isna()].unique()}")
    a = T[T.phase == "pre"][["mouse", "day", "drug", "event", "cell", "c"]].rename(columns={"c": "base"})
    b = T[T.phase == "post"][["mouse", "day", "event", "cell", "c"]].rename(columns={"c": "inj"})
    J = a.merge(b, on=["mouse", "day", "event", "cell"])
    return T, J


def mat(Z, a="base", b="inj"):
    M = np.zeros((3, 3), int)
    for i, ci in enumerate(CLS):
        for j, cj in enumerate(CLS):
            M[i, j] = int(((Z[a] == ci) & (Z[b] == cj)).sum())
    return M


def flows(J, ev):
    """pooled and per-mouse 3 x 3 matrices (rows = baseline class, cols = class after the injection) per drug day"""
    out = {}
    for drug in DRUGS:
        Z = J[(J.event == ev) & (J.drug == drug)]
        out[drug] = dict(M=mat(Z), mice={m: mat(Z[Z.mouse == m]) for m in MO if (Z.mouse == m).any()},
                         n_mice=int(Z.mouse.nunique()), n=len(Z), per_mouse_n={m: int((Z.mouse == m).sum()) for m in MO})
    return out


def within_mcnemar(M):
    """same neurons, baseline vs after: UP calls and DOWN calls (exact McNemar)"""
    r = {}
    for k, c in ((0, "UP"), (2, "DOWN")):
        b_ = int(M[k, :].sum() - M[k, k])                        # c at baseline, not after
        c_ = int(M[:, k].sum() - M[k, k])                        # not at baseline, c after
        r[c] = dict(base=int(M[k, :].sum()), after=int(M[:, k].sum()), lost=b_, gained=c_, p=mcnemar(b_, c_))
    return r


KEYTR = [("stay_up", 0, 0, ("stay\nUP", "UP\n유지")), ("nr_up", 1, 0, ("NR\n→ UP", "NR\n→ UP")),
         ("up_nr", 0, 1, ("UP\n→ NR", "UP\n→ NR")), ("nr_down", 1, 2, ("NR\n→ DOWN", "NR\n→ DOWN"))]
KEYTR_TXT = {"stay_up": ("stay UP", "UP 유지"), "nr_up": ("NR -> UP", "NR -> UP"), "up_nr": ("UP -> NR", "UP -> NR"),
             "nr_down": ("NR -> DOWN", "NR -> DOWN")}


def measure(M, kind):
    """(k, n) of a transition: a key transition (stay_up, nr_up, up_nr, nr_down) or a summary measure"""
    if kind == "stay_resp":
        return int(M[0, 0] + M[2, 2]), int(M[0].sum() + M[2].sum())
    if kind == "new_up":
        return int(M[1, 0]), int(M[1].sum())
    if kind == "new_down":
        return int(M[1, 2]), int(M[1].sum())
    for nm, i, j, _ in KEYTR:
        if nm == kind:
            return int(M[i, j]), int(M[i].sum())
    raise KeyError(kind)


def compare(F, kind):
    """vehicle day vs SBI-553 day for one transition: pooled k / n / %, Fisher p, per-mouse %"""
    kv, nv = measure(F["vehicle"]["M"], kind)
    ks, ns = measure(F["SBI-553"]["M"], kind)
    r = dict(k_veh=kv, n_veh=nv, pct_veh=pct(kv, nv), k_sbi=ks, n_sbi=ns, pct_sbi=pct(ks, ns), p_fisher=fisher(ks, ns, kv, nv))
    for drug, sh in (("vehicle", "veh"), ("SBI-553", "sbi")):
        r[f"mice_{sh}"] = {}
        for m, Mm in F[drug]["mice"].items():
            k, n = measure(Mm, kind)
            r[f"mice_{sh}"][m] = dict(k=k, n=n, pct=pct(k, n))
    return r


def dominance(F, kinds, ev, share=0.7):
    """denominators (>= 5 neurons) of which one mouse holds >= share: the pooled % then mostly reflects that mouse"""
    out = []
    for kind in kinds:
        for drug in DRUGS:
            k, n = measure(F[drug]["M"], kind)
            if n < 5:
                continue
            for m, Mm in F[drug]["mice"].items():
                km, nm = measure(Mm, kind)
                if nm / n >= share:
                    out.append(dict(event=ev, transition=kind, drug=drug, mouse=m, n_mouse=nm, n=n, k_mouse=km, k=k))
    return out


def dom_txt(D, lg, names):
    if not D:
        return ""
    if lg == "kr":
        return ("한 마우스가 분모의 대부분: " + "; ".join(f"{d['event']} {names[d['transition']][1]} ({d['drug']} 날) {d['n']} 중 "
                                                    f"{d['n_mouse']} = {d['mouse']}" for d in D) + ". ")
    return ("One mouse holds most of the denominator: " + "; ".join(f"{d['event']} {names[d['transition']][0]} ({d['drug']} day) "
                                                                    f"{d['n_mouse']} of {d['n']} from {d['mouse']}" for d in D) + ". ")


# ------------------------------------------------------------------------------------------------ drawing: the alluvial
def tops(h, top, gap):
    t, y = [], top
    for v in h:
        t.append(y)
        y -= v + gap
    return t


def band(ax, xa, xb, yl, yr, w, color, alpha, edge):
    x = np.linspace(xa, xb, 90)
    t = (x - xa) / (xb - xa)
    s = t * t * (3 - 2 * t)
    top = yl + (yr - yl) * s
    ax.fill_between(x, top - w, top, color=color, alpha=alpha, lw=0, zorder=2)
    for yy in (top, top - w):
        ax.plot(x, yy, color=edge, lw=0.6, alpha=min(1.0, alpha + 0.25), zorder=2.1)


def draw_flow(ax, M, xs, top, gap, use_pct, lg, lab_left=True, lab_right=True, right_n=True, bw=0.075, fs=FS):
    """one alluvial (two columns) of the 3 x 3 matrix M into ax (ylim must be set). use_pct: each baseline class = 100."""
    x0, x1 = xs
    n_i, n_j = M.sum(1), M.sum(0)
    if use_pct:
        W = np.where(n_i[:, None] > 0, 100.0 * M / np.maximum(n_i[:, None], 1), 0.0)
        hL = np.array([100.0, 100.0, 100.0])
        hR = W.sum(0)
    else:
        W = M.astype(float)
        hL, hR = n_i.astype(float), n_j.astype(float)
    tL, tR = tops(hL, top, gap), tops(hR, top, gap)
    segL, segR = {}, {}
    for i in range(3):
        y = tL[i]
        for j in range(3):
            segL[i, j] = y
            y -= W[i, j]
    for j in range(3):
        y = tR[j]
        for i in range(3):
            segR[i, j] = y
            y -= W[i, j]
    xa, xb = x0 + bw / 2, x1 - bw / 2
    few = [use_pct and 0 < n_i[i] < 5 for i in range(3)]
    for i in (1, 0, 2):                                          # the big grey NR bands underneath
        for j in range(3):
            if W[i, j] > 0:
                band(ax, xa, xb, segL[i, j], segR[i, j], W[i, j], COL[CLS[i]], ALPHA[CLS[i]] * (0.5 if few[i] else 1),
                     EDG[CLS[i]])
    for i in range(3):                                           # blocks
        c = CLS[i]
        if use_pct and n_i[i] == 0:
            ax.add_patch(Rectangle((x0 - bw / 2, tL[i] - hL[i]), bw, hL[i], facecolor="white", edgecolor=EDG[c], lw=1.0,
                                   hatch="//", zorder=3))
        else:
            ax.add_patch(Rectangle((x0 - bw / 2, tL[i] - hL[i]), bw, max(hL[i], 0.4), facecolor=COL[c], edgecolor=EDG[c],
                                   lw=0.9, alpha=0.55 if few[i] else 1, zorder=3))
        ax.add_patch(Rectangle((x1 - bw / 2, tR[i] - hR[i]), bw, max(hR[i], 0.4), facecolor=COL[c], edgecolor=EDG[c], lw=0.9,
                               zorder=3))
    sep = pt2y(ax, fs * 1.25)
    xl = xa + 0.055
    # counts on the bands, next to the baseline block; all labels of the column pushed apart together (top -> bottom)
    items = [(i, j, segL[i, j] - W[i, j] / 2) for i in range(3) for j in range(3) if M[i, j] >= BAND_MIN]
    if items:
        floor = tL[2] - hL[2] + 0.45 * sep                       # not below the lowest block (the stats text is there)
        got = repel_min([w for _, _, w in items], sep, floor, tL[0] + 0.3 * sep)
        for (i, j, yw), yg in zip(items, got):
            txt = f"{W[i, j]:.0f}% ({M[i, j]})" if use_pct else f"{M[i, j]}"
            if abs(yg - yw) > 0.3 * sep:
                ax.plot([xa + 0.004, xl - 0.006], [yw, yg], color="0.25", lw=0.8, zorder=4.5)
            # opaque box (10-24): bands crossing behind a label no longer run through its digits
            ax.text(xl, yg, txt, ha="left", va="center", fontsize=fs - 0.5, color="0.1", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="0.75", lw=0.5, alpha=1.0))
    sepb = pt2y(ax, (fs + 1) * 1.25)
    if lab_left:
        want = [tL[i] - hL[i] / 2 for i in range(3)]
        got = repel(want, sepb, -1e9, 1e9)
        for i in range(3):
            n = int(n_i[i])
            txt = f"{CLS[i]}  n = {n}"
            if few[i]:
                txt += "  (n < 5)"
            if use_pct and n == 0:
                txt += L(lg, "\n(no neuron)", "\n(뉴런 없음)")
            ax.text(x0 - bw / 2 - 0.025, got[i], txt, ha="right", va="center", fontsize=fs, color=EDG[CLS[i]],
                    fontweight="bold", zorder=5, linespacing=1.05)
    if lab_right:
        want = [tR[j] - hR[j] / 2 for j in range(3)]
        got = repel(want, sepb, -1e9, 1e9)
        for j in range(3):
            txt = f"{CLS[j]}  n = {int(n_j[j])}" if right_n else CLS[j]
            ax.text(x1 + bw / 2 + 0.025, got[j], txt, ha="left", va="center", fontsize=fs, color=EDG[CLS[j]],
                    fontweight="bold", zorder=5)
    return dict(tL=tL, tR=tR, hL=hL, hR=hR)


def n_under(ax, items, fs=12):
    """n of each bar written just under it (x, n, drug); the group labels move down"""
    for x, n, drug in items:
        ax.text(x, -0.012, f"{n}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=fs, color=CC[drug])
    ax.text(-0.52, -0.012, "n", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=fs, color="0.35",
            style="italic")
    ax.tick_params(axis="x", pad=fs * 1.75, length=0)


def scale_bar(ax, x, y0, h, txt, fs=11.5, rotation=90):
    ax.plot([x, x], [y0, y0 + h], color="k", lw=3.2, solid_capstyle="butt", zorder=5, clip_on=False)
    ax.text(x - 0.025, y0 + h / 2, txt, ha="right", va="center", rotation=rotation, fontsize=fs, color="k", zorder=5,
            linespacing=1.1)


def count_scale(ax, g, x0, lg, bw=0.075):
    """10-24: a SCALE_N-neuron bar right next to the baseline NR block (it was a small tick at the far left), under the NR label;
    falls back to the far left of the axes when the NR block is too short to hold it under the label"""
    nr_mid = g["tL"][1] - g["hL"][1] / 2
    dn_mid = g["tL"][2] - g["hL"][2] / 2
    lab = pt2y(ax, FS * 0.75)                                    # half height of a block label
    lo, hi = dn_mid + lab, nr_mid - lab                          # free room between the NR and the DOWN labels
    txt = L(lg, f"{SCALE_N} neurons", f"뉴런 {SCALE_N}")
    if hi - lo >= SCALE_N + 0.3 * lab:                          # the bar fits; its text sits at its middle
        scale_bar(ax, x0 - bw / 2 - 0.05, (lo + hi) / 2 - SCALE_N / 2, SCALE_N, txt, fs=12, rotation=0)
    else:
        scale_bar(ax, ax.get_xlim()[0] + 0.15, g["tL"][1] - g["hL"][1], SCALE_N, txt, fs=12)


def accent(ax, color):
    ax.plot([0.0, 1.0], [1.0, 1.0], transform=ax.transAxes, color=color, lw=5, solid_capstyle="butt", clip_on=False, zorder=6)


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def class_handles(lg, band_note=False):
    """the three class colours; the band-colour rule is a TEXT note next to them (key_block), not an empty white box (10-24)"""
    return [Patch(facecolor=COL["UP"], edgecolor=EDG["UP"], label=L(lg, "UP = activated", "UP = 활성")),
            Patch(facecolor=COL["NR"], edgecolor=EDG["NR"], label=L(lg, "NR = non-responsive", "NR = 무반응")),
            Patch(facecolor=COL["DOWN"], edgecolor=EDG["DOWN"], label=L(lg, "DOWN = inhibited", "DOWN = 억제"))]


def unlab_txt(lg):
    return L(lg, f"bands < {BAND_MIN} neurons unlabelled", f"뉴런 < {BAND_MIN}인 띠는 숫자 생략")


def drug_handles(lg, F=None):
    """bar colours = the drug day (their key sits above the bars, apart from the class key of the flows: 10-24)"""
    out = []
    for d in DRUGS:
        lab = L(lg, f"{d} day", f"{d} 날")
        if F is not None:
            lab += L(lg, f" ({F[d]['n']} neurons)", f" (뉴런 {F[d]['n']})")
        out.append(Patch(color=CC[d], alpha=0.9, label=lab))
    return out


def hatch_handle(lg):
    return Patch(facecolor="white", edgecolor="0.3", hatch="///", lw=0.8,
                 label=L(lg, "hatched = n < 5", "빗금 = n < 5"))


def mouse_handles(lg, ms=9):
    return [Line2D([], [], marker=MK[m], ls="", color=MC[m], markeredgecolor="white", markersize=ms,
                   label=m + L(lg, " (SBI-553 Day 1)" if SBI_DAY[m] == 1 else " (SBI-553 Day 2)",
                               " (SBI-553 Day 1)" if SBI_DAY[m] == 1 else " (SBI-553 Day 2)")) for m in MO]


def mouse_n_txt(F, drug):
    return ", ".join(f"{m} {F[drug]['per_mouse_n'][m]}" for m in MO)


# ------------------------------------------------------------------------------------------------ figure 1 / 2: FLOW
def fig_flow(F, ev, lg, fn, use_pct, R):
    font(lg)
    fig = plt.figure(figsize=(15, 6.3))
    nmax = max(F[d]["n"] for d in DRUGS)
    if use_pct:
        gap = 22.0
        top = 300 + 2 * gap
    else:
        gap = 0.07 * nmax
        top = nmax + 2 * gap
    fl = foot_lines(fig, foot_text(F, ev, lg, use_pct))
    # 10-24: no footer block any more (the methods text -> slide notes); the space goes to the flows
    if use_pct:
        bot = foot_h(fig) + 0.012                                # the % figure keeps the one-line star key (bar brackets)
        ax_top = 0.775
        pos = [[0.065, bot, 0.275, ax_top - bot], [0.37, bot, 0.275, ax_top - bot]]
    else:
        bot = 0.112                                              # the two McNemar lines under each flow
        ax_top = 0.815
        pos = [[0.045, bot, 0.43, ax_top - bot], [0.53, bot, 0.43, ax_top - bot]]
    axs = [fig.add_axes(p) for p in pos]
    for ax, drug in zip(axs, DRUGS):
        ax.set_xlim(-0.62 if not use_pct else -0.74, 1.52 if not use_pct else 1.32)
        ax.set_ylim(-0.02 * top, top * 1.14)
        clean(ax)
        M = F[drug]["M"]
        g = draw_flow(ax, M, (0.0, 1.0), top, gap, use_pct, lg, right_n=not use_pct)
        hdr = dict(ha="center", va="bottom", fontsize=13, fontweight="bold")
        ax.text(0.0, top + 0.035 * top, L(lg, "baseline", "기준"), color=CC["baseline"], **hdr)
        ax.text(1.0, top + 0.035 * top, L(lg, f"after {drug}", f"{drug} 후"), color=CC[drug], **hdr)
        ax.set_title(L(lg, f"{drug} day: {F[drug]['n']} neurons, {F[drug]['n_mice']} mice",
                       f"{drug} 날: 뉴런 {F[drug]['n']}, {F[drug]['n_mice']}마리"), color=CC[drug], fontsize=13.5,
                     fontweight="bold", pad=8)
        accent(ax, CC[drug])
        if not use_pct:
            w = within_mcnemar(M)
            lines = [L(lg, f"{c}: {w[c]['base']} → {w[c]['after']} neurons (lost {w[c]['lost']}, gained {w[c]['gained']}), "
                           f"McNemar {ptxt(w[c]['p'])}",
                       f"{c}: 뉴런 {w[c]['base']} → {w[c]['after']} (잃음 {w[c]['lost']}, 얻음 {w[c]['gained']}), "
                       f"McNemar {ptxt(w[c]['p'])}") for c in ("UP", "DOWN")]
            ax.text(0.5, -0.015, "\n".join(lines), transform=ax.transAxes, ha="center", va="top", fontsize=12, color="0.15",
                    linespacing=1.25)
            count_scale(ax, g, 0.0, lg)                          # on BOTH flows (same y scale), next to the NR block
        elif drug == "vehicle":
            nr_bot = g["tL"][1] - g["hL"][1]
            scale_bar(ax, -0.685, nr_bot + 0.25 * g["hL"][1], 50.0,
                      L(lg, "50 % of the\nbaseline class", "기준 유형의\n50 %"), fs=11.5)
    if use_pct:
        few_bar = any(R[nm][f"n_{sh}"] < 5 for nm, *_ in KEYTR for sh in ("veh", "sbi"))
        key_panel(fig, F, ev, lg, [0.75, bot + 0.135, 0.24, 0.665 - bot - 0.135], R)
        # two keys (10-24): the class colours above the flows; day colours, mice (and hatching) above the bar panel
        key_block(fig, class_handles(lg), L(lg, f"band label = % of the baseline class (neurons); {unlab_txt(lg)}",
                                            f"띠 숫자 = 기준 유형 중 % (뉴런 수); {unlab_txt(lg)}"),
                  y=0.935, xc=0.355, maxw=0.64)
        dh, mh = drug_handles(lg), mouse_handles(lg)
        col1, col2 = [dh[0], mh[0], mh[2]], [dh[1], mh[1], mh[3]]
        if few_bar:
            col1.append(hatch_handle(lg))
        key_block(fig, col1 + col2, "", y=0.935, xc=0.855, ncol=2, xmax=0.997, fs=11.5)
        st = L(lg, f"{ev}: the same flow, each baseline class scaled to 100 % - vehicle day vs SBI-553 day",
               f"{ev}: 같은 흐름, 기준 유형마다 100 % - vehicle 날 vs SBI-553 날")
    else:
        key_block(fig, class_handles(lg), L(lg, f"band colour = class at baseline; {unlab_txt(lg)}",
                                            f"띠 색 = 기준 세션의 유형; {unlab_txt(lg)}"), y=0.935)
        st = L(lg, f"{ev}: class of every neuron at baseline (left) and after the injection (right), the same neurons",
               f"{ev}: 모든 뉴런의 기준 (왼쪽) 과 주사 후 (오른쪽) 반응 유형, 같은 뉴런")
    fig.suptitle(st, fontsize=14.5, fontweight="bold", y=0.995, va="top")
    foot(fig, fl, lg, star=use_pct)                              # the count flows have no stars: no key line
    fig.savefig(fn, dpi=150)
    plt.close(fig)


def foot_text(F, ev, lg, use_pct):
    nv, ns = mouse_n_txt(F, "vehicle"), mouse_n_txt(F, "SBI-553")
    if lg == "kr":
        s = (f"각 날의 모든 뉴런 (vehicle 날 {F['vehicle']['n']} = {nv}; SBI-553 날 {F['SBI-553']['n']} = {ns}), 같은 날의 기준 세션과 "
             f"주사 세션은 같은 뉴런. 유형 = 최적 창 ({wtxt(ev, lg)}) 반응의 블록 내 시간 이동 검정, 세션마다 BH q <= 0.20. ")
        if use_pct:
            s += (f"띠 = 기준 유형 중 %, (괄호) = 뉴런 수 (>= {BAND_MIN}인 띠); 오른쪽 열 높이 = 들어온 % 의 합 (뉴런 수 아님); 흐린 블록 / 빗금 막대 = "
                  "기준 유형 뉴런 < 5 (% 불안정). 오른쪽 패널: 막대 = 뉴런 합산, 막대 아래 n = 기준 유형 뉴런 수, 표식 = 마우스 "
                  f"(그 기준 유형 뉴런 >= {MOUSE_MIN}), 괄호 = vehicle 날 vs SBI-553 날, Fisher exact (뉴런 합산 = 탐색적). ")
        else:
            s += (f"띠 = 기준 유형 (왼쪽) 에서 주사 후 유형 (오른쪽) 으로 간 뉴런, 기준 유형 색; 숫자 = 뉴런 수 (>= {BAND_MIN}인 띠). McNemar = 같은 "
                  "뉴런, 기준 vs 후 (exact). ")
        return s + KEY_KR + ". " + DRUG_ORDER_KR
    s = (f"Every neuron of each day (vehicle day {F['vehicle']['n']} = {nv}; SBI-553 day {F['SBI-553']['n']} = {ns}); within a day "
         f"the baseline and the injection session are the same neurons. Class = within-block time-shift test of the response in "
         f"the best window ({wtxt(ev, lg)}), BH q <= 0.20 per session. ")
    if use_pct:
        s += (f"Band = % of its baseline class, (n) = neurons (bands >= {BAND_MIN}); right-column heights = sum of the incoming % (not neurons); "
              "faint block / hatched bar = fewer than 5 neurons in that baseline class (% unreliable). "
              f"Right panel: bars = neurons pooled, n under each bar = neurons in that baseline class, markers = mice (>= {MOUSE_MIN} neurons "
              "in that baseline class); bracket = vehicle day vs SBI-553 day, Fisher exact (neurons pooled = exploratory). ")
    else:
        s += ("Band = neurons going from their baseline class (left) to their class after the injection (right), coloured by the "
              f"baseline class; numbers = neurons (bands >= {BAND_MIN}). McNemar = the same neurons, baseline vs after (exact). ")
    return s + PS.KEY + ". " + DRUG_ORDER_EN


def key_panel(fig, F, ev, lg, pos, R):
    ax = fig.add_axes(pos)
    ymax = 0.0
    tops_ = []
    for t, (nm, i, j, lab) in enumerate(KEYTR):
        r = R[nm]
        for dx, drug, sh in ((-0.21, "vehicle", "veh"), (0.21, "SBI-553", "sbi")):
            v = r[f"pct_{sh}"]
            few = r[f"n_{sh}"] < 5
            ax.bar(t + dx, 0 if not np.isfinite(v) else v, width=0.4, color=CC[drug], alpha=0.45 if few else 0.9,
                   hatch="//" if few else None, edgecolor="white" if not few else CC[drug], lw=0.5, zorder=2)
            hi = v if np.isfinite(v) else 0
            for k, m in enumerate(MO):
                mm = r[f"mice_{sh}"].get(m)
                if mm and mm["n"] >= MOUSE_MIN:
                    ax.scatter(t + dx + (k - 1.5) * 0.08, mm["pct"], marker=MK[m], color=MC[m], s=72, edgecolor="white",
                               lw=0.8, zorder=4, clip_on=False)            # 10-24: bigger; markers at 0 stay visible
                    hi = max(hi, mm["pct"])
            ymax = max(ymax, hi)
            tops_.append((t, hi))
    ytop = max(ymax, 10) * 1.0
    ax.set_ylim(0, ytop * 1.22 + 6)
    ax.set_yticks([t for t in range(0, 101, 20) if t <= ytop * 1.22 + 6])
    for t, (nm, i, j, lab) in enumerate(KEYTR):
        hi = max(h for tt, h in tops_ if tt == t)
        PS.bracket(ax, t - 0.21, t + 0.21, hi + 0.05 * ytop + 2, R[nm]["p_fisher"], "k", fs=13, inset=0.0)
    ax.set_xticks(range(len(KEYTR)), [lab[1 if lg == "kr" else 0] for _, _, _, lab in KEYTR], fontsize=12)
    n_under(ax, [(t - 0.21, R[nm]["n_veh"], "vehicle") for t, (nm, *_) in enumerate(KEYTR)]
            + [(t + 0.21, R[nm]["n_sbi"], "SBI-553") for t, (nm, *_) in enumerate(KEYTR)])
    ax.set_xlim(-0.55, len(KEYTR) - 0.45)
    ax.set_ylabel(L(lg, "% of the baseline class", "기준 유형 중 %"))
    ax.set_title(L(lg, "Key transitions:\nvehicle vs SBI-553 (Fisher)", "주요 이동:\nvehicle vs SBI-553 (Fisher)"),
                 fontsize=13, fontweight="bold", pad=6)


# ------------------------------------------------------------------------------------------------ figure 3: summary
SUMM = [("stay_resp", ("stay\nresponsive", "반응\n유지")), ("new_up", ("NR →\nUP", "NR →\nUP")),
        ("new_down", ("NR →\nDOWN", "NR →\nDOWN"))]
SUMM_TXT = {"stay_resp": ("stay responsive", "반응 유지"), "new_up": ("newly UP (NR -> UP)", "새 UP (NR -> UP)"),
            "new_down": ("newly DOWN (NR -> DOWN)", "새 DOWN (NR -> DOWN)")}


def fig_summary(FA, evs, lg, fn, fam):
    font(lg)
    fig = plt.figure(figsize=(15, 6.3))
    txt = summary_foot(FA, evs, lg)
    fl = foot_lines(fig, txt)
    fb = foot_h(fig)                                             # the one-line star key at the very bottom
    # 10-24: no footer block and no in-figure title (the slide title says it; it touched the slide key line) -> taller panels
    few_any = any(FA[ev]["S"][nm][f"n_{sh}"] < 5 for ev in evs for nm, _ in SUMM for sh in ("veh", "sbi"))
    h = drug_handles(lg) + mouse_handles(lg) + ([hatch_handle(lg)] if few_any else [])
    leg_bot = key_block(fig, h, "", y=0.985, ncol=len(h) if not few_any else 4, fs=12)
    top = leg_bot - 0.095                                        # room for the two-line panel titles
    defs_y = fb + 0.006                                          # the definitions line just above the star key
    bot = defs_y + 0.045 + 0.125                                 # + n row and two-line group labels under the bars
    n = len(evs)
    left, right, wsp = 0.065, 0.99, 0.025
    w = (right - left - wsp * (n - 1)) / n
    allv = []
    for ev in evs:
        for nm, _ in SUMM:
            r = FA[ev]["S"][nm]
            allv += [r["pct_veh"], r["pct_sbi"]]
            for sh in ("veh", "sbi"):
                allv += [mm["pct"] for mm in r[f"mice_{sh}"].values() if mm["n"] >= MOUSE_MIN]
    allv = [v for v in allv if np.isfinite(v)]
    ytop = max(allv) if allv else 100
    ylim = (0, ytop * 1.17 + 4)
    axs = []
    for k, ev in enumerate(evs):
        ax = fig.add_axes([left + k * (w + wsp), bot, w, top - bot])
        axs.append(ax)
        ax.set_ylim(*ylim)
        ax.set_yticks([t for t in range(0, 101, 20) if t <= ylim[1]])
        F = FA[ev]["F"]
        for g, (nm, lab) in enumerate(SUMM):
            r = FA[ev]["S"][nm]
            hi = 0.0
            for dx, drug, sh in ((-0.2, "vehicle", "veh"), (0.2, "SBI-553", "sbi")):
                v = r[f"pct_{sh}"]
                few = r[f"n_{sh}"] < 5
                ax.bar(g + dx, 0 if not np.isfinite(v) else v, width=0.38, color=CC[drug], alpha=0.45 if few else 0.9,
                       hatch="//" if few else None, edgecolor="white" if not few else CC[drug], lw=0.5, zorder=2)
                hi = max(hi, v if np.isfinite(v) else 0)
                for q, m in enumerate(MO):
                    mm = r[f"mice_{sh}"].get(m)
                    if mm and mm["n"] >= MOUSE_MIN:
                        ax.scatter(g + dx + (q - 1.5) * 0.08, mm["pct"], marker=MK[m], color=MC[m], s=64, edgecolor="white",
                                   lw=0.8, zorder=4, clip_on=False)      # 10-24: bigger; markers at 0 stay visible
                        hi = max(hi, mm["pct"])
            PS.bracket(ax, g - 0.2, g + 0.2, hi + 0.04 * ytop + 1.5, r["p_fisher"], "k", fs=13, inset=0.0)
        ax.set_xticks(range(len(SUMM)), [lab[1 if lg == "kr" else 0] for _, lab in SUMM], fontsize=12)
        n_under(ax, [(g - 0.2, FA[ev]["S"][nm]["n_veh"], "vehicle") for g, (nm, _) in enumerate(SUMM)]
                + [(g + 0.2, FA[ev]["S"][nm]["n_sbi"], "SBI-553") for g, (nm, _) in enumerate(SUMM)])
        ax.set_xlim(-0.6, len(SUMM) - 0.4)
        ax.set_title(f"{ev}\n" + L(lg, f"veh {F['vehicle']['n']} | SBI {F['SBI-553']['n']} neurons",
                                   f"뉴런 veh {F['vehicle']['n']} | SBI {F['SBI-553']['n']}"), fontsize=13, pad=5)
        if k == 0:
            ax.set_ylabel(L(lg, "% of the neurons in the baseline class", "기준 유형 뉴런 중 %"))
        else:
            ax.set_yticklabels([])
        ax.grid(axis="y", color="0.92", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
    # definitions just under the panels (n and tests are in the notes / on the slide bottom line)
    fig.text(0.5, defs_y, L(lg, "stay responsive = UP or DOWN at baseline and the SAME class after; NR → UP / NR → DOWN = % of "
                                "the baseline non-responsive neurons that became UP / DOWN",
                            "반응 유지 = 기준에서 UP 또는 DOWN, 주사 후 같은 유형; NR → UP / NR → DOWN = 기준 무반응 뉴런 중 "
                            "UP / DOWN이 된 %"), ha="center", va="bottom", fontsize=12, color="0.15")
    foot(fig, fl, lg)
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    return axs


def summary_foot(FA, evs, lg):
    ev0 = evs[0]
    F = FA[ev0]["F"]
    win = "; ".join(sorted({wtxt(e, lg) for e in evs}))
    if lg == "kr":
        return (f"각 날의 모든 뉴런 (vehicle 날 {F['vehicle']['n']}, SBI-553 날 {F['SBI-553']['n']}), 같은 날 기준 세션 -> 주사 세션. "
                f"유형 = 블록 내 시간 이동 검정, 세션마다 BH q <= 0.20, 최적 창 ({win}). 막대 = 뉴런 합산, 막대 아래 n = 분모 (기준 유형 "
                f"뉴런 수), 빗금 = 분모 < 5; 표식 = 마우스 (분모 뉴런 >= {MOUSE_MIN}). 괄호 = vehicle 날 vs SBI-553 날, Fisher exact (뉴런 합산 = 탐색적; 4마리로는 "
                f"마우스 간 검정이 p < 0.05에 못 미침). " + KEY_KR + ". " + DRUG_ORDER_KR)
    return (f"Every neuron of each day (vehicle day {F['vehicle']['n']}, SBI-553 day {F['SBI-553']['n']}), baseline session -> "
            f"injection session of the same day. Class = within-block time-shift test, BH q <= 0.20 per session, best window "
            f"({win}). Bars = neurons pooled, n under each bar = the denominator (neurons in the baseline class), hatched = "
            f"denominator < 5; markers = mice (>= {MOUSE_MIN} neurons in the denominator). Bracket = vehicle day vs SBI-553 day, Fisher exact (neurons pooled = exploratory; with 4 mice no test "
            f"across mice can reach p < 0.05). " + PS.KEY + ". " + DRUG_ORDER_EN)


# ------------------------------------------------------------------------------------------------ figure 4: tracked
def load_pairs():
    return {m: pd.read_csv(os.path.join(CD, f"{m}_pairs.csv")) for m in MO}


def tracked_table(T, P, ev):
    R = T[T.event == ev].set_index(["mouse", "day", "phase", "cell"]).c
    rows = []
    for m in MO:
        sd, vd = SBI_DAY[m], 3 - SBI_DAY[m]
        for p in P[m].itertuples():
            ids = {1: int(p.day1_id), 2: int(p.day2_id)}
            r = dict(event=ev, mouse=m, track_class=p.track_class, day1_id=ids[1], day2_id=ids[2])
            for c, (d, ph) in zip(("vb", "vi", "sb", "si"), ((vd, "pre"), (vd, "post"), (sd, "pre"), (sd, "post"))):
                r[c] = R.get((m, d, ph, ids[d]), None)
            rows.append(r)
    return pd.DataFrame(rows)


def tracked_stats(X):
    up_v, up_s = X.vi == "UP", X.si == "UP"
    dn_v, dn_s = X.vi == "DOWN", X.si == "DOWN"
    r = dict(n=len(X), per_mouse={m: int((X.mouse == m).sum()) for m in MO},
             M_veh=mat(X, "vb", "vi").tolist(), M_sbi=mat(X, "sb", "si").tolist(), M_after=mat(X, "vi", "si").tolist(),
             up_after_veh=int(up_v.sum()), up_after_sbi=int(up_s.sum()),
             up_b=int((up_v & ~up_s).sum()), up_c=int((~up_v & up_s).sum()),
             down_after_veh=int(dn_v.sum()), down_after_sbi=int(dn_s.sum()),
             down_b=int((dn_v & ~dn_s).sum()), down_c=int((~dn_v & dn_s).sum()))
    r["p_up"] = mcnemar(r["up_b"], r["up_c"])
    r["p_down"] = mcnemar(r["down_b"], r["down_c"])
    ub = (X.vb == "UP"), (X.sb == "UP")
    r["up_base_veh"], r["up_base_sbi"] = int(ub[0].sum()), int(ub[1].sum())
    r["p_up_baselines"] = mcnemar(int((ub[0] & ~ub[1]).sum()), int((~ub[0] & ub[1]).sum()))
    r["within_veh"] = within_mcnemar(mat(X, "vb", "vi"))
    r["within_sbi"] = within_mcnemar(mat(X, "sb", "si"))
    r["mice_up"] = {m: dict(veh=int(((X.mouse == m) & up_v).sum()), sbi=int(((X.mouse == m) & up_s).sum()),
                            n=int((X.mouse == m).sum())) for m in MO}
    return r


def fig_tracked(X, ev, lg, fn, st):
    font(lg)
    fig = plt.figure(figsize=(15, 6.3))
    n = len(X)
    gap = 0.08 * n
    top = n + 2 * gap
    fl = foot_lines(fig, tracked_foot(X, ev, lg, st))
    fb = foot_h(fig)                                             # 10-24: only the one-line star key at the bottom
    bot = fb + 0.092                                             # the two text lines under each panel (10-25: was three)
    # the class key + the band-colour rule of each panel (B is coloured by the class after vehicle), as text, not a white box
    leg_bot = key_block(fig, class_handles(lg),
                        L(lg, f"band colour: A = class at baseline, B = class after vehicle; {unlab_txt(lg)}",
                          f"띠 색: A = 기준 세션의 유형, B = vehicle 후 유형; {unlab_txt(lg)}"), y=0.935)
    ax_top = min(0.79, leg_bot - 0.092)                          # room for the two-line title of panel B
    # layout in inches so the block labels fit: 1 x unit = 1.5 in in both panels
    U = 1.5
    axA = fig.add_axes([0.3 / 15, bot, 8.85 / 15, ax_top - bot])
    axB = fig.add_axes([9.55 / 15, bot, 5.3 / 15, ax_top - bot])
    ylim = (-0.02 * top, top * 1.3)
    for ax in (axA, axB):
        ax.set_ylim(*ylim)
        clean(ax)
    axA.set_xlim(-0.95, -0.95 + 8.85 / U)
    axB.set_xlim(-0.95, -0.95 + 5.3 / U)
    xs1, xs2, xb = (0.0, 1.0), (3.05, 4.05), (0.0, 1.4)          # 10-24: more room for the two label columns
    sepx = (xs1[1] + xs2[0]) / 2
    Mv, Ms, Ma = mat(X, "vb", "vi"), mat(X, "sb", "si"), mat(X, "vi", "si")
    gA = draw_flow(axA, Mv, xs1, top, gap, False, lg, lab_right=True, fs=FS - 0.5)
    draw_flow(axA, Ms, xs2, top, gap, False, lg, lab_left=True, fs=FS - 0.5)
    axA.plot([sepx, sepx], [0, top], color="0.55", lw=1.2, ls=(0, (4, 3)), zorder=1)
    axA.text(sepx, top + 0.215 * top, L(lg, "| other day, the same neurons |", "| 다른 날, 같은 뉴런 |"), ha="center", va="center",
             fontsize=12, color="0.3")
    hdr = dict(ha="center", va="bottom", fontsize=12.5, fontweight="bold", linespacing=1.05)
    for x, t, c in ((xs1[0], L(lg, "vehicle day\nbaseline", "vehicle 날\n기준"), CC["baseline"]),
                    (xs1[1], L(lg, "after\nvehicle", "vehicle\n후"), CC["vehicle"]),
                    (xs2[0], L(lg, "SBI-553 day\nbaseline", "SBI-553 날\n기준"), CC["baseline"]),
                    (xs2[1], L(lg, "after\nSBI-553", "SBI-553\n후"), CC["SBI-553"])):
        axA.text(x, top + 0.03 * top, t, color=c, **hdr)
    axA.set_title(L(lg, f"A  Each day: baseline → after the injection ({n} tracked neurons, 4 mice)",
                    f"A  날마다: 기준 → 주사 후 (추적 뉴런 {n}, 4마리)"), fontsize=13, fontweight="bold", loc="left", pad=8)
    wv, ws = st["within_veh"], st["within_sbi"]
    for xs, w_, c in ((xs1, wv, "vehicle"), (xs2, ws, "SBI-553")):
        lines = [L(lg, f"{k}: {w_[k]['base']} → {w_[k]['after']}, McNemar {ptxt(w_[k]['p'])}",
                   f"{k}: {w_[k]['base']} → {w_[k]['after']}, McNemar {ptxt(w_[k]['p'])}") for k in ("UP", "DOWN")]
        axA.text(sum(xs) / 2, -0.045 * top, "\n".join(lines), ha="center", va="top", fontsize=12, color=CC[c], linespacing=1.25)
    count_scale(axA, gA, xs1[0], lg)                             # 10-24: bigger, next to the NR block, in both panels
    gB = draw_flow(axB, Ma, xb, top, gap, False, lg, fs=FS - 0.5)
    count_scale(axB, gB, xb[0], lg)
    for x, t, c in ((xb[0], L(lg, "after\nvehicle", "vehicle\n후"), CC["vehicle"]),
                    (xb[1], L(lg, "after\nSBI-553", "SBI-553\n후"), CC["SBI-553"])):
        axB.text(x, top + 0.03 * top, t, color=c, **hdr)
    nt0 = len(axB.texts)
    PS.bracket(axB, xb[0], xb[1], top + 0.19 * top, st["p_up"], "k", fs=12.5, inset=0.0,
               label=L(lg, "UP calls, McNemar:", "UP 판정, McNemar:"))
    for t in axB.texts[nt0:]:                                    # 10-25: 'ns' in the same size / weight as a star
        t.set_fontsize(12.5)
        t.set_fontweight("bold")
    axB.set_title(L(lg, "B  The same neurons:\nafter vehicle → after SBI-553", "B  같은 뉴런:\nvehicle 후 → SBI-553 후"),
                  fontsize=13, fontweight="bold", loc="left", pad=8)
    # 10-25: the discordant counts and the per-mouse counts go to the slide notes (tracked_foot), not the figure
    axB.text(0.55, -0.045 * top, L(lg, f"UP: {st['up_after_veh']} after vehicle vs {st['up_after_sbi']} after SBI-553\n"
                                     f"McNemar {ptxt(st['p_up'])}",
                                 f"UP: vehicle 후 {st['up_after_veh']} vs SBI-553 후 {st['up_after_sbi']}\n"
                                 f"McNemar {ptxt(st['p_up'])}"),
             ha="center", va="top", fontsize=12, color="0.15", linespacing=1.25)
    fig.suptitle(L(lg, f"{ev}: the {n} tracked neurons (the same cells on both days) through the four sessions",
                   f"{ev}: 추적 뉴런 {n} (두 날 같은 세포) 의 네 세션"), fontsize=14.5, fontweight="bold", y=0.995, va="top")
    foot(fig, fl, lg)
    fig.savefig(fn, dpi=150)
    plt.close(fig)


def tracked_foot(X, ev, lg, st):
    pm = ", ".join(f"{m} {st['per_mouse'][m]}" for m in MO)
    pmu = ", ".join(f"{m} {st['mice_up'][m]['veh']}->{st['mice_up'][m]['sbi']}" for m in MO)
    nA = int((X.track_class == "A").sum())
    if lg == "kr":
        return (f"B: UP vehicle 후만 {st['up_b']}, SBI-553 후만 {st['up_c']} (McNemar {ptxt(st['p_up'])}); 마우스별 UP "
                f"(veh -> SBI): {pmu}. "
                f"추적 뉴런 {len(X)} (crossday 쌍: A 같은 뉴런 {nA} + B 아마 같은 뉴런 {len(X) - nA}; {pm}), 네 세션 모두에 있음. "
                f"A: 날마다 기준 -> 주사 후 (띠 = 뉴런, 기준 유형 색; 두 날은 띠로 잇지 않음, 열은 같은 뉴런). B: 같은 뉴런의 vehicle 후 "
                f"유형 -> SBI-553 후 유형 (vehicle 후 유형 색); McNemar (exact) = 같은 뉴런의 UP 판정 쌍. 유형 = 시간 이동 검정, BH q <= "
                f"0.20, 최적 창 ({wtxt(ev, lg)}). " + KEY_KR + ". " + DRUG_ORDER_KR)
    return (f"B: UP only after vehicle {st['up_b']}, only after SBI-553 {st['up_c']} (McNemar {ptxt(st['p_up'])}); UP per mouse "
            f"(veh -> SBI): {pmu}. "
            f"{len(X)} tracked neurons (crossday pairs: A same neuron {nA} + B probably the same {len(X) - nA}; {pm}), each present "
            f"in all four sessions. A: each day baseline -> after the injection (band = neurons, coloured by the baseline class; the "
            f"two days are not linked by bands, the columns are the same neurons). B: class after vehicle -> class after SBI-553 of "
            f"the same neurons (coloured by the class after vehicle); McNemar (exact) on the paired UP calls. Class = time-shift "
            f"test, BH q <= 0.20, best window ({wtxt(ev, lg)}). " + PS.KEY + ". " + DRUG_ORDER_EN)


# ------------------------------------------------------------------------------------------------ summary.json texts
def jconv(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def clean_nan(o):
    if isinstance(o, dict):
        return {str(k): clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_nan(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean_nan(o.tolist())
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


LIMITS = {"key_en": 150, "key_kr": 150, "bottom_en": 230, "bottom_kr": 230}
WARN = []
CAVEATS = []


def entry(fname, numbers, title_en, title_kr, key_en, key_kr, bottom_en, bottom_kr, notes_en, notes_kr, priority,
          section="Q3 neurons"):
    e = dict(title_en=title_en, title_kr=title_kr, key_en=key_en, key_kr=key_kr, bottom_en=bottom_en, bottom_kr=bottom_kr,
             notes_en=notes_en, notes_kr=notes_kr, section=section, priority=priority, numbers=clean_nan(numbers))
    for k, lim in LIMITS.items():
        if len(e[k]) > lim:
            WARN.append(f"{fname} {k} {len(e[k])} > {lim}: {e[k]}")
    return e


def f0(v):
    return "n.a." if v is None or not np.isfinite(v) else f"{v:.0f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    T, J = load()
    P = load_pairs()
    evs_all = STIM + BEH + EXTRA
    evs, skipped = [], {}
    for ev in evs_all:
        ok = {d: int(((J.event == ev) & (J.drug == d)).sum()) for d in DRUGS}
        if min(ok.values()) > 0:
            evs.append(ev)
        else:
            skipped[ev] = {d: ok[d] for d in DRUGS}
    print("events:", evs, "| skipped:", skipped)
    FA, rows, trows = {}, [], []
    for ev in evs:
        F = flows(J, ev)
        R = {nm: compare(F, nm) for nm, *_ in KEYTR}
        S = {nm: compare(F, nm) for nm, _ in SUMM}
        FA[ev] = dict(F=F, R=R, S=S)
        for drug in DRUGS:
            for who, M in [("all", F[drug]["M"])] + list(F[drug]["mice"].items()):
                for i, ci in enumerate(CLS):
                    for j, cj in enumerate(CLS):
                        rows.append(dict(event=ev, drug=drug, mouse=who, base=ci, after=cj, n=int(M[i, j]),
                                         n_base_class=int(M[i].sum())))
        for nm, r in list(R.items()) + list(S.items()):
            trows.append(dict(event=ev, transition=nm, **{k: v for k, v in r.items() if not k.startswith("mice")}))
        for drug in DRUGS:
            w = within_mcnemar(F[drug]["M"])
            for c in ("UP", "DOWN"):
                trows.append(dict(event=ev, transition=f"within {drug} day {c} baseline vs after (McNemar)", **w[c]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "flow_counts.csv"), index=False)
    pd.DataFrame(trows).to_csv(os.path.join(OUT, "flow_tests.csv"), index=False)
    # tracked
    TR, XS = {}, []
    for ev in [e for e in evs if e in STIM + BEH]:
        X0 = tracked_table(T, P, ev)
        X = X0.dropna(subset=["vb", "vi", "sb", "si"])
        if len(X) < len(X0):
            print(f"tracked {ev}: {len(X0) - len(X)} neurons without a class in some session - left out")
        st = tracked_stats(X)
        stA = tracked_stats(X[X.track_class == "A"])
        TR[ev] = dict(X=X, st=st, stA=stA, n_missing=len(X0) - len(X))
        XS.append(X)
    pd.concat(XS).to_csv(os.path.join(OUT, "tracked_classes.csv"), index=False)
    # figures
    for lg in ("en", "kr"):
        sfx = "_kr" if lg == "kr" else ""
        for ev in evs:
            tg = tag_of(ev)
            fig_flow(FA[ev]["F"], ev, lg, os.path.join(OUT, f"FLOW_{tg}{sfx}.png"), False, FA[ev]["R"])
            fig_flow(FA[ev]["F"], ev, lg, os.path.join(OUT, f"FLOW_{tg}_pct{sfx}.png"), True, FA[ev]["R"])
        fig_summary(FA, [e for e in STIM if e in FA], lg, os.path.join(OUT, f"FLOW_summary_stim{sfx}.png"), "stim")
        fig_summary(FA, [e for e in BEH if e in FA], lg, os.path.join(OUT, f"FLOW_summary_beh{sfx}.png"), "beh")
        for ev, d in TR.items():
            fig_tracked(d["X"], ev, lg, os.path.join(OUT, f"FLOW_tracked_{tag_of(ev)}{sfx}.png"), d["st"])
    write_summary(FA, TR, evs, skipped)
    for w in WARN:
        print("WARNING", w)
    print("->", OUT)


def write_summary(FA, TR, evs, skipped):
    J_ = {}
    win_en = WW.en()
    win_kr = WW.kr()
    common_en = (f"Classes: within-block time-shift test, BH q <= 0.20 per session; best windows: {win_en}. Every neuron of each day; "
                 f"within a day the baseline and injection sessions are the same neurons. {DRUG_ORDER_EN} Neurons pooled over the 4 "
                 f"mice = exploratory; with 4 mice no test across mice can reach p < 0.05, per-mouse values are descriptive.")
    common_kr = (f"유형: 블록 내 시간 이동 검정, 세션마다 BH q <= 0.20; 최적 창: {win_kr}. 각 날의 모든 뉴런; 같은 날 기준과 주사 세션은 같은 "
                 f"뉴런. {DRUG_ORDER_KR} 4마리 뉴런 합산 = 탐색적; 4마리로는 마우스 간 검정이 p < 0.05에 못 미침, 마우스별 값은 기술적.")
    for ev in evs:
        F, R, S = FA[ev]["F"], FA[ev]["R"], FA[ev]["S"]
        tg = tag_of(ev)
        Mv, Ms = F["vehicle"]["M"], F["SBI-553"]["M"]
        wv, ws = within_mcnemar(Mv), within_mcnemar(Ms)
        prio = "main" if ev in MAIN_EV else "backup"
        nums = dict(event=ev, window=list(WW.of(ev)), classes=CLS,
                    vehicle=dict(M=Mv, n=F["vehicle"]["n"], per_mouse_n=F["vehicle"]["per_mouse_n"],
                                 mice={m: v for m, v in F["vehicle"]["mice"].items()}, within_mcnemar=wv),
                    sbi553=dict(M=Ms, n=F["SBI-553"]["n"], per_mouse_n=F["SBI-553"]["per_mouse_n"],
                                mice={m: v for m, v in F["SBI-553"]["mice"].items()}, within_mcnemar=ws))
        nu_v, nu_s = int(Mv[0].sum()), int(Ms[0].sum())
        key_en = (f"{ev}: {Mv[0, 0]}/{nu_v} baseline-UP neurons stayed UP after vehicle vs {Ms[0, 0]}/{nu_s} after SBI-553; "
                  f"NR -> UP {Mv[1, 0]} vs {Ms[1, 0]}")
        key_kr = (f"{ev}: 기준 UP 뉴런 중 UP 유지 vehicle 후 {Mv[0, 0]}/{nu_v} vs SBI-553 후 {Ms[0, 0]}/{nu_s}; "
                  f"NR -> UP {Mv[1, 0]} vs {Ms[1, 0]}")
        bot_en = (f"Every neuron, baseline -> injection session of the same day (vehicle day {F['vehicle']['n']}, SBI-553 day "
                  f"{F['SBI-553']['n']} neurons, 4 mice); time-shift test, q <= 0.20; McNemar within day.")
        bot_kr = (f"모든 뉴런, 같은 날 기준 -> 주사 세션 (vehicle 날 뉴런 {F['vehicle']['n']}, SBI-553 날 {F['SBI-553']['n']}, 4마리); "
                  f"시간 이동 검정, q <= 0.20; 날 안에서 McNemar.")
        notes_en = (f"Within-day McNemar (same neurons, baseline vs after): vehicle day UP {wv['UP']['base']} -> {wv['UP']['after']} "
                    f"({ptxt(wv['UP']['p'])}), DOWN {wv['DOWN']['base']} -> {wv['DOWN']['after']} ({ptxt(wv['DOWN']['p'])}); SBI-553 "
                    f"day UP {ws['UP']['base']} -> {ws['UP']['after']} ({ptxt(ws['UP']['p'])}), DOWN {ws['DOWN']['base']} -> "
                    f"{ws['DOWN']['after']} ({ptxt(ws['DOWN']['p'])}). Band labels shown for bands >= {BAND_MIN} neurons. " + common_en)
        notes_kr = (f"날 안 McNemar (같은 뉴런, 기준 vs 후): vehicle 날 UP {wv['UP']['base']} -> {wv['UP']['after']} "
                    f"({ptxt(wv['UP']['p'])}), DOWN {wv['DOWN']['base']} -> {wv['DOWN']['after']} ({ptxt(wv['DOWN']['p'])}); SBI-553 날 "
                    f"UP {ws['UP']['base']} -> {ws['UP']['after']} ({ptxt(ws['UP']['p'])}), DOWN {ws['DOWN']['base']} -> "
                    f"{ws['DOWN']['after']} ({ptxt(ws['DOWN']['p'])}). 띠 숫자는 뉴런 >= {BAND_MIN}인 띠만. " + common_kr)
        J_[f"FLOW_{tg}.png"] = entry(f"FLOW_{tg}.png", nums, f"{ev}: where each neuron's class went after the injection",
                                     f"{ev}: 주사 후 각 뉴런의 반응 유형은 어디로 갔나", key_en, key_kr, bot_en, bot_kr, notes_en,
                                     notes_kr, prio)
        su, nu, un, nd = R["stay_up"], R["nr_up"], R["up_nr"], R["nr_down"]
        key_en = (f"{ev}: stay UP {f0(su['pct_veh'])}% after vehicle vs {f0(su['pct_sbi'])}% after SBI-553 ({ptxt(su['p_fisher'])}); "
                  f"NR -> UP {f0(nu['pct_veh'])}% vs {f0(nu['pct_sbi'])}% ({ptxt(nu['p_fisher'])})")
        key_kr = (f"{ev}: UP 유지 vehicle 후 {f0(su['pct_veh'])}% vs SBI-553 후 {f0(su['pct_sbi'])}% ({ptxt(su['p_fisher'])}); "
                  f"NR -> UP {f0(nu['pct_veh'])}% vs {f0(nu['pct_sbi'])}% ({ptxt(nu['p_fisher'])})")
        bot_en = (f"Each baseline class = 100 %; every neuron (vehicle day {F['vehicle']['n']}, SBI-553 day {F['SBI-553']['n']}, "
                  f"4 mice); key transitions vehicle vs SBI-553 day, Fisher exact (neurons pooled), markers = mice.")
        bot_kr = (f"기준 유형마다 100 %; 모든 뉴런 (vehicle 날 {F['vehicle']['n']}, SBI-553 날 {F['SBI-553']['n']}, 4마리); 주요 이동 "
                  f"vehicle vs SBI-553 날, Fisher exact (뉴런 합산), 표식 = 마우스.")
        tr_en = "; ".join(f"{KEYTR_TXT[k][0]} {R[k]['k_veh']}/{R[k]['n_veh']} vs {R[k]['k_sbi']}/{R[k]['n_sbi']} "
                          f"({ptxt(R[k]['p_fisher'])})" for k, *_ in KEYTR)
        tr_kr = "; ".join(f"{KEYTR_TXT[k][1]} {R[k]['k_veh']}/{R[k]['n_veh']} vs {R[k]['k_sbi']}/{R[k]['n_sbi']} "
                          f"({ptxt(R[k]['p_fisher'])})" for k, *_ in KEYTR)
        few = [f"{c} ({d})" for d in DRUGS for i, c in enumerate(CLS) if 0 < F[d]["M"][i].sum() < 5]
        Dm = dominance(F, [k for k, *_ in KEYTR], ev)
        CAVEATS.extend(Dm)

        def _pm(k):
            g = lambda d_, m: f"{d_[m]['k']}/{d_[m]['n']}" if m in d_ else "-"
            return ", ".join(f"{m} {g(R[k]['mice_veh'], m)} | {g(R[k]['mice_sbi'], m)}" for m in MO)
        tr_en += (f". Per mouse (vehicle day k/n | SBI-553 day k/n): stay UP {_pm('stay_up')}; NR -> UP {_pm('nr_up')}")
        tr_kr += (f". 마우스별 (vehicle 날 k/n | SBI-553 날 k/n): UP 유지 {_pm('stay_up')}; NR -> UP {_pm('nr_up')}")
        notes_en = (f"Vehicle k/n vs SBI-553 k/n (Fisher exact): {tr_en}. Right-column heights = sum of the incoming %, not neurons. "
                    + (f"Baseline classes with < 5 neurons (shown faint, % unreliable): {', '.join(few)}. " if few else "")
                    + dom_txt(Dm, "en", KEYTR_TXT) + common_en)
        notes_kr = (f"vehicle k/n vs SBI-553 k/n (Fisher exact): {tr_kr}. 오른쪽 열 높이 = 들어온 % 의 합 (뉴런 수 아님). "
                    + (f"뉴런 < 5인 기준 유형 (흐리게, % 불안정): {', '.join(few)}. " if few else "")
                    + dom_txt(Dm, "kr", KEYTR_TXT) + common_kr)
        J_[f"FLOW_{tg}_pct.png"] = entry(f"FLOW_{tg}_pct.png", dict(event=ev, transitions=R), f"{ev}: the flow as % of each "
                                         f"baseline class, vehicle vs SBI-553", f"{ev}: 기준 유형별 % 흐름, vehicle vs SBI-553", key_en,
                                         key_kr, bot_en, bot_kr, notes_en, notes_kr, prio)
    for fam, lst in (("stim", STIM), ("beh", BEH)):
        es = [e for e in lst if e in FA]
        if not es:
            continue
        nums = {e: FA[e]["S"] for e in es}
        sr = [(e, FA[e]["S"]["stay_resp"]) for e in es]
        lower = [e for e, r in sr if np.isfinite(r["pct_sbi"]) and np.isfinite(r["pct_veh"]) and r["pct_sbi"] < r["pct_veh"]]
        sig = [(e, nm, FA[e]["S"][nm]) for e in es for nm, _ in SUMM if np.isfinite(FA[e]["S"][nm]["p_fisher"])
               and FA[e]["S"][nm]["p_fisher"] < 0.05]
        fam_en = "stimuli" if fam == "stim" else "behaviours"
        fam_kr = "자극" if fam == "stim" else "행동"
        # key: the p < 0.05 comparisons (vehicle % -> SBI-553 %), the main events first, as many as fit in 150 characters
        sig.sort(key=lambda t: (t[0] not in MAIN_EV, t[2]["p_fisher"]))
        short = {"stay_resp": ("stay responsive", "반응 유지"), "new_up": ("NR->UP", "NR->UP"), "new_down": ("NR->DOWN", "NR->DOWN")}
        key_en, key_kr = "Vehicle -> SBI-553 day, Fisher p < 0.05: ", "vehicle -> SBI-553 날, Fisher p < 0.05: "
        if not sig:
            key_en = f"No vehicle vs SBI-553 difference with Fisher p < 0.05 in the {len(es)} {fam_en}"
            key_kr = f"{fam_kr} {len(es)}개 모두 vehicle vs SBI-553 Fisher p < 0.05 차이 없음"
        else:
            pe, pk = [], []
            for e, nm, r in sig:
                te = f"{e} {short[nm][0]} {f0(r['pct_veh'])}% -> {f0(r['pct_sbi'])}%"
                tk = f"{e} {short[nm][1]} {f0(r['pct_veh'])}% -> {f0(r['pct_sbi'])}%"
                if len(key_en + "; ".join(pe + [te])) > 150 or len(key_kr + "; ".join(pk + [tk])) > 150:
                    break
                pe.append(te)
                pk.append(tk)
            key_en += "; ".join(pe)
            key_kr += "; ".join(pk)
        nlow = f"stay responsive lower after SBI-553 in {len(lower)}/{len(es)} {fam_en}"
        F0 = FA[es[0]]["F"]
        bot_en = (f"Every neuron, baseline -> injection of the same day (vehicle day {F0['vehicle']['n']}, SBI-553 day "
                  f"{F0['SBI-553']['n']}, 4 mice); bars = pooled, markers = mice; Fisher exact vehicle vs SBI-553 (exploratory).")
        bot_kr = (f"모든 뉴런, 같은 날 기준 -> 주사 (vehicle 날 {F0['vehicle']['n']}, SBI-553 날 {F0['SBI-553']['n']}, 4마리); 막대 = "
                  f"합산, 표식 = 마우스; Fisher exact vehicle vs SBI-553 (탐색적).")
        det_en = " | ".join(f"{e}: " + "; ".join(f"{SUMM_TXT[nm][0]} {FA[e]['S'][nm]['k_veh']}/{FA[e]['S'][nm]['n_veh']} vs "
                                                  f"{FA[e]['S'][nm]['k_sbi']}/{FA[e]['S'][nm]['n_sbi']} ({ptxt(FA[e]['S'][nm]['p_fisher'])})"
                                                  for nm, _ in SUMM) for e in es)
        det_kr = " | ".join(f"{e}: " + "; ".join(f"{SUMM_TXT[nm][1]} {FA[e]['S'][nm]['k_veh']}/{FA[e]['S'][nm]['n_veh']} vs "
                                                  f"{FA[e]['S'][nm]['k_sbi']}/{FA[e]['S'][nm]['n_sbi']} ({ptxt(FA[e]['S'][nm]['p_fisher'])})"
                                                  for nm, _ in SUMM) for e in es)
        Dm = [d for e in es for d in dominance(FA[e]["F"], [nm for nm, _ in SUMM], e)]
        CAVEATS.extend(Dm)
        notes_en = ("stay responsive = UP or DOWN at baseline and the same class after the injection; newly UP / DOWN = % of the "
                    f"baseline NR neurons that became UP / DOWN ({nlow}). Vehicle k/n vs SBI-553 k/n: " + det_en + ". Markers: mice with >= "
                    f"{MOUSE_MIN} neurons in the denominator. " + dom_txt(Dm, "en", SUMM_TXT) + common_en)
        notes_kr = ("반응 유지 = 기준에서 UP 또는 DOWN이고 주사 후 같은 유형; 새 UP / DOWN = 기준 NR 뉴런 중 UP / DOWN이 된 %. vehicle k/n vs "
                    "SBI-553 k/n: " + det_kr + f". 표식: 분모 뉴런 >= {MOUSE_MIN}인 마우스. " + dom_txt(Dm, "kr", SUMM_TXT) + common_kr)
        J_[f"FLOW_summary_{fam}.png"] = entry(
            f"FLOW_summary_{fam}.png", nums,
            ("Stimuli" if fam == "stim" else "Behaviours") + ": who stays responsive and who is recruited after the injection",
            fam_kr + ": 주사 후 반응 유지 뉴런과 새로 반응하는 뉴런", key_en, key_kr, bot_en, bot_kr, notes_en, notes_kr, "main")
    for ev, d in TR.items():
        st, stA = d["st"], d["stA"]
        tg = tag_of(ev)
        key_en = (f"{ev}, the same {st['n']} neurons: UP {st['up_after_veh']} after vehicle vs {st['up_after_sbi']} after SBI-553 "
                  f"(McNemar {ptxt(st['p_up'])})")
        key_kr = (f"{ev}, 같은 뉴런 {st['n']}: UP vehicle 후 {st['up_after_veh']} vs SBI-553 후 {st['up_after_sbi']} "
                  f"(McNemar {ptxt(st['p_up'])})")
        bot_en = (f"Tracked neurons only ({st['n']}, 4 mice), classes in all four sessions; each day baseline -> after, and after "
                  f"vehicle -> after SBI-553 of the same neurons; McNemar on the paired UP calls.")
        bot_kr = (f"추적 뉴런만 ({st['n']}, 4마리), 네 세션의 유형; 날마다 기준 -> 후, 같은 뉴런의 vehicle 후 -> SBI-553 후; "
                  f"UP 판정 쌍 McNemar.")
        mu = st["mice_up"]
        pm = ", ".join(f"{m} {mu[m]['veh']} -> {mu[m]['sbi']}" for m in MO)
        # day check: per mouse, more UP after the Day-2 injection than after the Day-1 injection?
        d2 = [m for m in MO if (mu[m]["sbi"] if SBI_DAY[m] == 2 else mu[m]["veh"]) > (mu[m]["veh"] if SBI_DAY[m] == 2 else mu[m]["sbi"])]
        lo_s = [m for m in MO if mu[m]["sbi"] < mu[m]["veh"]]
        st["mice_fewer_up_after_sbi"], st["mice_more_up_after_day2_injection"] = lo_s, d2
        if len(d2) in (0, 4):
            why_en = "the difference follows the imaging day in every mouse, so it cannot be told apart from a day effect"
            why_kr = "차이가 모든 마우스에서 촬영 날을 따름, 날 효과와 구분 불가"
        elif len(lo_s) in (0, 4):
            why_en = "the same direction in all 4 mice, i.e. in both drug orders"
            why_kr = "4마리 모두 같은 방향, 즉 두 약물 순서 모두"
        else:
            why_en, why_kr = "mixed across mice", "마우스마다 다름"
        st["day_check_en"], st["day_check_kr"] = why_en, why_kr
        notes_en = (f"UP only after vehicle {st['up_b']}, only after SBI-553 {st['up_c']}. UP per mouse, after vehicle -> after SBI-553: "
                    f"{pm} (fewer after SBI-553 in {len(lo_s)}/4 mice; more after the Day-2 than the Day-1 injection in {len(d2)}/4 "
                    f"mice: {why_en}). DOWN after vehicle {st['down_after_veh']} vs "
                    f"after SBI-553 {st['down_after_sbi']} (McNemar {ptxt(st['p_down'])}); baselines UP vehicle day {st['up_base_veh']} "
                    f"vs SBI-553 day {st['up_base_sbi']} (McNemar {ptxt(st['p_up_baselines'])}). Class A only ({stA['n']} neurons): UP "
                    f"{stA['up_after_veh']} vs {stA['up_after_sbi']} (McNemar {ptxt(stA['p_up'])}). Neurons pooled = exploratory. "
                    + DRUG_ORDER_EN)
        notes_kr = (f"vehicle 후만 UP {st['up_b']}, SBI-553 후만 UP {st['up_c']}. 마우스별 UP, vehicle 후 -> SBI-553 후: {pm} "
                    f"(SBI-553 후 적음 {len(lo_s)}/4마리; Day 2 주사 후가 Day 1 주사 후보다 많음 {len(d2)}/4마리: {why_kr}). "
                    f"DOWN vehicle 후 {st['down_after_veh']} vs SBI-553 후 "
                    f"{st['down_after_sbi']} (McNemar {ptxt(st['p_down'])}); 기준 UP vehicle 날 {st['up_base_veh']} vs SBI-553 날 "
                    f"{st['up_base_sbi']} (McNemar {ptxt(st['p_up_baselines'])}). A 등급만 ({stA['n']}): UP {stA['up_after_veh']} vs "
                    f"{stA['up_after_sbi']} (McNemar {ptxt(stA['p_up'])}). 뉴런 합산 = 탐색적. " + DRUG_ORDER_KR)
        J_[f"FLOW_tracked_{tg}.png"] = entry(f"FLOW_tracked_{tg}.png", dict(event=ev, all=st, class_A_only=stA,
                                                                             n_missing=d["n_missing"]),
                                             f"{ev}: the same tracked neurons, class after vehicle vs after SBI-553",
                                             f"{ev}: 같은 추적 뉴런, vehicle 후 vs SBI-553 후 유형", key_en, key_kr, bot_en, bot_kr,
                                             notes_en, notes_kr, "backup")
    J_["_meta"] = dict(
        script=os.path.abspath(__file__), made="2026-10-23", input=os.path.join(EA, "cells_classes.csv"),
        drug_order=DRUG_ORDER_EN, drug_order_kr=DRUG_ORDER_KR, events=evs,
        events_skipped={e: dict(neurons_with_both_sessions=v, reason="no flow on one drug day: SBI-553 sessions have no escape / "
                                                                      "grooming (>= 5 events), and no vehicle-day baseline had >= 5 "
                                                                      "attending onsets") for e, v in skipped.items()},
        priority_rule="Pin prick and Heat flow (count and % versions) + the two summary figures = main; other events = backup; "
                      "tracked = backup",
        figures_kr="every figure also as <name>_kr.png",
        caveats_one_mouse_dominates=[dict(t) for t in {tuple(sorted(d.items())) for d in CAVEATS}])
    json.dump(clean_nan(J_), open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False,
              default=jconv)


if __name__ == "__main__":
    main()
