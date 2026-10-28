"""ca_session_blocks.py - whole sessions with the stimulus blocks in ONE order, stimulus-block activity on both days, and
behaviour normalized to the same-day baseline (PI requests, 2026-10-23).

WHY
  The earlier whole-session heatmap ('Every neuron over a whole session (5941, SBI-553)') kept every session in its recorded
  order; the four stimulus blocks come in a different order in every session, so the sessions could not be compared by eye.
  The stimulus-block activity slide ('Stimulus-block activity after the injection') was Day 1 only. The PI asked for every
  mouse, then the average, both days, and baseline-normalized comparisons (injection minus its own same-day baseline,
  vehicle day vs SBI-553 day).

WHAT
  1 SESS_<mouse>.png       one figure per mouse: four rows = vehicle-day baseline, after vehicle, SBI-553-day baseline, after
                           SBI-553; rows = the neurons tracked on both days (analysis\\crossday\\<m>_pairs.csv), the SAME row
                           order in all four (sorted by the mean z in the stimulus blocks of the two baseline sessions); 1-s
                           bins of z; time axis rebuilt as: the first 180 s (no stimulus), then the blocks in the fixed order
                           Light touch, Mild touch, Pin prick, Heat, each followed by its own 30-s rest. The stimulus of each
                           block is read from the scoring file (dTypes / stimNames of the deliveries inside the block).
    SESS_average.png       four panels (same session types), rows = population mean of each mouse (tracked neurons) + the
                           mean of the 4 mice.
    SESS_average_norm.png  injection minus same-day baseline (population mean, per mouse rows + mean), vehicle day vs
                           SBI-553 day, plus the mean per segment (C).
  2 RATE_blocks.png        Ca transients per min of still frames in the stimulus blocks (whole-session tables, 'still rate |
                           <block>' averaged over the 4 blocks weighted by the still minutes, as make_day1_ca_slides.whole),
                           baseline vs injection, every neuron of each day, Wilcoxon signed-rank per mouse-day; left = vehicle
                           day, right = SBI-553 day (Day 1 + Day 2).
    RATE_blocks_norm.png   tracked neurons: log2((inj + 0.1) / (base + 0.1)) on the vehicle day vs on the SBI-553 day, a line
                           per neuron, Wilcoxon signed-rank paired by neuron per mouse (+ the 4 mice pooled, exploratory).
  3 BEH_norm_delivery.png  per behaviour, % of deliveries followed by it: injection - same-day baseline, vehicle day vs
                           SBI-553 day, a line per mouse + mean (descriptive, 4 mice).
    BEH_norm_stimulus.png  the same per stimulus type (bars = mean of mice, markers = mice).
    BEH_norm_seconds.png   the same for the seconds of each held behaviour in the 3 s after a delivery (*_s columns).
  Every figure in English (<name>.png) and Korean (<name>_kr.png); summary.json = the numbers + proposed slide texts.

INPUTS
  ca_event_analysis.load_all() (whole-session z = dF/F / robust SD (MAD) of the session; timing-corrected scoring
  scoring_mini1p\\<date>_final_timefix), analysis\\crossday\\<m>_pairs.csv, analysis\\ca_day1\\ca_d1_cells_whole_session.csv,
  analysis\\ca_day2\\ca_d2_cells_whole_session.csv, analysis\\event_analysis\\P6_per_delivery.csv.
OUTPUTS
  analysis\\pi_extra\\session\\ (figures, summary.json, sess_binned_cache.npz = the reordered 1-s bins, session_blocks.csv,
  rate_blocks.csv, rate_blocks_tracked.csv, beh_norm.csv)

  python -I -X utf8 ca_session_blocks.py            (first run loads the sessions, ~1 min; later runs use the cache)
  python -I -X utf8 ca_session_blocks.py --rebuild  (reload the sessions)
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=RuntimeWarning)       # nanmean of the empty first 15 s

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
import pairstars as PS  # noqa: E402  (every star shows the pair it compares)
import figfoot  # noqa: E402  (10-24: the methods text goes to the slide notes, <png>.notes.txt)

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
OUT = os.path.join(B, "analysis", "pi_extra", "session")
CACHE = os.path.join(OUT, "sess_binned_cache.npz")
MO = ["5941", "5939", "5937", "1191"]
SEX = {"5941": "F", "5939": "M", "5937": "M", "1191": "F"}
SBI_DAY = {"5941": 1, "5939": 1, "5937": 2, "1191": 2}
VEH_DAY = {m: 3 - d for m, d in SBI_DAY.items()}
CC = {"baseline": "#7f8c8d", "vehicle": "#2471a3", "SBI-553": "#c0392b"}
MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#34495e"}
MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]
SCOL = {"Light touch": "#5dade2", "Mild touch": "#f5b041", "Pin prick": "#e74c3c", "Heat": "#8e44ad"}
FPS = 30.0
SKIP_S = 15                                      # the first 15 s are excluded in every analysis (z is NaN there)
BLOCK_S = [(180, 360), (390, 570), (600, 780), (810, 990)]
T0, BLK, SEG = 180, 180, 210                     # no-stimulus start, block length, block + its 30-s rest
TOT = T0 + 4 * SEG                               # 1020 s on the rebuilt axis
COND = ["vb", "vi", "sb", "si"]                  # vehicle-day baseline, after vehicle, SBI-553-day baseline, after SBI-553
FW, FH, DPI = 15, 6.3, 150
EPS = 0.1                                        # transients / min added before the log ratio (as ca_robustness.EPS)
BEH = ["withdrawal", "flinch", "attending", "movement", "escape/rearing", "grooming"]
HELD = ["attending", "movement", "escape/rearing", "grooming"]
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11.5,
                     "ytick.labelsize": 11.5, "legend.fontsize": 11.5, "axes.spines.top": False, "axes.spines.right": False})


def font(lg):
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"] if lg == "kr" else ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def ylab(ax, txt):
    """a y label of up to two lines: the second line is the axis label, the first is drawn left of it as its own text
    (a two-line Malgun Gothic label overlaps its own lines whatever the linespacing)"""
    parts = txt.split("\n")
    ax.set_ylabel(parts[-1])
    gap = 8 if any("가" <= ch <= "힣" for ch in txt) else 2     # Hangul glyphs extend past their measured box
    if len(parts) > 1:
        ax.annotate(parts[0], xy=(0, 0.5), xycoords=ax.yaxis.label, xytext=(-gap, 0), textcoords="offset points", rotation=90,
                    ha="right", va="center", fontsize=plt.rcParams["axes.labelsize"])


def lsp(lg):
    """line spacing of a multi-line label (Malgun Gothic lines overlap at the default)"""
    return 2.0 if lg == "kr" else 1.2


def L(lg, en, kr):
    return kr if lg == "kr" else en


def foot(fig, txt, fs=11.5, y=0.006):
    """10-24 (user: details in the notes, not on the slide): the methods text is kept for the slide notes (figfoot writes it
    next to the PNG as <name>.png.notes.txt); the figure shows only the one-line star key when the text has one. Returns the
    number of lines drawn in the figure (1)."""
    figfoot.note(fig, txt, "kr" if any("가" <= ch <= "힣" for ch in txt) else "en", fs=fs, y=y)
    return 1


def cbar_ends(cb, lo, hi, nd=2):
    """colour-bar ticks with the min and the max always labelled (plus round ticks in between that do not crowd them)"""
    from matplotlib.ticker import MaxNLocator
    span = hi - lo
    mid = [t for t in MaxNLocator(5).tick_values(lo, hi) if lo + 0.12 * span < t < hi - 0.12 * span]
    ticks = [lo] + mid + [hi]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.{nd}f}".replace("-0.00", "0.00") for t in ticks])


def cond_day(m, c):
    return VEH_DAY[m] if c in ("vb", "vi") else SBI_DAY[m]


def cond_key(m, c):
    return (m, cond_day(m, c), "pre" if c in ("vb", "sb") else "post")


def ccol(c):
    return CC["vehicle"] if c == "vi" else CC["SBI-553"] if c == "si" else CC["baseline"]


def order_txt(m, lg):
    return L(lg, f"SBI-553 Day {SBI_DAY[m]}, vehicle Day {VEH_DAY[m]}", f"SBI-553 Day {SBI_DAY[m]}, vehicle Day {VEH_DAY[m]}")


def first_txt(m, lg):
    return L(lg, "SBI-553 first" if SBI_DAY[m] == 1 else "vehicle first", "SBI-553 먼저" if SBI_DAY[m] == 1 else "vehicle 먼저")


DRUG_ORDER = {"en": "Drug order: SBI-553 on Day 1 in 5941, 5939, on Day 2 in 5937, 1191 (so drug is confounded with day)",
              "kr": "투여 순서: 5941·5939는 Day 1, 5937·1191은 Day 2에 SBI-553 (약물과 날이 겹침)"}


def wil(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.isfinite(d)]
    if len(d) < 6 or np.all(d == 0):
        return np.nan
    return float(wilcoxon(d, zero_method="wilcox").pvalue)


def pfmt(p):
    return "n.a." if not np.isfinite(p) else "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def jconv(o):
    if isinstance(o, dict):
        return {str(k): jconv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jconv(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ------------------------------------------------------------------------------------------------ data: rebuilt sessions
def reorder(Zb, order):
    """1-s bins (bins x cells) in the recorded order -> TOT bins: 0-180 s as recorded, then block + 30-s rest of each stimulus
    in the order STIM (NaN where not recorded)"""
    n = Zb.shape[1]
    M = np.full((TOT, n), np.nan)
    k = min(T0, len(Zb))
    M[:k] = Zb[:k]
    M[:SKIP_S] = np.nan
    for i, (b0, _) in enumerate(BLOCK_S):
        j = STIM.index(order[i])
        src = Zb[b0:b0 + SEG]
        M[T0 + SEG * j:T0 + SEG * j + len(src)] = src
    return M


def build_cache():
    import ca_event_analysis as E
    S = E.load_all()
    C, rows = {}, []
    for (m, d, ph), s in sorted(S.items()):
        Zb, _ = E.binned(s, 1.0)
        Sm, _ = E.CA.scoring_for(E.SCORE[d], m, ph)
        names = [str(x) for x in np.atleast_1d(Sm["stimNames"])]
        dF = np.atleast_1d(Sm["dFrames"]).astype(int)
        dT = np.atleast_1d(Sm["dTypes"]).astype(int)
        t = dF / FPS
        order, tk_t, tk_s = [], [], []
        for i, (b0, b1) in enumerate(BLOCK_S):
            sel = (t >= b0) & (t < b0 + SEG)                  # the block and its 30-s rest
            u, c = np.unique(dT[sel], return_counts=True)
            st = names[u[np.argmax(c)] - 1]
            order.append(st)
            rows.append(dict(mouse=m, day=d, phase=ph, block=i + 1, start_s=b0, stimulus=st, n_deliveries=int(sel.sum()),
                             n_majority=int(c.max()), n_in_rest=int((sel & (t >= b1)).sum()),
                             stimNames_order=names[i], agrees=names[i] == st))
            j = STIM.index(st)
            tk_t.append(t[sel] - b0 + T0 + SEG * j)
            tk_s.append(np.array([names[x - 1] for x in dT[sel]]))
        n_out = int(len(t) - sum(len(x) for x in tk_t))
        dur = float(s["T"]["t_s"].max())
        C[f"{m}|{d}|{ph}"] = dict(M=reorder(Zb, order).astype(np.float32), ids=s["ids"].astype(int), order=order,
                                  tk_t=np.concatenate(tk_t), tk_s=np.concatenate(tk_s), n_out=n_out, dur=dur,
                                  stim_order=list(s["B"]["stim_order"]))
        print(f"  {m} d{d} {ph}: {Zb.shape[1]} cells, {dur:.1f} s, blocks {order}, deliveries outside blocks {n_out}", flush=True)
    np.savez_compressed(CACHE, C=np.array([C], dtype=object))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "session_blocks.csv"), index=False)
    del S
    return C


def load_pairs():
    P = {}
    for m in MO:
        p = pd.read_csv(os.path.join(B, "analysis", "crossday", f"{m}_pairs.csv"))
        P[m] = p[["day1_id", "day2_id", "track_class"]].astype({"day1_id": int, "day2_id": int})
    return P


def tracked_mats(C, P):
    """mouse -> cond -> (n tracked x TOT) z, in the pairs order"""
    X = {}
    for m in MO:
        X[m] = {}
        for c in COND:
            mm, d, ph = cond_key(m, c)
            s = C[f"{mm}|{d}|{ph}"]
            pos = {int(v): i for i, v in enumerate(s["ids"])}
            idx = [pos[int(v)] for v in P[m][f"day{d}_id"]]
            X[m][c] = s["M"][:, idx].T
    return X


SEGS = {"no stimulus": (SKIP_S, T0)}
SEGS.update({st: (T0 + SEG * j, T0 + SEG * j + BLK) for j, st in enumerate(STIM)})


def blockmask():
    v = np.zeros(TOT, bool)
    for st in STIM:
        a, b = SEGS[st]
        v[a:b] = True
    return v


def seg_means(v):
    o = {k: float(np.nanmean(v[a:b])) for k, (a, b) in SEGS.items()}
    o["stimulus blocks"] = float(np.nanmean(v[blockmask()]))
    return o


# ------------------------------------------------------------------------------------------------ figure helpers
def time_axis(ax, lg, label=True):
    ax.set_xlim(0, TOT)
    ax.set_xticks([0, T0] + [T0 + SEG * j for j in range(1, 5)])
    if label:
        ax.set_xlabel(L(lg, "time in the rebuilt session (s): 0-180 s no stimulus, then each block (180 s) followed by its 30-s rest",
                        "재구성한 세션 시간 (s): 0-180 s 자극 없음, 이후 각 블록 (180 s) + 그 뒤 30 s 휴식"))
    else:
        ax.set_xticklabels([])


def block_lines(ax):
    for j in range(4):
        ax.axvline(T0 + SEG * j, color="w", lw=1.3)
        ax.axvline(T0 + SEG * j + BLK, color="w", lw=1.1, ls=(0, (2, 2)))


def block_labels(fig, Lx, Rx, y, lg):
    fx = lambda t: Lx + (Rx - Lx) * t / TOT
    fig.text(fx(T0 / 2), y, L(lg, "no stimulus", "자극 없음"), ha="center", va="bottom", fontsize=12, color="0.35")
    for j, st in enumerate(STIM):
        fig.text(fx(T0 + SEG * j + BLK / 2), y, st, ha="center", va="bottom", fontsize=12.5, color=SCOL[st], fontweight="bold")
        fig.text(fx(T0 + SEG * j + BLK + 15), y, L(lg, "rest", "휴식"), ha="center", va="bottom", fontsize=11, color="0.5")


def magma():
    cm = plt.get_cmap("magma").copy()
    cm.set_bad("#d5d5d5")
    return cm


def rdbu():
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("#d5d5d5")
    return cm


ROWLAB = {"vb": ("vehicle day\n(Day {d})\nbaseline", "vehicle 날\n(Day {d})\n기준"), "vi": ("after\nvehicle\n(Day {d})", "vehicle 후\n(Day {d})"),
          "sb": ("SBI-553 day\n(Day {d})\nbaseline", "SBI-553 날\n(Day {d})\n기준"), "si": ("after\nSBI-553\n(Day {d})", "SBI-553 후\n(Day {d})")}
ROWLAB2 = {"vb": ("vehicle day\nbaseline", "vehicle 날\n기준"), "vi": ("after\nvehicle", "vehicle\n후"),
           "sb": ("SBI-553 day\nbaseline", "SBI-553 날\n기준"), "si": ("after\nSBI-553", "SBI-553\n후")}
ZLAB = ("z (dF/F / robust SD of the session), 1-s bins", "z (dF/F / 세션 robust SD), 1초 bin")


# ------------------------------------------------------------------------------------------------ 1 SESS per mouse
def fig_mouse(m, X, C, P, lg, fn):
    font(lg)
    n = len(P[m])
    bm = blockmask()
    score = np.nanmean(np.hstack([X[m]["vb"][:, bm], X[m]["sb"][:, bm]]), 1)
    o = np.argsort(-np.nan_to_num(score, nan=-9))
    fig = plt.figure(figsize=(FW, FH))
    Lx, Rx = 0.15, 0.905                     # 10-24: no in-figure title / footer -> taller rows
    top, rowh, tkh, hh = 0.872, 0.193, 0.026, 0.160
    im = None
    for r, c in enumerate(COND):
        mm, d, ph = cond_key(m, c)
        s = C[f"{mm}|{d}|{ph}"]
        yt = top - r * rowh
        tk = fig.add_axes([Lx, yt - tkh, Rx - Lx, tkh])
        for st in STIM:
            tt = s["tk_t"][s["tk_s"] == st]
            tk.vlines(tt, 0.08, 0.92, color=SCOL[st], lw=0.9)
        tk.set_xlim(0, TOT); tk.set_ylim(0, 1); tk.set_axis_off()
        ax = fig.add_axes([Lx, yt - tkh - hh, Rx - Lx, hh])
        im = ax.imshow(np.ma.masked_invalid(X[m][c][o]), aspect="auto", cmap=magma(), vmin=-0.5, vmax=3, interpolation="nearest",
                       extent=[0, TOT, n, 0])
        block_lines(ax)
        ax.set_yticks([0, n], ["1", str(n)], fontsize=11.5)
        time_axis(ax, lg, label=r == 3)
        lab = ROWLAB[c][1 if lg == "kr" else 0].format(d=d)
        fig.text(0.008, yt - tkh - hh / 2, lab, ha="left", va="center", fontsize=12, color=ccol(c), fontweight="bold",
                 linespacing=1.05)
    ymid = (top - tkh + (top - 3 * rowh - tkh - hh)) / 2          # one y label for the 4 heatmaps
    fig.text(Lx - 0.043, ymid, L(lg, "neurons (1 row = 1 neuron)\nsame order in all 4 sessions",
                                 "뉴런 (1행 = 뉴런 1개)\n네 세션 모두 같은 순서"),
             rotation=90, ha="center", va="center", fontsize=11.5, linespacing=lsp(lg) if lg == "kr" else 1.15)
    block_labels(fig, Lx, Rx, top + 0.006, lg)
    cax = fig.add_axes([0.918, top - 3 * rowh - tkh - hh + 0.02, 0.011, 3 * rowh + hh - 0.04])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label(L(lg, *ZLAB), fontsize=11.5)
    cb.ax.tick_params(labelsize=11.5)
    early = {c: foreign(C["|".join(map(str, cond_key(m, c)))]) for c in COND}
    h = [Line2D([], [], color="0.25", marker="|", ls="", ms=11, mew=1.6, label=L(lg, "tick = one delivery (colour = stimulus)",
                                                                                "틱 = 자극 1회 (색 = 자극)")),
         # 10-25: the mid-row grey = end of the 30-s rest of the block that ran LAST in the real session (recording stopped at
         # ~992-1011 s < 1020 s); reorder() moves that block to its stimulus slot, so the gap can sit mid-row (no data change)
         Patch(color="#d5d5d5", label=L(lg, "grey = not recorded: first 15 s; end of the rest after the block run last (session ended)",
                                        "회색 = 기록 없음: 처음 15 s; 마지막에 진행한 블록 뒤 휴식의 끝 (세션 종료)")),
         Line2D([], [], color="0.55", lw=1.6, label=L(lg, "white solid / dotted line = block start / end", "흰 실선 / 점선 = 블록 시작 / 끝"))]
    if any(early.values()):
        st0 = next(v for v in early.values() if v)[0][0]
        h.append(Line2D([], [], color=SCOL[st0], marker="|", ls="", ms=11, mew=1.6,
                        label=L(lg, "tick of another colour in a rest = the next block began early",
                                "휴식 안의 다른 색 틱 = 다음 블록이 일찍 시작")))
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.005, 0.998), ncol=2, frameon=False, fontsize=11.5, handlelength=1.6,
               columnspacing=2.0, labelspacing=0.35)
    ex_en = "; ".join(f"{ROWLAB2[c][0].replace(chr(10), ' ')}: {len(v)} {v[0][0]} deliveries {min(e for _, e in v):.0f}-"
                      f"{max(e for _, e in v):.0f} s before the {v[0][0]} block started" for c, v in early.items() if v)
    ex_kr = "; ".join(f"{ROWLAB2[c][1].replace(chr(10), ' ')}: {v[0][0]} 자극 {len(v)}회가 {v[0][0]} 블록 시작 "
                      f"{min(e for _, e in v):.0f}-{max(e for _, e in v):.0f} s 전" for c, v in early.items() if v)
    dd = [C["|".join(map(str, cond_key(m, c)))]["dur"] for c in COND]
    durs = (min(dd), max(dd))
    foot(fig, L(lg, f"{m} ({SEX[m]}; {order_txt(m, lg)}), {n} neurons tracked on both days. "
                    "Rows = neurons, the same order in all four sessions (sorted by the mean z in the stimulus blocks of the two "
                    "baselines). Blocks moved to the order Light touch, Mild touch, Pin prick, Heat (stimulus read from the scoring file), "
                    "each with its own 30-s rest. Grey = not recorded: the first 15 s, and the end of the 30-s rest after the "
                    f"block that ran last in the real session (recording stopped at {durs[0]:.0f}-{durs[1]:.0f} s, before 1020 s); "
                    "after the blocks are moved into one order this gap can sit mid-row." + (f" A tick of another colour in a rest = the next block began early ({ex_en})."
                                                      if ex_en else "") + f" {DRUG_ORDER['en']}.",
                f"{m} ({SEX[m]}; {order_txt(m, lg)}), 두 날 모두 추적한 뉴런 {n}개. "
                "행 = 뉴런, 네 세션 모두 같은 순서 (두 기준 세션 자극 블록의 평균 z 순). 블록을 Light touch, Mild touch, Pin prick, "
                "Heat 순서로 옮김 (자극은 채점 파일에서), 각 블록은 자기 30 s 휴식과 함께. 회색 = 기록 없음: 처음 15 s, 그리고 "
                f"실제 세션에서 마지막에 진행한 블록 뒤 30 s 휴식의 끝 (기록이 {durs[0]:.0f}-{durs[1]:.0f} s에 끝나 1020 s 전에 멈춤); "
                "블록을 한 순서로 옮기면 이 빈칸이 행 중간에 올 수 있음." +
                (f" 휴식 안의 다른 색 틱 = 다음 블록이 일찍 시작 ({ex_kr})." if ex_kr else "") + f" {DRUG_ORDER['kr']}."))
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)
    return dict(n_tracked=n, early_deliveries={c: v for c, v in early.items() if v})


def foreign(s):
    """deliveries of another stimulus inside a block's segment: the next recorded block began before its scheduled start, so its
    first deliveries fall in the 30-s rest of the block before -> [(stimulus, seconds before the scheduled start)]"""
    t, st = s["tk_t"], s["tk_s"]
    j = np.clip(((t - T0) // SEG).astype(int), 0, 3)
    bad = np.array([STIM[k] != x for k, x in zip(j, st)], bool)
    return [(str(x), float(T0 + SEG * (k + 1) - tt)) for tt, x, k in zip(t[bad], st[bad], j[bad])]


# ------------------------------------------------------------------------------------------------ 1 SESS average
def pop(X):
    """mouse -> cond -> population mean over the tracked neurons (TOT,)"""
    return {m: {c: np.nanmean(X[m][c], 0) for c in COND} for m in MO}


def pop_all(C):
    """the same with every neuron of each session (check)"""
    return {m: {c: np.nanmean(C["|".join(map(str, cond_key(m, c)))]["M"], 1) for c in COND} for m in MO}


def stack_rows(Pm, c):
    R = np.array([Pm[m][c] for m in MO])
    return np.vstack([R, np.nanmean(R, 0, keepdims=True)])


def row_labels(fig, ax, lg, x):
    for i, m in enumerate(MO):
        ax.text(x, i + 0.5, m, ha="right", va="center", fontsize=11.5, color=MC[m], fontweight="bold",
                transform=ax.get_yaxis_transform())
    ax.text(x, 4.5, L(lg, "mean of 4", "4마리 평균"), ha="right", va="center", fontsize=11.5, color="k", fontweight="bold",
            transform=ax.get_yaxis_transform())
    ax.axhline(4, color="w", lw=2.5)


def fig_average(Pm, P, lg, fn):
    font(lg)
    A = {c: stack_rows(Pm, c) for c in COND}
    allv = np.concatenate([A[c].ravel() for c in COND])
    lo, hi = float(np.nanpercentile(allv, 1)), float(np.nanpercentile(allv, 99))
    lo = min(lo, 0.0)
    fig = plt.figure(figsize=(FW, FH))
    Lx, Rx = 0.165, 0.905                    # 10-24: no in-figure title / footer -> taller rows
    top, rowh, hh = 0.862, 0.193, 0.172
    im = None
    for r, c in enumerate(COND):
        yt = top - r * rowh
        ax = fig.add_axes([Lx, yt - hh, Rx - Lx, hh])
        im = ax.imshow(np.ma.masked_invalid(A[c]), aspect="auto", cmap=magma(), vmin=lo, vmax=hi, interpolation="nearest",
                       extent=[0, TOT, 5, 0])
        block_lines(ax)
        ax.set_yticks([])
        row_labels(fig, ax, lg, -0.006)
        time_axis(ax, lg, label=r == 3)
        fig.text(0.008, yt - hh / 2, ROWLAB2[c][1 if lg == "kr" else 0], ha="left", va="center", fontsize=12.5, color=ccol(c),
                 fontweight="bold", linespacing=1.05)
    block_labels(fig, Lx, Rx, top + 0.006, lg)
    cax = fig.add_axes([0.918, top - 3 * rowh - hh + 0.02, 0.011, 3 * rowh + hh - 0.04])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label(L(lg, "population mean z (dF/F / robust SD), 1-s bins", "집단 평균 z (dF/F / robust SD), 1초 bin"), fontsize=11.5)
    cb.ax.tick_params(labelsize=11.5)
    cbar_ends(cb, lo, hi)
    nt = ", ".join(f"{m} {len(P[m])}" for m in MO)
    fig.text(0.008, 0.993, L(lg, f"row = one mouse: mean of its neurons tracked on both days (n = {nt}), the same neurons in all 4 "
                                 "sessions; bottom row = mean of 4 mice",
                             f"행 = 마우스 1마리: 두 날 모두 추적한 뉴런의 평균 (n = {nt}), 네 세션 모두 같은 뉴런; 맨 아래 행 = 4마리 평균"),
             fontsize=12, va="top", color="0.15")
    fig.legend(handles=[Patch(color="#d5d5d5", label=L(lg, "grey = not recorded", "회색 = 기록 없음")),
                        Line2D([], [], color="0.55", lw=1.6, label=L(lg, "white solid / dotted line = block start / end (a 30-s rest follows)",
                                                                     "흰 실선 / 점선 = 블록 시작 / 끝 (뒤에 30 s 휴식)"))],
               loc="upper left", bbox_to_anchor=(0.005, 0.958), ncol=2, frameon=False, fontsize=11.5, handlelength=1.6, columnspacing=2.0)
    foot(fig, L(lg, f"Each session's blocks were moved to the order Light touch, Mild touch, Pin prick, Heat (stimulus read from the "
                    f"scoring file); white solid / dotted line = block start / end (30-s rest follows). Colour range = 1st-99th percentile "
                    f"of all rows. {DRUG_ORDER['en']}.",
                f"각 세션의 블록을 Light touch, Mild touch, Pin prick, Heat 순서로 옮김 (자극은 채점 파일에서); 흰 실선 / 점선 = 블록 "
                f"시작 / 끝 (뒤에 30 s 휴식). 색 범위 = 모든 행의 1-99 백분위. {DRUG_ORDER['kr']}."))
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)
    return dict(color_range=[lo, hi])


def fig_average_norm(Pm, P, lg, fn):
    font(lg)
    D = {"v": np.array([Pm[m]["vi"] - Pm[m]["vb"] for m in MO]), "s": np.array([Pm[m]["si"] - Pm[m]["sb"] for m in MO])}
    D = {k: np.vstack([v, np.nanmean(v, 0, keepdims=True)]) for k, v in D.items()}
    allv = np.concatenate([D[k].ravel() for k in D])
    vmax = float(np.nanpercentile(np.abs(allv), 98))
    fig = plt.figure(figsize=(FW, FH))
    Lx, Rx = 0.155, 0.645                    # 10-24: no in-figure title / footer -> taller heatmaps
    top, rowh, hh = 0.858, 0.392, 0.330
    im = None
    lab = {"v": (("A  vehicle day:\nafter -\nbaseline", "A  vehicle 날:\n후 - 기준"), CC["vehicle"]),
           "s": (("B  SBI-553 day:\nafter -\nbaseline", "B  SBI-553 날:\n후 - 기준"), CC["SBI-553"])}
    for r, k in enumerate(("v", "s")):
        yt = top - r * rowh
        ax = fig.add_axes([Lx, yt - hh, Rx - Lx, hh])
        im = ax.imshow(np.ma.masked_invalid(D[k]), aspect="auto", cmap=rdbu(), vmin=-vmax, vmax=vmax, interpolation="nearest",
                       extent=[0, TOT, 5, 0])
        for j in range(4):
            ax.axvline(T0 + SEG * j, color="k", lw=0.9)
            ax.axvline(T0 + SEG * j + BLK, color="k", lw=0.8, ls=(0, (2, 2)))
        ax.set_yticks([])
        row_labels(fig, ax, lg, -0.008)
        ax.axhline(4, color="k", lw=1.5)
        ax.set_xlim(0, TOT)
        ax.set_xticks([0, T0] + [T0 + SEG * j for j in range(1, 5)])
        if r == 1:
            ax.set_xlabel(L(lg, "time in the rebuilt session (s); block 180 s + 30-s rest", "재구성한 세션 시간 (s); 블록 180 s + 30 s 휴식"))
        else:
            ax.set_xticklabels([])
        fig.text(0.008, yt - hh / 2, lab[k][0][1 if lg == "kr" else 0], ha="left", va="center", fontsize=12.5, color=lab[k][1],
                 fontweight="bold", linespacing=1.05)
    fx = lambda t: Lx + (Rx - Lx) * t / TOT
    fig.text(fx(T0 / 2), top + 0.006, L(lg, "no stim.", "자극 없음"), ha="center", va="bottom", fontsize=11.5, color="0.35")
    for j, st in enumerate(STIM):
        fig.text(fx(T0 + SEG * j + SEG / 2), top + 0.006, st, ha="center", va="bottom", fontsize=11.5, color=SCOL[st], fontweight="bold")
    cax = fig.add_axes([0.655, top - rowh - hh + 0.03, 0.009, rowh + hh - 0.06])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label(L(lg, "after - same-day baseline\n(population mean z)", "후 - 같은 날 기준\n(집단 평균 z)"), fontsize=11.5, labelpad=9, linespacing=lsp(lg))
    cb.ax.tick_params(labelsize=11.5)
    cbar_ends(cb, -vmax, vmax)
    # C: mean per segment
    ax = fig.add_axes([0.81, 0.20, 0.18, 0.62])
    segs = ["no stimulus"] + STIM
    res = {}
    for i, sg in enumerate(segs):
        a, b = SEGS[sg]
        for k, dx, col in (("v", -0.18, CC["vehicle"]), ("s", 0.18, CC["SBI-553"])):
            vals = np.array([np.nanmean(D[k][q, a:b]) for q in range(4)])
            res[(sg, k)] = vals
            ax.bar(i + dx, vals.mean(), width=0.34, color=col, alpha=0.35, edgecolor=col)
            for q, m in enumerate(MO):
                ax.scatter(i + dx, vals[q], marker=MK[m], s=34, color=MC[m], edgecolor="k", lw=0.5, zorder=3)
        for q, m in enumerate(MO):
            ax.plot([i - 0.18, i + 0.18], [res[(sg, "v")][q], res[(sg, "s")][q]], color=MC[m], lw=0.8, alpha=0.7, zorder=2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(5), [L(lg, "no\nstim.", "자극\n없음"), "light", "mild", "pin", "heat"], fontsize=11.5)
    ax.set_ylabel(L(lg, "after - baseline (mean z)", "후 - 기준 (평균 z)"))
    ax.set_title(L(lg, "C  mean per segment", "C  구간별 평균"), loc="left", fontsize=13)
    h = [Patch(color=CC["vehicle"], alpha=0.5, label=L(lg, "vehicle day (bar = mean)", "vehicle 날 (막대 = 평균)")),
         Patch(color=CC["SBI-553"], alpha=0.5, label=L(lg, "SBI-553 day", "SBI-553 날"))]
    h += [Line2D([], [], marker=MK[m], color=MC[m], mec="k", mew=0.5, ls="-", lw=0.8, ms=7,
                 label=f"{m} ({first_txt(m, lg)})") for m in MO]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.005, 0.958), ncol=6, frameon=False, fontsize=11.5, handlelength=1.5,
               columnspacing=1.3)
    # 10-24: what the grey bands and the black lines in A and B are (was only on the slide before)
    fig.legend(handles=[Patch(facecolor="#d5d5d5", edgecolor="0.6", label=L(lg, "A, B: grey = not recorded", "A, B: 회색 = 기록 없음")),
                        Line2D([], [], color="k", lw=0.9, label=L(lg, "solid line = block start", "실선 = 블록 시작")),
                        Line2D([], [], color="k", lw=0.9, ls=(0, (2, 2)), label=L(lg, "dotted line = block end (a 30-s rest follows)",
                                                                                   "점선 = 블록 끝 (뒤에 30 s 휴식)"))],
               loc="upper left", bbox_to_anchor=(0.005, 0.998), ncol=3, frameon=False, fontsize=11.5, handlelength=1.8,
               columnspacing=2.0)
    nt = ", ".join(f"{m} {len(P[m])}" for m in MO)
    foot(fig, L(lg, f"Population mean z of the neurons tracked on both days (n = {nt}) in 1-s bins, blocks in the order light, mild, "
                    f"pin, heat; each injection session minus the baseline session of the same day. C = mean over the 180-s block (or "
                    f"15-180 s for no stimulus); marker = mouse, line joins its two days. Descriptive (4 mice). {DRUG_ORDER['en']}.",
                f"두 날 모두 추적한 뉴런 (n = {nt})의 집단 평균 z, 1초 bin, 블록 순서 light, mild, pin, heat; 각 주사 세션에서 같은 "
                f"날 기준 세션을 뺌. C = 180 s 블록 평균 (자극 없음은 15-180 s); 표식 = 마우스, 선 = 그 마우스의 두 날. 기술 통계 "
                f"(4마리). {DRUG_ORDER['kr']}."))
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)
    return {f"{sg} | {'vehicle day' if k == 'v' else 'SBI-553 day'}": dict(zip(MO, v)) for (sg, k), v in res.items()}, vmax


# ------------------------------------------------------------------------------------------------ 2 rates
def whole_tables():
    out = {}
    for d, f in ((1, os.path.join(B, "analysis", "ca_day1", "ca_d1_cells_whole_session.csv")),
                 (2, os.path.join(B, "analysis", "ca_day2", "ca_d2_cells_whole_session.csv"))):
        W = pd.read_csv(f)
        W["mouse"] = W.mouse.astype(str)
        num = sum(W[f"still rate | {s}"].fillna(0) * W[f"still min | {s}"].fillna(0) for s in STIM)
        den = sum(W[f"still min | {s}"].fillna(0) for s in STIM)
        W["stim_still"] = num / den.replace(0, np.nan)
        W["day"] = d
        out[d] = W
    return out


def rate_data(W):
    R, rows = {}, []
    for m in MO:
        for drug, d in (("vehicle", VEH_DAY[m]), ("SBI-553", SBI_DAY[m])):
            Wd = W[d]
            assert set(Wd[Wd.mouse == m].arm) == {drug}, (m, d, drug)
            Pv = Wd[Wd.mouse == m].pivot(index="cell", columns="phase", values="stim_still").dropna()
            y = np.log2((Pv.post + EPS) / (Pv.pre + EPS))
            p = wil(Pv.post, Pv.pre)
            R[(m, drug)] = Pv
            rows.append(dict(mouse=m, drug=drug, day=d, n_cells=len(Pv), base_median=float(Pv.pre.median()),
                             inj_median=float(Pv.post.median()), log2_ratio_median=float(np.median(y)),
                             change_pct=float(100 * (2 ** np.median(y) - 1)), p_wilcoxon=p,
                             n_zero_both=int(((Pv.pre == 0) & (Pv.post == 0)).sum())))
    return R, pd.DataFrame(rows)


def fig_rate(R, T, lg, fn):
    font(lg)
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.08, 0.215, 0.91, 0.64])            # 10-24: no in-figure title / footer -> taller axes
    rng = np.random.default_rng(0)
    ymax = max(float(np.nanmax(np.r_[P.pre, P.post])) for P in R.values())
    yb = ymax * 1.06
    for h_, drug in enumerate(("vehicle", "SBI-553")):
        for i, m in enumerate(MO):
            Pv = R[(m, drug)]
            x0 = 13 * h_ + 3 * i
            x1 = x0 + 1
            for _, r in Pv.iterrows():
                ax.plot([x0, x1], [r.pre, r.post], color="0.8", lw=0.5, zorder=1)
            for x, ph, col in ((x0, "pre", CC["baseline"]), (x1, "post", CC[drug])):
                # 10-24: dots / medians at 0 sit on the x axis: not clipped, drawn above the axis line
                ax.scatter(np.full(len(Pv), x) + rng.uniform(-0.14, 0.14, len(Pv)), Pv[ph], s=9, color=col, alpha=0.75, lw=0, zorder=3,
                           clip_on=False)
                ax.plot([x - 0.33, x + 0.33], [Pv[ph].median()] * 2, color="k", lw=2.6, zorder=4, clip_on=False)
            rr = T[(T.mouse == m) & (T.drug == drug)].iloc[0]
            PS.bracket(ax, x0, x1, yb, rr.p_wilcoxon, color=CC[drug], fs=13, inset=0.0)
            d = int(rr.day)
            if (Pv.pre == 0).mean() > 0.5:                     # floor: most neurons without a transient
                nz = int(((Pv.pre == 0) & (Pv.post == 0)).sum())
                ax.text((x0 + x1) / 2, ymax * 0.42, L(lg, f"{nz}/{len(Pv)} neurons\n0 / min in\nboth sessions", f"뉴런 {nz}/{len(Pv)}\n두 세션 모두\n0 / 분"),
                        ha="center", va="bottom", fontsize=11, color="0.35")
            ax.text((x0 + x1) / 2, -0.12, f"{m}\nDay {d}\nn = {int(rr.n_cells)}", ha="center", va="top", fontsize=11.5, color=MC[m],
                    fontweight="bold", transform=ax.get_xaxis_transform(), linespacing=1.05)
        xc = 13 * h_ + 4.5
        ax.text(xc, 1.03, L(lg, f"{drug} day", f"{drug} 날"), ha="center", va="bottom", fontsize=14, fontweight="bold", color=CC[drug],
                transform=ax.get_xaxis_transform())
    ax.axvline(11.5, color="0.6", lw=1, ls="--")
    ax.set_xticks([x for h_ in range(2) for i in range(4) for x in (13 * h_ + 3 * i, 13 * h_ + 3 * i + 1)],
                  [L(lg, "base", "기준"), L(lg, "inj.", "주사")] * 8, fontsize=11.5)
    ax.set_xlim(-0.8, 23.8)
    ax.set_ylim(0, ymax * 1.16)
    ylab(ax, L(lg, "Ca transients / min\n(still frames, 4 stimulus blocks)", "분당 Ca transient\n(정지 프레임, 자극 블록 4개)"))
    h = [Line2D([], [], marker="o", color=CC["baseline"], ls="", ms=6, label=L(lg, "baseline (same day)", "기준 (같은 날)")),
         Line2D([], [], marker="o", color=CC["vehicle"], ls="", ms=6, label=L(lg, "after vehicle", "vehicle 후")),
         Line2D([], [], marker="o", color=CC["SBI-553"], ls="", ms=6, label=L(lg, "after SBI-553", "SBI-553 후")),
         Line2D([], [], color="0.75", lw=1, label=L(lg, "one neuron, baseline to injection", "뉴런 1개, 기준 → 주사")),
         Line2D([], [], color="k", lw=2.6, label=L(lg, "median of the neurons", "뉴런 중앙값"))]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.005, 0.998), ncol=5, frameon=False, fontsize=11.5)
    foot(fig, L(lg, "Dot = one neuron (every neuron of that day; the same neurons in the baseline and injection sessions of a day). "
                    "Rate = Ca transients per min while the mouse is still, the 4 stimulus blocks pooled (weighted by the still minutes). "
                    "Brackets: Wilcoxon signed-rank, injection vs same-day baseline, over the neurons of one mouse-day. "
                    f"{PS.KEY}. {DRUG_ORDER['en']}.",
                "점 = 뉴런 1개 (그날의 모든 뉴런; 같은 날 기준·주사 세션은 같은 뉴런). 비율 = 정지 상태에서 분당 Ca transient, 자극 블록 "
                "4개 합침 (정지 시간 가중). 괄호: Wilcoxon signed-rank, 주사 vs 같은 날 기준, 마우스-날 하나의 뉴런들. "
                f"* p < 0.05, ** p < 0.01, *** p < 0.001, ns = 유의하지 않음. {DRUG_ORDER['kr']}."))
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)


def rate_tracked(R, P):
    rows = []
    for m in MO:
        dv, ds = VEH_DAY[m], SBI_DAY[m]
        Rv, Rs = R[(m, "vehicle")], R[(m, "SBI-553")]
        for _, p in P[m].iterrows():
            iv, is_ = int(p[f"day{dv}_id"]), int(p[f"day{ds}_id"])
            if iv not in Rv.index or is_ not in Rs.index:
                continue
            a, b = Rv.loc[iv], Rs.loc[is_]
            rows.append(dict(mouse=m, track_class=p.track_class, veh_id=iv, sbi_id=is_, veh_base=a.pre, veh_inj=a.post,
                             sbi_base=b.pre, sbi_inj=b.post, veh_log2=np.log2((a.post + EPS) / (a.pre + EPS)),
                             sbi_log2=np.log2((b.post + EPS) / (b.pre + EPS))))
    return pd.DataFrame(rows)


def fig_rate_norm(Tk, lg, fn):
    font(lg)
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.095, 0.215, 0.895, 0.69])          # 10-24: no in-figure title / footer -> taller axes
    rng = np.random.default_rng(1)
    res = {}
    groups = [(m, Tk[Tk.mouse == m]) for m in MO] + [("all", Tk)]
    lo, hi = float(np.nanmin(Tk[["veh_log2", "sbi_log2"]].values)), float(np.nanmax(Tk[["veh_log2", "sbi_log2"]].values))
    span = hi - lo
    for i, (m, G) in enumerate(groups):
        x0 = 3 * i + (0.6 if m == "all" else 0)
        x1 = x0 + 1
        for _, r in G.iterrows():
            ax.plot([x0, x1], [r.veh_log2, r.sbi_log2], color=MC[r.mouse] if m == "all" else "0.78", lw=0.5,
                    alpha=0.45 if m == "all" else 1, zorder=1)
        for x, col, cl in ((x0, CC["vehicle"], "veh_log2"), (x1, CC["SBI-553"], "sbi_log2")):
            ax.scatter(np.full(len(G), x) + rng.uniform(-0.12, 0.12, len(G)), G[cl], s=10, color=col, alpha=0.75, lw=0, zorder=2)
            ax.plot([x - 0.33, x + 0.33], [G[cl].median()] * 2, color="k", lw=2.6, zorder=3)
        p = wil(G.sbi_log2, G.veh_log2)
        res[m] = dict(n=len(G), veh_median=float(G.veh_log2.median()), sbi_median=float(G.sbi_log2.median()),
                      diff_median=float((G.sbi_log2 - G.veh_log2).median()), p_wilcoxon=p,
                      n_lower_on_sbi=int((G.sbi_log2 < G.veh_log2).sum()))
        PS.bracket(ax, x0, x1, hi + 0.08 * span, p, color="k", fs=13, inset=0.0)
        lab = (f"{m}\n{first_txt(m, lg)}\nn = {len(G)}" if m != "all" else
               L(lg, f"4 mice pooled\n(exploratory)\nn = {len(G)}", f"4마리 합침\n(탐색적)\nn = {len(G)}"))
        if m != "all" and ((G.sbi_base == 0) & (G.sbi_inj == 0)).mean() > 0.5:      # floor (5937 Day 2)
            nz = int(((G.sbi_base == 0) & (G.sbi_inj == 0)).sum())
            ax.text(x1 + 0.35, hi - 0.05 * span, L(lg, f"SBI day: {nz}/{len(G)}\nneurons 0 / min\nin both sessions\n(log2 = 0)",
                                                    f"SBI 날: 뉴런\n{nz}/{len(G)} 두 세션\n모두 0 / 분\n(log2 = 0)"),
                    ha="left", va="top", fontsize=11, color="0.35")
        ax.text((x0 + x1) / 2, -0.12, lab, ha="center", va="top", fontsize=11.5, color=MC.get(m, "k"), fontweight="bold",
                transform=ax.get_xaxis_transform(), linespacing=1.05)
    ax.axvline(11.8, color="0.6", lw=1, ls="--")
    ax.axhline(0, color="0.3", lw=0.9, ls=":")
    ax.set_xticks([x for i in range(5) for x in (3 * i + (0.6 if i == 4 else 0), 3 * i + 1 + (0.6 if i == 4 else 0))],
                  [L(lg, "veh. day", "veh 날"), L(lg, "SBI day", "SBI 날")] * 5, fontsize=11.5)
    ax.set_xlim(-0.8, 14.4)
    ax.set_ylim(lo - 0.05 * span, hi + 0.2 * span)
    ylab(ax, L(lg, "log2 fold change, injection / same-day baseline\n(transients / min)",
                    "log2 배수 변화, 주사 / 같은 날 기준\n(분당 transient)"))
    h = [Line2D([], [], marker="o", color=CC["vehicle"], ls="", ms=6, label=L(lg, "vehicle day: change from its baseline", "vehicle 날: 기준 대비 변화")),
         Line2D([], [], marker="o", color=CC["SBI-553"], ls="", ms=6, label=L(lg, "SBI-553 day: change from its baseline", "SBI-553 날: 기준 대비 변화")),
         Line2D([], [], color="0.75", lw=1, label=L(lg, "one tracked neuron (pooled: colour = mouse)", "추적 뉴런 1개 (합침: 색 = 마우스)")),
         Line2D([], [], color="k", lw=2.6, label=L(lg, "median", "중앙값")),
         Line2D([], [], color="0.3", lw=0.9, ls=":", label=L(lg, "0 = no change", "0 = 변화 없음"))]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.005, 0.998), ncol=5, frameon=False, fontsize=11.5, columnspacing=1.2)
    foot(fig, L(lg, "Neurons tracked on both days (class A + B). Per neuron and day: log2 of (rate after the injection + 0.1) / "
                    "(rate at the same-day baseline + 0.1), rate = Ca transients per min of still frames in the 4 stimulus blocks. "
                    "Brackets: Wilcoxon signed-rank paired by neuron, SBI-553 day vs vehicle day (pooled = neurons of 4 mice, "
                    f"exploratory). {PS.KEY}. {DRUG_ORDER['en']}.",
                "두 날 모두 추적한 뉴런 (A + B 등급). 뉴런·날마다: log2 (주사 후 비율 + 0.1) / (같은 날 기준 비율 + 0.1), 비율 = 자극 "
                "블록 4개의 정지 프레임 분당 Ca transient. 괄호: 뉴런별 짝지은 Wilcoxon signed-rank, SBI-553 날 vs vehicle 날 (합침 = "
                f"4마리 뉴런, 탐색적). * p < 0.05, ** p < 0.01, *** p < 0.001, ns = 유의하지 않음. {DRUG_ORDER['kr']}."))
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)
    return res


# ------------------------------------------------------------------------------------------------ 3 behaviour
def beh_norm():
    D = pd.read_csv(os.path.join(B, "analysis", "event_analysis", "P6_per_delivery.csv"))
    D["mouse"] = D.mouse.astype(str)
    rows = []
    for m in MO:
        for drug, d in (("vehicle", VEH_DAY[m]), ("SBI-553", SBI_DAY[m])):
            X = D[(D.mouse == m) & (D.day == d)]
            assert set(X[X.phase == "post"].condition) == {drug}, (m, d, drug)
            for st in ["all"] + STIM:
                Y = X if st == "all" else X[X.stim == st]
                b, i = Y[Y.phase == "pre"], Y[Y.phase == "post"]
                r = dict(mouse=m, drug=drug, day=d, stim=st, n_base=len(b), n_inj=len(i))
                for bn in BEH:
                    r[f"{bn} base %"] = 100 * b[bn].astype(float).mean()
                    r[f"{bn} inj %"] = 100 * i[bn].astype(float).mean()
                    r[f"{bn} change"] = r[f"{bn} inj %"] - r[f"{bn} base %"]
                for bn in HELD:
                    r[f"{bn}_s base"] = b[f"{bn}_s"].mean()
                    r[f"{bn}_s inj"] = i[f"{bn}_s"].mean()
                    r[f"{bn}_s change"] = r[f"{bn}_s inj"] - r[f"{bn}_s base"]
                rows.append(r)
    return pd.DataFrame(rows)


def beh_lines(fig, axes, N, cols, lg, ylab_):
    res = {}
    for a, (bn, col) in zip(axes, cols):
        V = N[N.stim == "all"].pivot(index="mouse", columns="drug", values=col).reindex(MO)
        for m in MO:
            a.plot([0, 1], [V.at[m, "vehicle"], V.at[m, "SBI-553"]], color=MC[m], marker=MK[m], ms=8, mec="k", mew=0.5, lw=1.4)
        for x, drug in ((0, "vehicle"), (1, "SBI-553")):
            a.plot([x - 0.22, x + 0.22], [V[drug].mean()] * 2, color="k", lw=3, zorder=4)
        a.axhline(0, color="0.4", lw=0.9, ls=":")
        a.set_xticks([0, 1], [L(lg, "vehicle\nday", "vehicle\n날"), L(lg, "SBI-553\nday", "SBI-553\n날")])
        for t, c in zip(a.get_xticklabels(), (CC["vehicle"], CC["SBI-553"])):
            t.set_color(c)
            t.set_fontweight("bold")
        a.set_xlim(-0.45, 1.45)
        a.set_title(bn, fontsize=13)
        lo, hi = a.get_ylim()
        m_ = max(abs(lo), abs(hi))
        a.set_ylim(-m_ * 1.08, m_ * 1.08)
        res[bn] = dict(vehicle_day=dict(V["vehicle"]), sbi_day=dict(V["SBI-553"]), mean_vehicle=float(V["vehicle"].mean()),
                       mean_sbi=float(V["SBI-553"].mean()), n_mice_sbi_lower=int((V["SBI-553"] < V["vehicle"]).sum()))
    ylab(axes[0], ylab_)
    h = [Line2D([], [], color=MC[m], marker=MK[m], mec="k", mew=0.5, ms=8, lw=1.4, label=f"{m} ({first_txt(m, lg)})") for m in MO]
    h += [Line2D([], [], color="k", lw=3, label=L(lg, "mean of 4 mice", "4마리 평균")),
          Line2D([], [], color="0.4", lw=0.9, ls=":", label=L(lg, "0 = same as baseline", "0 = 기준과 같음"))]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.005, 0.998), ncol=6, frameon=False, fontsize=11.5, columnspacing=1.4)
    return res


def fig_beh_delivery(N, lg, fn):
    font(lg)
    fig, axes = plt.subplots(1, 6, figsize=(FW, FH))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.875, bottom=0.12, wspace=0.42)   # 10-24: no title / footer -> taller
    res = beh_lines(fig, axes, N, [(b, f"{b} change") for b in BEH], lg,
                    L(lg, "change in % of deliveries\n(injection - same-day baseline)", "자극 대비 비율 변화 (%p)\n(주사 - 같은 날 기준)"))
    nb = N[N.stim == "all"]
    foot(fig, L(lg, f"Per mouse and day: % of deliveries followed by the behaviour after the injection minus the same in the baseline "
                    f"session of that day (deliveries per session {int(nb[['n_base', 'n_inj']].values.min())}-"
                    f"{int(nb[['n_base', 'n_inj']].values.max())}). Withdrawal / flinch = within 2 s of the delivery (or until the next); "
                    f"the others = a bout starting 0-3 s after it. Descriptive, n = 4 mice (no test can reach p < 0.05). "
                    f"{DRUG_ORDER['en']}.",
                f"마우스·날마다: 주사 후 그 행동이 뒤따른 자극의 % - 그날 기준 세션의 같은 값 (세션당 자극 "
                f"{int(nb[['n_base', 'n_inj']].values.min())}-{int(nb[['n_base', 'n_inj']].values.max())}회). Withdrawal / flinch = "
                f"자극 후 2 s 이내 (또는 다음 자극까지); 나머지 = 자극 후 0-3 s에 시작한 bout. 기술 통계, n = 4마리 (p < 0.05 불가). "
                f"{DRUG_ORDER['kr']}."))
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)
    return res


def fig_beh_stimulus(N, lg, fn):
    font(lg)
    fig, axes = plt.subplots(2, 3, figsize=(FW, FH))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.885, bottom=0.07, wspace=0.22, hspace=0.36)   # 10-24: no title / footer
    res = {}
    for a, bn in zip(axes.ravel(), BEH):
        res[bn] = {}
        for i, st in enumerate(STIM):
            for drug, dx in (("vehicle", -0.19), ("SBI-553", 0.19)):
                V = N[(N.stim == st) & (N.drug == drug)].set_index("mouse").reindex(MO)[f"{bn} change"]
                a.bar(i + dx, V.mean(), width=0.36, color=CC[drug], alpha=0.45, edgecolor=CC[drug])
                for m in MO:
                    a.scatter(i + dx, V[m], marker=MK[m], s=26, color=MC[m], edgecolor="k", lw=0.5, zorder=3)
                res[bn][f"{st} | {drug} day"] = dict(V)
        a.axhline(0, color="0.3", lw=0.8)
        a.set_xticks(range(4), ["light", "mild", "pin", "heat"])
        a.set_title(bn, fontsize=13)
        lo, hi = a.get_ylim()
        m_ = max(abs(lo), abs(hi))
        a.set_ylim(-m_ * 1.05, m_ * 1.05)
    for a in axes[:, 0]:
        ylab(a, L(lg, "change in % of\ndeliveries (inj. - base)", "자극 대비 % 변화\n(주사 - 기준)"))
    h = [Patch(color=CC["vehicle"], alpha=0.55, label=L(lg, "vehicle day (bar = mean of 4 mice)", "vehicle 날 (막대 = 4마리 평균)")),
         Patch(color=CC["SBI-553"], alpha=0.55, label=L(lg, "SBI-553 day", "SBI-553 날"))]
    h += [Line2D([], [], marker=MK[m], color=MC[m], mec="k", mew=0.5, ls="", ms=7, label=f"{m} ({first_txt(m, lg)})") for m in MO]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.005, 0.998), ncol=6, frameon=False, fontsize=11.5, columnspacing=1.3)
    foot(fig, L(lg, "Per mouse, day and stimulus: % of the deliveries followed by the behaviour after the injection minus the baseline "
                    "session of that day (about 10-35 deliveries per stimulus and session, so single-mouse values are coarse). "
                    f"Descriptive, n = 4 mice. {DRUG_ORDER['en']}.",
                "마우스·날·자극마다: 주사 후 그 행동이 뒤따른 자극의 % - 그날 기준 세션 (자극·세션당 약 10-35회라 마우스별 값은 "
                f"거침). 기술 통계, n = 4마리. {DRUG_ORDER['kr']}."))
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)
    return res


def fig_beh_seconds(N, lg, fn):
    font(lg)
    fig, axes = plt.subplots(1, 4, figsize=(FW, FH))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.875, bottom=0.12, wspace=0.35)   # 10-24: no title / footer -> taller
    res = beh_lines(fig, axes, N, [(b, f"{b}_s change") for b in HELD], lg,
                    L(lg, "change in s per delivery (3 s after)\n(injection - same-day baseline)",
                      "자극당 초 변화 (자극 후 3 s)\n(주사 - 같은 날 기준)"))
    foot(fig, L(lg, "Per mouse and day: mean seconds of the behaviour in the 3 s after each delivery (0-3 s) after the injection "
                    f"minus the baseline session of that day. Descriptive, n = 4 mice. {DRUG_ORDER['en']}.",
                "마우스·날마다: 각 자극 후 3 s (0-3 s) 동안 그 행동의 평균 초, 주사 후 - 그날 기준 세션. 기술 통계, n = 4마리. "
                f"{DRUG_ORDER['kr']}."))
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)
    return res


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="reload the sessions (else the cache)")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    C = build_cache() if (a.rebuild or not os.path.isfile(CACHE)) else np.load(CACHE, allow_pickle=True)["C"][0]
    P = load_pairs()
    X = tracked_mats(C, P)
    Pm, Pa = pop(X), pop_all(C)
    f = lambda nm, lg: os.path.join(OUT, f"{nm}{'_kr' if lg == 'kr' else ''}.png")
    SUM = {}
    blk = pd.read_csv(os.path.join(OUT, "session_blocks.csv"))
    miss = {k: round(float(np.clip(SEG - (v["dur"] - BLOCK_S[3][0]), 0, SEG)), 1)
            for k, v in C.items()}                           # seconds of the last recorded block's rest not recorded

    # 1 per mouse
    segtab = {}
    for m in MO:
        for lg in ("en", "kr"):
            r = fig_mouse(m, X, C, P, lg, f(f"SESS_{m}", lg))
        sm = {c: seg_means(Pm[m][c]) for c in COND}
        segtab[m] = sm
        ordr = {c: C["|".join(map(str, cond_key(m, c)))]["order"] for c in COND}
        dv = sm["vi"]["stimulus blocks"] - sm["vb"]["stimulus blocks"]
        ds = sm["si"]["stimulus blocks"] - sm["sb"]["stimulus blocks"]
        n = len(P[m])
        SUM[f"SESS_{m}.png"] = dict(
            section="Q2 activity", priority="main",
            numbers=dict(n_tracked=n, recorded_block_order=ordr, population_mean_z_segments=sm,
                         stim_block_change_vehicle_day=dv, stim_block_change_sbi_day=ds,
                         early_deliveries_of_the_next_block=r["early_deliveries"]),
            title_en=f"{m}: the same {n} neurons in all four sessions, stimulus blocks in one order",
            title_kr=f"{m}: 같은 뉴런 {n}개를 네 세션 모두에서, 자극 블록을 같은 순서로",
            key_en=(f"{m}: stimulus-block population z changed {dv:+.2f} after vehicle and {ds:+.2f} after SBI-553 (each vs its own "
                    f"baseline)."),
            key_kr=f"{m}: 자극 블록 집단 z 변화 vehicle 후 {dv:+.2f}, SBI-553 후 {ds:+.2f} (각자 같은 날 기준 대비).",
            bottom_en=(f"We moved every session's 4 stimulus blocks to one order (light, mild, pin, heat; stimulus from the scoring file) "
                       f"and plotted the {n} tracked neurons, 1-s bins of z, the same rows in all 4 sessions. Descriptive."),
            bottom_kr=(f"모든 세션의 자극 블록 4개를 한 순서 (light, mild, pin, heat; 자극은 채점 파일)로 옮기고 추적 뉴런 {n}개를 1초 "
                       f"bin z로, 네 세션 같은 행으로 그림. 기술 통계."),
            notes_en=(f"{m} ({SEX[m]}), {order_txt(m, 'en')}. Recorded block orders: " +
                      "; ".join(f"{c} {' > '.join(o)}" for c, o in ordr.items()) +
                      ". Rows sorted by the mean z in the stimulus blocks of the two baselines. z = dF/F / robust SD (MAD) of the "
                      "session, so a session with less activity has a smaller SD (z is not an absolute scale across sessions). "
                      "First 15 s excluded (grey); the rest after the last recorded block is partly missing (session ended at "
                      f"{min(C[k]['dur'] for k in C if k.startswith(m)):.0f}-{max(C[k]['dur'] for k in C if k.startswith(m)):.0f} s)."),
            notes_kr=(f"{m} ({SEX[m]}), {order_txt(m, 'kr')}. 기록된 블록 순서: " +
                      "; ".join(f"{c} {' > '.join(o)}" for c, o in ordr.items()) +
                      ". 행 = 두 기준 세션 자극 블록의 평균 z 순. z = dF/F / 세션의 robust SD (MAD) - 활동이 적은 세션은 SD가 작음 "
                      "(z는 세션 간 절대 척도가 아님). 처음 15 s 제외 (회색); 마지막 블록 뒤 휴식은 일부 없음 (세션 종료 "
                      f"{min(C[k]['dur'] for k in C if k.startswith(m)):.0f}-{max(C[k]['dur'] for k in C if k.startswith(m)):.0f} s)."))
    # 1 average
    for lg in ("en", "kr"):
        ra = fig_average(Pm, P, lg, f("SESS_average", lg))
    mean4 = {c: seg_means(np.nanmean(np.array([Pm[m][c] for m in MO]), 0)) for c in COND}
    allneur = {m: {c: seg_means(Pa[m][c])["stimulus blocks"] for c in COND} for m in MO}
    SUM["SESS_average.png"] = dict(
        section="Q2 activity", priority="main",
        numbers=dict(per_mouse=segtab, mean_of_4=mean4, all_neurons_stim_blocks_check=allneur, color_range=ra["color_range"]),
        title_en="Population activity over the whole session, blocks in one order: every mouse and the mean",
        title_kr="세션 전체 집단 활동, 블록을 같은 순서로: 마우스별 + 평균",
        key_en=(f"Mean of 4 mice, stimulus blocks: baseline {mean4['vb']['stimulus blocks']:.2f} -> after vehicle "
                f"{mean4['vi']['stimulus blocks']:.2f} z; baseline {mean4['sb']['stimulus blocks']:.2f} -> after SBI-553 "
                f"{mean4['si']['stimulus blocks']:.2f} z."),
        key_kr=(f"4마리 평균, 자극 블록: 기준 {mean4['vb']['stimulus blocks']:.2f} -> vehicle 후 {mean4['vi']['stimulus blocks']:.2f} z; "
                f"기준 {mean4['sb']['stimulus blocks']:.2f} -> SBI-553 후 {mean4['si']['stimulus blocks']:.2f} z."),
        bottom_en=("Population mean z (1-s bins) of the 185 tracked neurons per mouse (55, 43, 22, 65), blocks rebuilt in one order "
                   "(light, mild, pin, heat), 4 session types; bottom row = mean of 4 mice. Descriptive."),
        bottom_kr=("마우스별 추적 뉴런 185개 (55, 43, 22, 65)의 집단 평균 z (1초 bin), 블록을 한 순서 (light, mild, pin, heat)로 "
                   "재구성, 세션 4종; 맨 아래 행 = 4마리 평균. 기술 통계."),
        notes_en=("Per mouse stimulus-block mean z (vb / vi / sb / si): " + "; ".join(
            f"{m} {segtab[m]['vb']['stimulus blocks']:.2f} / {segtab[m]['vi']['stimulus blocks']:.2f} / "
            f"{segtab[m]['sb']['stimulus blocks']:.2f} / {segtab[m]['si']['stimulus blocks']:.2f}" for m in MO) +
            ". Check with every neuron of each session (not only tracked): " + "; ".join(
            f"{m} {allneur[m]['vb']:.2f} / {allneur[m]['vi']:.2f} / {allneur[m]['sb']:.2f} / {allneur[m]['si']:.2f}" for m in MO) +
            ". z = dF/F / robust SD of each session (session-relative scale). " + DRUG_ORDER["en"] + "."),
        notes_kr=("마우스별 자극 블록 평균 z (vb / vi / sb / si): " + "; ".join(
            f"{m} {segtab[m]['vb']['stimulus blocks']:.2f} / {segtab[m]['vi']['stimulus blocks']:.2f} / "
            f"{segtab[m]['sb']['stimulus blocks']:.2f} / {segtab[m]['si']['stimulus blocks']:.2f}" for m in MO) +
            ". 각 세션 모든 뉴런으로 확인: " + "; ".join(
            f"{m} {allneur[m]['vb']:.2f} / {allneur[m]['vi']:.2f} / {allneur[m]['sb']:.2f} / {allneur[m]['si']:.2f}" for m in MO) +
            ". z = dF/F / 각 세션 robust SD (세션 상대 척도). " + DRUG_ORDER["kr"] + "."))
    for lg in ("en", "kr"):
        rn, vmax = fig_average_norm(Pm, P, lg, f("SESS_average_norm", lg))
    sb_v = np.array([segtab[m]["vi"]["stimulus blocks"] - segtab[m]["vb"]["stimulus blocks"] for m in MO])
    sb_s = np.array([segtab[m]["si"]["stimulus blocks"] - segtab[m]["sb"]["stimulus blocks"] for m in MO])
    ns_v = np.array([segtab[m]["vi"]["no stimulus"] - segtab[m]["vb"]["no stimulus"] for m in MO])
    ns_s = np.array([segtab[m]["si"]["no stimulus"] - segtab[m]["sb"]["no stimulus"] for m in MO])
    nlow = int((sb_s < sb_v).sum())
    SUM["SESS_average_norm.png"] = dict(
        section="Q2 activity", priority="main",
        numbers=dict(per_segment=rn, stim_blocks_vehicle_day=dict(zip(MO, sb_v)), stim_blocks_sbi_day=dict(zip(MO, sb_s)),
                     no_stim_vehicle_day=dict(zip(MO, ns_v)), no_stim_sbi_day=dict(zip(MO, ns_s)),
                     n_mice_sbi_lower_stim_blocks=nlow, color_range=[-vmax, vmax]),
        title_en="Whole session, normalized to the same-day baseline: vehicle day vs SBI-553 day",
        title_kr="세션 전체, 같은 날 기준으로 정규화: vehicle 날 vs SBI-553 날",
        key_en=(f"Stimulus blocks, after - baseline: vehicle day {sb_v.mean():+.2f}, SBI-553 day {sb_s.mean():+.2f} z (mean of 4); "
                f"lower on the SBI-553 day in {nlow}/4 mice."),
        key_kr=(f"자극 블록, 후 - 기준: vehicle 날 {sb_v.mean():+.2f}, SBI-553 날 {sb_s.mean():+.2f} z (4마리 평균); SBI-553 날이 "
                f"낮은 마우스 {nlow}/4."),
        bottom_en=("Population mean z of the tracked neurons (185; 55, 43, 22, 65 per mouse), injection minus same-day baseline per "
                   "1-s bin, blocks in one order; C = mean per segment. Descriptive, n = 4 mice."),
        bottom_kr=("추적 뉴런 (185; 마우스별 55, 43, 22, 65)의 집단 평균 z, 1초 bin마다 주사 - 같은 날 기준, 블록 한 순서; C = 구간별 "
                   "평균. 기술 통계, n = 4마리."),
        notes_en=("Stimulus blocks after - baseline per mouse (vehicle day / SBI-553 day): " + "; ".join(
            f"{m} {v:+.2f} / {s:+.2f}" for m, v, s in zip(MO, sb_v, sb_s)) + ". First 3 min without stimulus: " + "; ".join(
            f"{m} {v:+.2f} / {s:+.2f}" for m, v, s in zip(MO, ns_v, ns_s)) + f". {DRUG_ORDER['en']}; z is session-relative "
            "(dF/F / robust SD of each session)."),
        notes_kr=("마우스별 자극 블록 후 - 기준 (vehicle 날 / SBI-553 날): " + "; ".join(
            f"{m} {v:+.2f} / {s:+.2f}" for m, v, s in zip(MO, sb_v, sb_s)) + ". 자극 없는 처음 3분: " + "; ".join(
            f"{m} {v:+.2f} / {s:+.2f}" for m, v, s in zip(MO, ns_v, ns_s)) + f". {DRUG_ORDER['kr']}; z는 세션 상대 척도 "
            "(dF/F / 각 세션 robust SD)."))

    # 2 rates
    W = whole_tables()
    R, T = rate_data(W)
    T.round(5).to_csv(os.path.join(OUT, "rate_blocks.csv"), index=False)
    for lg in ("en", "kr"):
        fig_rate(R, T, lg, f("RATE_blocks", lg))
    tv, ts = T[T.drug == "vehicle"].set_index("mouse").reindex(MO), T[T.drug == "SBI-553"].set_index("mouse").reindex(MO)
    SUM["RATE_blocks.png"] = dict(
        section="Q2 activity", priority="main",
        numbers=T.to_dict("records"),
        title_en="Stimulus-block activity after the injection, both days: vehicle day vs SBI-553 day",
        title_kr="주사 후 자극 블록 활동, 두 날: vehicle 날 vs SBI-553 날",
        key_en=("Median change per neuron: vehicle day " + ", ".join(f"{m} {tv.at[m, 'change_pct']:+.0f}%" for m in MO) +
                "; SBI-553 day " + ", ".join(f"{m} {ts.at[m, 'change_pct']:+.0f}%" for m in MO) + "."),
        key_kr=("뉴런당 변화 중앙값: vehicle 날 " + ", ".join(f"{m} {tv.at[m, 'change_pct']:+.0f}%" for m in MO) +
                "; SBI-553 날 " + ", ".join(f"{m} {ts.at[m, 'change_pct']:+.0f}%" for m in MO) + "."),
        bottom_en=(f"Ca transients per min of still frames in the 4 stimulus blocks, every neuron of each day (vehicle day "
                   f"{int(tv.n_cells.sum())}, SBI-553 day {int(ts.n_cells.sum())}), baseline vs injection; Wilcoxon signed-rank per "
                   "mouse-day."),
        bottom_kr=(f"자극 블록 4개의 정지 프레임 분당 Ca transient, 그날의 모든 뉴런 (vehicle 날 {int(tv.n_cells.sum())}, SBI-553 날 "
                   f"{int(ts.n_cells.sum())}), 기준 vs 주사; 마우스-날별 Wilcoxon signed-rank."),
        notes_en=("Per mouse-day (median base -> inj., p): " + "; ".join(
            f"{r.mouse} {r.drug} day (Day {r.day}, n = {r.n_cells}) {r.base_median:.1f} -> {r.inj_median:.1f}, {pfmt(r.p_wilcoxon)}"
            for r in T.itertuples()) + ". Change % = 100 x (2^median log2((inj + 0.1)/(base + 0.1)) - 1), as the Day-1 slide. "
            "Blocks pooled weighted by the still minutes. " + "".join(
                f"FLOOR: {r.mouse} Day {r.day} has {r.n_zero_both}/{r.n_cells} neurons with 0 transients/min in the stimulus blocks "
                f"in both sessions (median 0), so its change (+0%) and its star rest on a few neurons. "
                for r in T.itertuples() if r.n_zero_both > r.n_cells / 2) + DRUG_ORDER["en"] + "."),
        notes_kr=("마우스-날별 (중앙값 기준 -> 주사, p): " + "; ".join(
            f"{r.mouse} {r.drug} 날 (Day {r.day}, n = {r.n_cells}) {r.base_median:.1f} -> {r.inj_median:.1f}, {pfmt(r.p_wilcoxon)}"
            for r in T.itertuples()) + ". 변화 % = 100 x (2^중앙값 log2((주사 + 0.1)/(기준 + 0.1)) - 1), Day-1 슬라이드와 같음. "
            "블록은 정지 시간 가중 합침. " + "".join(
                f"바닥 효과: {r.mouse} Day {r.day}는 뉴런 {r.n_zero_both}/{r.n_cells}개가 두 세션 모두 자극 블록 transient 0 / 분 "
                f"(중앙값 0) - 변화 (+0%)와 별표는 소수 뉴런에 의존. "
                for r in T.itertuples() if r.n_zero_both > r.n_cells / 2) + DRUG_ORDER["kr"] + "."))
    Tk = rate_tracked(R, P)
    Tk.round(5).to_csv(os.path.join(OUT, "rate_blocks_tracked.csv"), index=False)
    for lg in ("en", "kr"):
        rr = fig_rate_norm(Tk, lg, f("RATE_blocks_norm", lg))
    TA = Tk[Tk.track_class == "A"]
    pA = wil(TA.sbi_log2, TA.veh_log2)
    nlow = sum(rr[m]["diff_median"] < 0 for m in MO)
    SUM["RATE_blocks_norm.png"] = dict(
        section="Q2 activity", priority="main",
        numbers=dict(per_group=rr, class_A_only=dict(n=len(TA), diff_median=float((TA.sbi_log2 - TA.veh_log2).median()), p_wilcoxon=pA)),
        title_en="Stimulus-block activity, baseline-normalized, the same neurons: vehicle day vs SBI-553 day",
        title_kr="자극 블록 활동, 기준 정규화, 같은 뉴런: vehicle 날 vs SBI-553 날",
        key_en=(f"Same neurons, SBI-553 day minus vehicle day (median log2): " +
                ", ".join(f"{m} {rr[m]['diff_median']:+.2f}" for m in MO) + f"; lower on the SBI-553 day in {nlow}/4 mice."),
        key_kr=(f"같은 뉴런, SBI-553 날 변화 - vehicle 날 변화 (중앙값 log2): " +
                ", ".join(f"{m} {rr[m]['diff_median']:+.2f}" for m in MO) + f"; SBI-553 날이 낮은 마우스 {nlow}/4."),
        bottom_en=(f"Per tracked neuron (n = {len(Tk)}), log2((inj + 0.1)/(base + 0.1)) of still-frame transients/min in the stimulus "
                   "blocks on each day; Wilcoxon signed-rank paired by neuron per mouse (pooled = exploratory)."),
        bottom_kr=(f"추적 뉴런 (n = {len(Tk)})마다 날별 자극 블록 정지 프레임 분당 transient의 log2((주사 + 0.1)/(기준 + 0.1)); 마우스별 "
                   "뉴런 짝지은 Wilcoxon signed-rank (합침 = 탐색적)."),
        notes_en=("Per group (n, vehicle-day median, SBI-day median, median of the difference, p): " + "; ".join(
            f"{m} n = {v['n']}, {v['veh_median']:+.2f}, {v['sbi_median']:+.2f}, {v['diff_median']:+.2f}, {pfmt(v['p_wilcoxon'])}"
            for m, v in rr.items()) + f". Class A only (same neuron): n = {len(TA)}, {pfmt(pA)}. " + "".join(
                f"FLOOR: {m} SBI-553 day: {nz}/{len(G)} tracked neurons 0 transients/min in both sessions (log2 = 0), so its "
                f"comparison cannot show a drop; without {m}, SBI-553 day lower in {sum(rr[x]['diff_median'] < 0 for x in MO if x != m)}/3 mice. "
                for m, G, nz in [(m, Tk[Tk.mouse == m], int(((Tk.mouse == m) & (Tk.sbi_base == 0) & (Tk.sbi_inj == 0)).sum())) for m in MO]
                if nz > len(G) / 2) + DRUG_ORDER["en"] + "."),
        notes_kr=("그룹별 (n, vehicle 날 중앙값, SBI 날 중앙값, 차이 중앙값, p): " + "; ".join(
            f"{m} n = {v['n']}, {v['veh_median']:+.2f}, {v['sbi_median']:+.2f}, {v['diff_median']:+.2f}, {pfmt(v['p_wilcoxon'])}"
            for m, v in rr.items()) + f". A 등급만 (같은 뉴런): n = {len(TA)}, {pfmt(pA)}. " + "".join(
                f"바닥 효과: {m} SBI-553 날 추적 뉴런 {nz}/{len(G)}개가 두 세션 모두 0 / 분 (log2 = 0) - 감소를 보일 수 없음; {m} 빼면 "
                f"SBI-553 날이 낮은 마우스 {sum(rr[x]['diff_median'] < 0 for x in MO if x != m)}/3. "
                for m, G, nz in [(m, Tk[Tk.mouse == m], int(((Tk.mouse == m) & (Tk.sbi_base == 0) & (Tk.sbi_inj == 0)).sum())) for m in MO]
                if nz > len(G) / 2) + DRUG_ORDER["kr"] + "."))

    # 3 behaviour
    N = beh_norm()
    N.round(5).to_csv(os.path.join(OUT, "beh_norm.csv"), index=False)
    for lg in ("en", "kr"):
        rb = fig_beh_delivery(N, lg, f("BEH_norm_delivery", lg))
    SUM["BEH_norm_delivery.png"] = dict(
        section="Q1 behaviour", priority="main", numbers=rb,
        title_en="Behaviour per delivery, normalized to the same-day baseline: vehicle day vs SBI-553 day",
        title_kr="자극당 행동, 같은 날 기준 정규화: vehicle 날 vs SBI-553 날",
        key_en=("Change from baseline, mean of 4 (vehicle / SBI-553 day, % of deliveries): " + ", ".join(
            f"{b} {rb[b]['mean_vehicle']:+.0f} / {rb[b]['mean_sbi']:+.0f}" for b in ("withdrawal", "flinch", "movement")) + "."),
        key_kr=("기준 대비 변화, 4마리 평균 (vehicle / SBI-553 날, 자극 대비 %): " + ", ".join(
            f"{b} {rb[b]['mean_vehicle']:+.0f} / {rb[b]['mean_sbi']:+.0f}" for b in ("withdrawal", "flinch", "movement")) + "."),
        bottom_en=("Per mouse and day: % of deliveries followed by each behaviour after the injection minus the same-day baseline "
                   "(Day 1 + Day 2); one line per mouse. Descriptive, n = 4 mice; drug order confounded with day."),
        bottom_kr=("마우스·날마다: 주사 후 각 행동이 뒤따른 자극 % - 같은 날 기준 (Day 1 + Day 2); 선 = 마우스. 기술 통계, n = 4마리; "
                   "투여 순서와 날이 겹침."),
        notes_en=("Per behaviour (mean vehicle day / SBI-553 day; mice lower on the SBI-553 day): " + "; ".join(
            f"{b} {v['mean_vehicle']:+.1f} / {v['mean_sbi']:+.1f} ({v['n_mice_sbi_lower']}/4)" for b, v in rb.items()) +
            ". " + DRUG_ORDER["en"] + ". Withdrawal / flinch within 2 s of the delivery (or until the next); held behaviours = bout "
            "starting 0-3 s after."),
        notes_kr=("행동별 (평균 vehicle 날 / SBI-553 날; SBI-553 날이 낮은 마우스): " + "; ".join(
            f"{b} {v['mean_vehicle']:+.1f} / {v['mean_sbi']:+.1f} ({v['n_mice_sbi_lower']}/4)" for b, v in rb.items()) +
            ". " + DRUG_ORDER["kr"] + ". Withdrawal / flinch = 자극 후 2 s 이내 (또는 다음 자극까지); 지속 행동 = 자극 후 0-3 s에 "
            "시작한 bout."))
    for lg in ("en", "kr"):
        rs = fig_beh_stimulus(N, lg, f("BEH_norm_stimulus", lg))
    mstim = {b: {st: (float(np.nanmean(list(rs[b][f"{st} | vehicle day"].values()))),
                      float(np.nanmean(list(rs[b][f"{st} | SBI-553 day"].values())))) for st in STIM} for b in BEH}
    SUM["BEH_norm_stimulus.png"] = dict(
        section="Q1 behaviour", priority="backup", numbers=dict(per_mouse=rs, mean_of_4_vehicle_sbi=mstim),
        title_en="Behaviour per stimulus, normalized to the same-day baseline: vehicle day vs SBI-553 day",
        title_kr="자극 종류별 행동, 같은 날 기준 정규화: vehicle 날 vs SBI-553 날",
        key_en=("Withdrawal change from baseline (vehicle / SBI-553 day, mean of 4): " + ", ".join(
            f"{st.split()[0].lower()} {mstim['withdrawal'][st][0]:+.0f} / {mstim['withdrawal'][st][1]:+.0f}" for st in STIM) + " %."),
        key_kr=("Withdrawal 기준 대비 변화 (vehicle / SBI-553 날, 4마리 평균): " + ", ".join(
            f"{st.split()[0].lower()} {mstim['withdrawal'][st][0]:+.0f} / {mstim['withdrawal'][st][1]:+.0f}" for st in STIM) + " %."),
        bottom_en=("Per mouse, day and stimulus: % of deliveries followed by each behaviour, injection minus same-day baseline; bars = "
                   "mean of 4 mice, markers = mice. Descriptive; drug order confounded with day."),
        bottom_kr=("마우스·날·자극마다: 각 행동이 뒤따른 자극 %, 주사 - 같은 날 기준; 막대 = 4마리 평균, 표식 = 마우스. 기술 통계; "
                   "투여 순서와 날이 겹침."),
        notes_en="Mean of 4 (vehicle day, SBI-553 day) per behaviour and stimulus: " + "; ".join(
            f"{b}: " + ", ".join(f"{st.split()[0].lower()} {v[0]:+.0f}/{v[1]:+.0f}" for st, v in mstim[b].items()) for b in BEH) +
            ". " + DRUG_ORDER["en"] + ".",
        notes_kr="행동·자극별 4마리 평균 (vehicle 날, SBI-553 날): " + "; ".join(
            f"{b}: " + ", ".join(f"{st.split()[0].lower()} {v[0]:+.0f}/{v[1]:+.0f}" for st, v in mstim[b].items()) for b in BEH) +
            ". " + DRUG_ORDER["kr"] + ".")
    for lg in ("en", "kr"):
        rsec = fig_beh_seconds(N, lg, f("BEH_norm_seconds", lg))
    SUM["BEH_norm_seconds.png"] = dict(
        section="Q1 behaviour", priority="backup", numbers=rsec,
        title_en="Seconds of each held behaviour after a delivery, normalized to the same-day baseline",
        title_kr="자극 후 지속 행동 시간, 같은 날 기준 정규화",
        key_en=("Change in s per delivery (veh / SBI day, mean of 4): " + ", ".join(
            f"{b.split('/')[0]} {v['mean_vehicle']:+.2f}/{v['mean_sbi']:+.2f}" for b, v in rsec.items()) + "."),
        key_kr=("자극당 초 변화 (veh / SBI 날, 4마리 평균): " + ", ".join(
            f"{b.split('/')[0]} {v['mean_vehicle']:+.2f}/{v['mean_sbi']:+.2f}" for b, v in rsec.items()) + "."),
        bottom_en=("Per mouse and day: mean seconds of each held behaviour in the 3 s after a delivery, injection minus same-day "
                   "baseline (Day 1 + Day 2); one line per mouse. Descriptive, n = 4 mice."),
        bottom_kr=("마우스·날마다: 자극 후 3 s 동안 각 지속 행동의 평균 초, 주사 - 같은 날 기준 (Day 1 + Day 2); 선 = 마우스. 기술 통계, "
                   "n = 4마리."),
        notes_en="Per behaviour (mice lower on the SBI-553 day): " + "; ".join(
            f"{b} {v['n_mice_sbi_lower']}/4" for b, v in rsec.items()) + ". " + DRUG_ORDER["en"] + ".",
        notes_kr="행동별 (SBI-553 날이 낮은 마우스): " + "; ".join(
            f"{b} {v['n_mice_sbi_lower']}/4" for b, v in rsec.items()) + ". " + DRUG_ORDER["kr"] + ".")

    # data checks
    SUM["_checks"] = dict(
        block_stimulus_from_deliveries_agrees_with_stimNames=bool(blk.agrees.all()),
        deliveries_in_the_30s_rest_after_a_block=blk[blk.n_in_rest > 0][["mouse", "day", "phase", "block", "n_in_rest"]].to_dict("records"),
        of_these_early_deliveries_of_the_next_block={k: [(st, round(e, 1)) for st, e in foreign(v)] for k, v in C.items() if foreign(v)},
        deliveries_outside_every_block_or_rest={k: int(v["n_out"]) for k, v in C.items() if v["n_out"]},
        session_end_s={k: round(v["dur"], 1) for k, v in C.items()},
        last_rest_seconds_missing=miss,
        n_tracked={m: len(P[m]) for m in MO},
        rate_floor_mouse_days=[dict(mouse=r.mouse, day=r.day, n_zero_both=r.n_zero_both, n=r.n_cells, base_median=r.base_median,
                                    inj_median=r.inj_median) for r in T.itertuples() if r.n_zero_both > r.n_cells / 2])
    for k, v in SUM.items():
        if k.startswith("_"):
            continue
        for fld in ("key_en", "key_kr"):
            if len(v[fld]) > 150:
                print(f"  NOTE {k} {fld} has {len(v[fld])} characters")
        for fld in ("bottom_en", "bottom_kr"):
            if len(v[fld]) > 230:
                print(f"  NOTE {k} {fld} has {len(v[fld])} characters")
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(jconv(SUM), fh, ensure_ascii=False, indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
