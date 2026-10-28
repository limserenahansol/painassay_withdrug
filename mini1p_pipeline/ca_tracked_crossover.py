"""ca_tracked_crossover.py - SBI-553 vs vehicle in the SAME neurons, baseline-normalized (2026-10-23, user: 'for all longitudinal
comparison of neuron z-score we should use only co-existing neurons ... track each neuron ... baseline norm vehicle vs SBI';
'% responsive UP / DOWN can use every neuron of each day').

1  Same neurons on both days (analysis\\crossday\\<m>_pairs.csv: 185 neurons; class A 'same neuron' 101 + B 'probably the same'
   84; A only = sensitivity). Per tracked neuron and event: the response r of the event analysis (cells_classes.csv: response-
   window mean z minus baseline-window mean z, trial mean; the best window of each event) in the four sessions of the crossover:
   vb = vehicle-day baseline, vi = after vehicle, sb = SBI-553-day baseline, si = after SBI-553.
     after-injection difference   si - vi                      (the same neuron after SBI-553 vs after vehicle)
     baseline-normalized          d = (si - sb) - (vi - vb)    (each injection minus the baseline of its own day)
   Tests: Wilcoxon signed-rank paired by neuron (neurons pooled over 4 mice = exploratory) and, as the check that allows for the
   mouse, a linear mixed model  value ~ 1 + (1 | mouse)  (statsmodels MixedLM, REML; 4 groups, so approximate); per-mouse
   medians are shown (4 mice: no test across mice can reach p < 0.05).
2  % responsive, every neuron of each day: per neuron the change from the same-day baseline (UP: +1 gained, 0 same, -1 lost;
   the same for DOWN); vehicle-day neurons vs SBI-553-day neurons (Mann-Whitney U); per mouse the change in % (markers).

Figures (analysis\\tracked_crossover, EN and _kr): TX0_summary, TX1_heat_<ev> (heatmaps: the same neurons, the same row order, in
all four sessions), TX2_neurons_<ev> (each neuron through the four sessions + baseline-normalized vehicle vs SBI-553),
TX3_responsive_norm (% UP / DOWN change from the same-day baseline). Tables tracked_responses.csv, tracked_tests.csv,
responsive_norm.csv; tracked_summary.json.

  python ca_tracked_crossover.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401
import pairstars as PS  # noqa: E402  (every star shows the pair it compares)
import winwords as WW  # noqa: E402
import figfoot  # noqa: E402  (10-24: methods text -> <png>.notes.txt -> slide notes)
from matplotlib.patches import Patch  # noqa: E402

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
EA = os.path.join(B, "analysis", os.environ.get("CA_EVENT_OUT", "event_analysis"))
CD = os.path.join(B, "analysis", "crossday")
CACHE = os.environ.get("PSTH_CACHE", os.path.join(B, "analysis", "new_decks", "psth_cache_win3_tf.npz"))   # 10-24: variants
OUT = os.path.join(B, "analysis", "tracked_crossover")
MO = ["5941", "5939", "5937", "1191"]
SBI_DAY = {"5941": 1, "5939": 1, "5937": 2, "1191": 2}
CC = {"baseline": "#7f8c8d", "vehicle": "#2471a3", "SBI-553": "#c0392b"}
MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#34495e"}
MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
EVENTS = ["Light touch", "Mild touch", "Pin prick", "Heat", "withdrawal", "flinch", "movement onset"]
COND = ["vb", "vi", "sb", "si"]
SEED = 20261023
# 10-24: the deck shows a 15 x 6.3 in figure at ~0.83x, so 12 pt here = ~10 pt on the slide (rule: >= 9 pt)
plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13.5, "axes.labelsize": 12.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12, "legend.title_fontsize": 12, "axes.spines.top": False, "axes.spines.right": False})
FIG = (15, 6.3)                                                # = the 12.4 x 5.2 in picture box of the deck
VL1 = 0.5                                                      # 10-24: one fixed colour range on every event slide (display
VL4, VL4D = 0.8, 1.0                                           # only): TX1 +-0.5 z, TX4 changes +-0.8 z, TX4 difference +-1 z
MEDC = "0.15"                                                  # TX0 median + CI: neutral (red / blue mean the drug elsewhere)
XT = [-1, 0, 1, 2, 3]                                       # heatmap time ticks without the edge ticks ('4' met '-2')

TXT = {
    "vb": ("vehicle day\nbaseline", "vehicle 날\n기준"), "vi": ("after\nvehicle", "vehicle\n후"),
    "sb": ("SBI-553 day\nbaseline", "SBI-553 날\n기준"), "si": ("after\nSBI-553", "SBI-553\n후"),
    "vb1": ("vehicle day: baseline", "vehicle 날: 기준"), "vi1": ("after vehicle", "vehicle 후"),
    "sb1": ("SBI-553 day: baseline", "SBI-553 날: 기준"), "si1": ("after SBI-553", "SBI-553 후"),
    "time": ("time from event (s)", "사건부터 시간 (s)"),
    "zc": ("z (dF/F / robust SD),\nminus the 3 s before", "z (dF/F / robust SD),\n직전 3 s 평균을 뺌"),
    "rows": ("neurons (1 row = 1 neuron), grouped by mouse", "뉴런 (1 행 = 뉴런 1개), 마우스별로 묶음"),
    "bar": ("response window {a:g}-{b:g} s", "반응 창 {a:g}-{b:g} s"),
    "t0": ("event (t = 0)", "사건 (t = 0)"),
    "cb4": ("after - same-day baseline (z)", "후 - 같은 날 기준 (z)"),
    "cb4d": ("SBI-553 change - vehicle change (z)", "SBI-553 변화 - vehicle 변화 (z)"),
    "dv": ("vehicle day: after - baseline (z)", "vehicle 날: 후 - 기준 (z)"),
    "ds": ("SBI-553 day: after - baseline (z)", "SBI-553 날: 후 - 기준 (z)"),
    "A": ("A  after SBI-553 minus after vehicle\n(same neurons, no normalization)", "A  SBI-553 후 - vehicle 후\n(같은 뉴런, 정규화 없음)"),
    "Bn": ("B  baseline-normalized:\n(SBI-553 - its baseline) - (vehicle - its baseline)", "B  기준 정규화:\n(SBI-553 - 그날 기준) - (vehicle - 그날 기준)"),
    "diff": ("difference (z), median of neurons +- 95% CI", "차이 (z), 뉴런 중앙값 +- 95% CI"),
    "mouse_med": ("marker = median of one mouse", "표식 = 마우스 하나의 중앙값"),
    "pair_A": ("after SBI-553 vs\nafter vehicle", "SBI-553 후 vs\nvehicle 후"),
    "pair_B": ("change SBI-553 vs\nchange vehicle", "SBI-553 변화 vs\nvehicle 변화"),
    "neur": ("neurons", "뉴런"),
}


def T_(k, lg):
    return TXT[k][1 if lg == "kr" else 0]


def font(lg):
    if lg == "kr":
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def wil(x, y=None):
    x = np.asarray(x, float)
    d = x if y is None else x - np.asarray(y, float)
    d = d[np.isfinite(d)]
    if len(d) < 6 or np.all(d == 0):
        return np.nan
    return float(wilcoxon(d, zero_method="wilcox").pvalue)


def mouse_t(v, mice):
    """mouse-level check (10-23 review: the mixed model was degenerate with 4 mice): one-sample t-test of the 4 mouse means
    against 0 (df = 3). Returns (mean of the mouse means, p)."""
    from scipy.stats import ttest_1samp
    df = pd.DataFrame(dict(v=np.asarray(v, float), mouse=np.asarray(mice)))
    mm = df[np.isfinite(df.v)].groupby("mouse").v.mean()
    if len(mm) < 3 or mm.std() == 0:
        return np.nan, np.nan
    return float(mm.mean()), float(ttest_1samp(mm, 0).pvalue)


SBI_FIRST = [m for m in MO if SBI_DAY[m] == 1]                 # 5941, 5939 (SBI-553 on Day 1); 5937, 1191 got it on Day 2


def boot_ci(v, n=4000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(SEED)
    m = np.median(v[rng.integers(0, len(v), (n, len(v)))], 1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def foot(fig, txt, lg=None, fs=11.5):
    """10-24 (user: details in the notes, not on the slide): the methods text goes to <png>.notes.txt (figfoot -> slide notes);
    only the one-line star key stays in the figure (11.5 pt = ~9.5 pt on the slide). Returns the number of lines drawn (1)."""
    figfoot.note(fig, txt, lg, fs=fs)
    return 1


def heat_key(fig, w0, lg, y=0.995):
    """the legend of the heatmap marks (10-24 audit: the black bar was never explained): black bar = response window,
    dashed line = the event"""
    h = [Patch(color="k", label=T_("bar", lg).format(a=w0[2], b=w0[3])),
         Line2D([], [], color="k", ls="--", lw=1, label=T_("t0", lg))]
    fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.995, y), ncol=2, frameon=False, handlelength=1.6,
               columnspacing=1.4)


def mouse_rows(a, Z, lg, first):
    """horizontal lines between the mice; on the first panel the mouse names (left of the axis) and the y label"""
    w_in = a.get_position().width * a.figure.get_figwidth()  # offsets in inches, so every layout gets the same margin
    y0 = 0
    for m in MO:
        n = int((Z.mouse == m).sum())
        if n:
            if y0:
                a.axhline(y0, color="k", lw=1.2)
            if first:
                a.text(-0.09 / w_in, y0 + n / 2, f"{m}\n(n = {n})", ha="right", va="center", fontsize=12, color=MC[m],
                       fontweight="bold", transform=a.get_yaxis_transform())
            y0 += n
    a.set_yticks([])
    if first:
        a.set_ylabel(T_("rows", lg), fontsize=12)
        a.yaxis.set_label_coords(-1.0 / w_in, 0.5)


def ptxt(p):
    return "n.a." if not np.isfinite(p) else "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


# ------------------------------------------------------------------------------------------------ data
def load():
    T = pd.read_csv(os.path.join(EA, "cells_classes.csv")); T["mouse"] = T.mouse.astype(str)
    R = T.set_index(["mouse", "day", "phase", "event", "cell"])
    rows = []
    for m in MO:
        P = pd.read_csv(os.path.join(CD, f"{m}_pairs.csv"))
        sd, vd = SBI_DAY[m], 3 - SBI_DAY[m]
        for p in P.itertuples():
            ids = {1: int(p.day1_id), 2: int(p.day2_id)}
            for ev in EVENTS:
                r = dict(mouse=m, track_class=p.track_class, day1_id=ids[1], day2_id=ids[2], event=ev, veh_day=vd, sbi_day=sd,
                         veh_id=ids[vd], sbi_id=ids[sd])
                for c, (d, ph) in zip(COND, ((vd, "pre"), (vd, "post"), (sd, "pre"), (sd, "post"))):
                    k = (m, d, ph, ev, ids[d])
                    r[c] = float(R.resp.get(k, np.nan)) if k in R.index else np.nan
                    r["cls_" + c] = R.cls.get(k, "") if k in R.index else ""
                rows.append(r)
    X = pd.DataFrame(rows)
    X["after_diff"] = X.si - X.vi
    X["chg_veh"] = X.vi - X.vb
    X["chg_sbi"] = X.si - X.sb
    X["d_norm"] = X.chg_sbi - X.chg_veh
    return T, X


def tests(X, which="A+B"):
    out = []
    for ev in EVENTS:
        Z = X[(X.event == ev) & X[COND].notna().all(1)]
        if which == "A":
            Z = Z[Z.track_class == "A"]
        if not len(Z):
            out.append(dict(event=ev, tracking=which, n=0)); continue
        r = dict(event=ev, tracking=which, n=len(Z), n_mice=Z.mouse.nunique())
        for c in COND:
            r[f"med_{c}"] = float(Z[c].median())
        for nm, col in (("after", "after_diff"), ("norm", "d_norm")):
            v = Z[col].to_numpy()
            r[f"med_{nm}"] = float(np.median(v))
            r[f"lo_{nm}"], r[f"hi_{nm}"] = boot_ci(v)
            r[f"p_{nm}"] = wil(v)
            r[f"mouse_{nm}"], r[f"p_mouse_{nm}"] = mouse_t(v, Z.mouse)
            # balanced over the drug order: SBI-553 first (Day 1) vs second (Day 2); drug = mean of the two group medians,
            # day = (Day 2 - Day 1) estimate = half their difference (after - after: SBI-first = D1 - D2, SBI-second = D2 - D1)
            g1, g2 = Z[Z.mouse.isin(SBI_FIRST)][col], Z[~Z.mouse.isin(SBI_FIRST)][col]
            r[f"g1_{nm}"], r[f"g2_{nm}"] = float(g1.median()), float(g2.median())
            r[f"n_g1"], r[f"n_g2"] = len(g1), len(g2)
            r[f"p_g1_{nm}"], r[f"p_g2_{nm}"] = wil(g1), wil(g2)
            r[f"bal_{nm}"] = (r[f"g1_{nm}"] + r[f"g2_{nm}"]) / 2
            r[f"day_{nm}"] = (r[f"g2_{nm}"] - r[f"g1_{nm}"]) / 2
            mm = Z.groupby("mouse")[col].median()
            r[f"mice_{nm}"] = {k: float(x) for k, x in mm.items()}
            r[f"mice_neg_{nm}"] = int((mm < 0).sum())
        r["p_vb_vi"], r["p_sb_si"] = wil(Z.vi, Z.vb), wil(Z.si, Z.sb)
        r["p_vb_sb"] = wil(Z.sb, Z.vb)
        r["med_chg_veh"], r["med_chg_sbi"] = float(Z.chg_veh.median()), float(Z.chg_sbi.median())
        out.append(r)
    return pd.DataFrame(out)


def responsive_norm(T):
    """every neuron of each day: per neuron change from the same-day baseline of UP (+1 / 0 / -1) and of DOWN"""
    rows, cells = [], []
    for ev in EVENTS:
        for m in MO:
            for d in (1, 2):
                drug = "SBI-553" if SBI_DAY[m] == d else "vehicle"
                a = T[(T.mouse == m) & (T.day == d) & (T.event == ev) & (T.phase == "pre")].set_index("cell").cls
                b = T[(T.mouse == m) & (T.day == d) & (T.event == ev) & (T.phase == "post")].set_index("cell").cls
                c = a.index.intersection(b.index)
                if not len(c):
                    continue
                up = (b[c] == "activated").astype(int) - (a[c] == "activated").astype(int)
                dn = (b[c] == "inhibited").astype(int) - (a[c] == "inhibited").astype(int)
                cells.append(pd.DataFrame(dict(event=ev, mouse=m, day=d, drug=drug, cell=c, chg_up=up.values, chg_down=dn.values,
                                               base_up=(a[c] == "activated").values, inj_up=(b[c] == "activated").values,
                                               base_down=(a[c] == "inhibited").values, inj_down=(b[c] == "inhibited").values)))
    C = pd.concat(cells)
    for ev in EVENTS:
        for k in ("up", "down"):
            Z = C[C.event == ev]
            v, s = Z[Z.drug == "vehicle"][f"chg_{k}"], Z[Z.drug == "SBI-553"][f"chg_{k}"]
            p = float(mannwhitneyu(s, v).pvalue) if len(v) and len(s) and (v.std() > 0 or s.std() > 0) else np.nan
            mm = Z.groupby(["mouse", "drug"])[f"chg_{k}"].mean().unstack() * 100
            dd = {}
            for d in (1, 2):
                vd_, sd_ = Z[(Z.drug == "vehicle") & (Z.day == d)][f"chg_{k}"], Z[(Z.drug == "SBI-553") & (Z.day == d)][f"chg_{k}"]
                dd[f"chg_veh_d{d}"], dd[f"chg_sbi_d{d}"] = 100 * vd_.mean(), 100 * sd_.mean()
                dd[f"p_d{d}"] = (float(mannwhitneyu(sd_, vd_).pvalue) if len(vd_) and len(sd_) and (vd_.std() > 0 or sd_.std() > 0)
                                 else np.nan)
            rows.append(dict(event=ev, kind=k, n_veh=len(v), n_sbi=len(s), chg_veh=100 * v.mean(), chg_sbi=100 * s.mean(), **dd,
                             base_veh=100 * Z[Z.drug == "vehicle"][f"base_{k}"].mean(), base_sbi=100 * Z[Z.drug == "SBI-553"][f"base_{k}"].mean(),
                             inj_veh=100 * Z[Z.drug == "vehicle"][f"inj_{k}"].mean(), inj_sbi=100 * Z[Z.drug == "SBI-553"][f"inj_{k}"].mean(),
                             p=p, mice={m: dict(vehicle=float(mm.loc[m, "vehicle"]) if "vehicle" in mm and m in mm.index else np.nan,
                                                sbi=float(mm.loc[m, "SBI-553"]) if "SBI-553" in mm and m in mm.index else np.nan) for m in MO},
                             mice_lower=int(sum(1 for m in MO if m in mm.index and mm.loc[m].get("SBI-553", np.nan) < mm.loc[m].get("vehicle", np.nan)))))
    return C, pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ figures
def fig_summary(S, lg, fn):
    font(lg)
    fig, ax = plt.subplots(1, 2, figsize=FIG, sharey=True)
    y = np.arange(len(EVENTS))[::-1]
    for a, nm, tk, pk in ((ax[0], "after", "A", "pair_A"), (ax[1], "norm", "Bn", "pair_B")):
        a.axvline(0, color="0.5", lw=1)
        for yi, r in zip(y, S.itertuples()):
            if not r.n:
                continue
            col = MEDC                                         # 10-25: one neutral colour (red / blue = drug elsewhere)
            a.plot([getattr(r, f"lo_{nm}"), getattr(r, f"hi_{nm}")], [yi, yi], color=col, lw=3, solid_capstyle="butt")
            a.plot(getattr(r, f"med_{nm}"), yi, "o", color=col, ms=9, zorder=4)
            a.scatter(getattr(r, f"bal_{nm}"), yi - 0.24, marker="D", s=40, facecolor="white", edgecolor="k", lw=1.3, zorder=5)
            for j, m in enumerate(MO):
                v = getattr(r, f"mice_{nm}").get(m, np.nan)
                if np.isfinite(v):
                    a.scatter(v, yi + 0.28, marker=MK[m], s=34, color=MC[m], edgecolor="white", lw=0.5, zorder=5)
        lim = np.nanmax(np.abs(np.r_[S[f"lo_{nm}"], S[f"hi_{nm}"], [v for d_ in S[f"mice_{nm}"] for v in d_.values()]]))
        lim = 1.1 * lim if np.isfinite(lim) and lim > 0 else 0.1
        a.set_xlim(-lim, lim * 2.05)
        a.axvspan(lim * 1.12, lim * 2.05, color="0.96", lw=0)
        xs = lim * 1.585                                       # 10-24: the compared pair is written once, as the column header
        for yi, r in zip(y, S.itertuples()):
            if r.n:
                st = PS.star(getattr(r, f"p_{nm}"))
                a.text(xs, yi, st, ha="center", va="center", fontsize=13 if st != "ns" else 12.5,
                       fontweight="bold" if st not in ("ns", "n.a.") else "normal", zorder=6)
        a.text(xs, len(EVENTS) - 0.42, T_(pk, lg), ha="center", va="bottom", fontsize=12, linespacing=1.1, zorder=6)
        a.set_xticks([t for t in a.get_xticks() if -lim <= t <= lim])
        a.set_title(T_(tk, lg), loc="left", fontsize=13.5)
        a.set_xlabel(T_("diff", lg))
        a.set_ylim(-0.7, len(EVENTS) + 0.55)
    ax[0].set_yticks(y); ax[0].set_yticklabels([f"{ev}\n(n = {n})" for ev, n in zip(EVENTS, S.n)], fontsize=12.5)
    h = [Line2D([], [], marker=MK[m], ls="", color=MC[m], ms=8, label=m) for m in MO]
    h += [Line2D([], [], color=MEDC, lw=3, marker="o", ms=8,
                 label="all neurons: median +- 95% CI" if lg != "kr" else "모든 뉴런: 중앙값 +- 95% CI"),
          Line2D([], [], color="k", marker="D", mfc="white", ls="", label="balanced over drug order" if lg != "kr" else "투여 순서 균형 추정")]
    fig.legend(handles=h, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.53, 1.0), handlelength=1.6,
               columnspacing=1.1, title=T_("mouse_med", lg))
    foot(fig, ("Neurons followed on both days. Drug order: SBI-553 on Day 1 in 5941, 5939, on Day 2 in 5937, 1191, so the drug "
               "comparison is also a day comparison; the hollow diamond averages the two drug-order groups (balanced). Stars (right column): "
               "Wilcoxon signed-rank paired by neuron; A = after SBI-553 vs after vehicle, B = the change from its own baseline on the "
               "SBI-553 day vs on the vehicle day. " + PS.KEY)
         if lg != "kr" else ("두 날 모두 추적한 뉴런. 투여 순서: 5941·5939는 Day 1, 5937·1191은 Day 2에 SBI-553 - 약물 비교가 날 비교이기도 함; "
                             "빈 마름모 = 두 투여 순서 그룹의 평균 (균형). 별표 (오른쪽 열): 뉴런별 짝지은 Wilcoxon signed-rank; A = SBI-553 후 "
                             "vs vehicle 후, B = SBI-553 날 기준 대비 변화 vs vehicle 날 기준 대비 변화. " + PS.KEY), lg)
    fig.tight_layout(rect=(0, 0.035, 1, 0.885), w_pad=2.0)
    fig.savefig(fn, dpi=150); plt.close(fig)


def heat_rows(X, D, ev):
    """tracked neurons with all four sessions for ev, sorted within each mouse, and their four trial-mean traces"""
    Z = X[(X.event == ev) & X[COND].notna().all(1)].copy()
    if not len(Z):
        return None, None, None
    Z["order"] = Z[COND].mean(1)
    Z = pd.concat([Z[Z.mouse == m].sort_values("order", ascending=False) for m in MO])
    keys = {"vb": lambda r: (r.mouse, r.veh_day, "pre", r.veh_id), "vi": lambda r: (r.mouse, r.veh_day, "post", r.veh_id),
            "sb": lambda r: (r.mouse, r.sbi_day, "pre", r.sbi_id), "si": lambda r: (r.mouse, r.sbi_day, "post", r.sbi_id)}
    M, tt = {}, None
    for c in COND:
        rows = []
        for r in Z.itertuples():
            m, d, ph, cid = keys[c](r)
            v = D.get((m, d, ph, ev))
            if v is None:
                rows.append(None); continue
            tt = v["tt"]
            j = np.flatnonzero(np.asarray(v["ids"]).astype(int) == cid)
            rows.append(v["mean"][:, j[0]] if len(j) else None)
        M[c] = np.array([x if x is not None else np.full(len(tt), np.nan) for x in rows])
    return Z, M, tt


def fig_heatnorm(X, D, ev, lg, fn):
    """10-23 (user): baseline-normalized heatmap of the SAME neurons: after - same-day baseline on the vehicle day, on the SBI-553
    day, and the difference (SBI-553 change - vehicle change); same row order as TX1"""
    font(lg)
    Z, M, tt = heat_rows(X, D, ev)
    if Z is None:
        return None
    k = (tt >= -2) & (tt <= 4)
    P = {"dv": M["vi"] - M["vb"], "ds": M["si"] - M["sb"]}
    P["dd"] = P["ds"] - P["dv"]
    lab = {"dv": ("vehicle day: after - baseline", "vehicle 날: 후 - 기준"), "ds": ("SBI-553 day: after - baseline", "SBI-553 날: 후 - 기준"),
           "dd": ("SBI-553 change - vehicle change", "SBI-553 변화 - vehicle 변화")}
    colt = {"dv": CC["vehicle"], "ds": CC["SBI-553"], "dd": "k"}
    fig = plt.figure(figsize=FIG)
    # 10-24 layout: [vehicle change][SBI-553 change][its colour bar]   [difference][its own colour bar] - the difference is
    # another quantity, so it gets its own bar and label; one fixed range on every event slide (display only)
    x0, g, cbw, cbg = 0.088, 0.022, 0.0105, 0.009
    w = (1 - x0 - g - 2 * (cbg + cbw) - 0.083 - 0.072) / 3
    xs = [x0, x0 + w + g, x0 + 2 * w + g + cbg + cbw + 0.083]
    yb, hh = 0.105, 0.775                                      # 10-25: the title row's height goes to the rows
    w0 = WW.of(ev)
    ims = {}
    for i, c in enumerate(("dv", "ds", "dd")):
        a = fig.add_axes([xs[i], yb, w, hh])
        vl = VL4D if c == "dd" else VL4
        ims[c] = a.imshow(P[c][:, k], aspect="auto", cmap="RdBu_r", vmin=-vl, vmax=vl, interpolation="nearest",
                          extent=[tt[k][0], tt[k][-1], len(Z), 0])
        a.axvline(0, color="k", lw=0.9, ls="--")
        a.axvspan(w0[2], w0[3], ymin=1.008, ymax=1.035, color="k", clip_on=False)   # above the rows, not over them
        a.set_title(lab[c][1 if lg == "kr" else 0], color=colt[c], fontsize=13, fontweight="bold", pad=17)
        a.set_xlabel(T_("time", lg))
        a.set_xticks(XT)
        mouse_rows(a, Z, lg, i == 0)
    for c, x, vl, lb in (("ds", xs[1] + w + cbg, VL4, "cb4"), ("dd", xs[2] + w + cbg, VL4D, "cb4d")):
        cax = fig.add_axes([x, yb + 0.06, cbw, hh - 0.12])
        cb = fig.colorbar(ims[c], cax=cax, extend="both", ticks=np.linspace(-vl, vl, 5))
        cb.ax.set_yticklabels([f"{t:+g}" if t else "0" for t in np.linspace(-vl, vl, 5)])
        cb.set_label(T_(lb, lg), fontsize=12)
    heat_key(fig, w0, lg)                                      # 10-25: no in-figure title (it repeated the slide title)
    fig.savefig(fn, dpi=150); plt.close(fig)
    return len(Z)


def fig_heat(X, D, ev, lg, fn):
    """the same tracked neurons, the same row order, in all four sessions: trial-mean z minus the 3 s before"""
    font(lg)
    Z = X[(X.event == ev) & X[COND].notna().all(1)].copy()
    if not len(Z):
        return None
    Z["order"] = Z[COND].mean(1)
    Z = pd.concat([Z[Z.mouse == m].sort_values("order", ascending=False) for m in MO])
    keys = {"vb": lambda r: (r.mouse, r.veh_day, "pre", r.veh_id), "vi": lambda r: (r.mouse, r.veh_day, "post", r.veh_id),
            "sb": lambda r: (r.mouse, r.sbi_day, "pre", r.sbi_id), "si": lambda r: (r.mouse, r.sbi_day, "post", r.sbi_id)}
    M, tt = {}, None
    for c in COND:
        rows = []
        for r in Z.itertuples():
            m, d, ph, cid = keys[c](r)
            v = D.get((m, d, ph, ev))
            if v is None:
                rows.append(None); continue
            tt = v["tt"]
            j = np.flatnonzero(np.asarray(v["ids"]).astype(int) == cid)
            rows.append(v["mean"][:, j[0]] if len(j) else None)
        M[c] = np.array([x if x is not None else np.full(len(tt), np.nan) for x in rows])
    k = (tt >= -2) & (tt <= 4)
    fig = plt.figure(figsize=FIG)
    x0, g, cbw = 0.088, 0.024, 0.0105                          # 10-24: wider gaps + no edge ticks ('4' no longer meets '-2')
    w = (1 - x0 - 3 * g - 0.009 - cbw - 0.085) / 4
    yb, hh = 0.105, 0.775                                      # 10-25: the title row's height goes to the rows
    w0 = WW.of(ev)
    for i, c in enumerate(COND):
        a = fig.add_axes([x0 + i * (w + g), yb, w, hh])
        im = a.imshow(M[c][:, k], aspect="auto", cmap="RdBu_r", vmin=-VL1, vmax=VL1, interpolation="nearest",
                      extent=[tt[k][0], tt[k][-1], len(Z), 0])
        a.axvline(0, color="k", lw=0.9, ls="--")
        a.axvspan(w0[2], w0[3], ymin=1.008, ymax=1.035, color="k", clip_on=False)   # above the rows, not over them
        cond_col = CC["vehicle"] if c == "vi" else CC["SBI-553"] if c == "si" else CC["baseline"]
        a.set_title(T_(c + "1", lg), color=cond_col, fontsize=13.5, fontweight="bold", pad=17)
        a.set_xlabel(T_("time", lg))
        a.set_xticks(XT)
        mouse_rows(a, Z, lg, i == 0)
    cax = fig.add_axes([x0 + 4 * w + 3 * g + 0.009, yb + 0.06, cbw, hh - 0.12])
    tk = np.linspace(-VL1, VL1, 5)
    cb = fig.colorbar(im, cax=cax, extend="both", ticks=tk)
    cb.ax.set_yticklabels([f"{t:+g}" if t else "0" for t in tk])
    cb.set_label(T_("zc", lg), fontsize=12)
    heat_key(fig, w0, lg)                                      # 10-25: no in-figure title (it repeated the slide title)
    fig.savefig(fn, dpi=150); plt.close(fig)
    return len(Z)


def fig_neurons(X, S, ev, lg, fn):
    font(lg)
    Z = X[(X.event == ev) & X[COND].notna().all(1)]
    if not len(Z):
        return None
    r = S.set_index("event").loc[ev]
    w0 = WW.of(ev)
    kr = lg == "kr"
    fig = plt.figure(figsize=FIG)
    yb, hh = 0.265, 0.6                                        # 10-24: footer gone; the room below holds the keys of A and B
    a = fig.add_axes([0.068, yb, 0.33, hh])
    xs = {"vb": 0, "vi": 1, "sb": 2.4, "si": 3.4}
    for z in Z.itertuples():
        a.plot([0, 1], [z.vb, z.vi], color=MC[z.mouse], lw=0.6, alpha=0.35)
        a.plot([2.4, 3.4], [z.sb, z.si], color=MC[z.mouse], lw=0.6, alpha=0.35)
    for c in COND:
        col = CC["vehicle"] if c == "vi" else CC["SBI-553"] if c == "si" else CC["baseline"]
        v = Z[c]
        a.scatter(np.full(len(v), xs[c]) + np.random.default_rng(1).uniform(-0.07, 0.07, len(v)), v, s=9, color=col, alpha=0.5, lw=0)
        a.plot([xs[c] - 0.22, xs[c] + 0.22], [v.median()] * 2, color="k", lw=3)
    lo, hi = np.nanpercentile(Z[COND].to_numpy(), [1, 99])
    span = hi - lo
    a.set_ylim(lo - 0.05 * span, hi + 0.42 * span)
    yk = hi + 0.06 * span
    PS.bracket(a, 0, 1, yk, r.p_vb_vi, color=CC["vehicle"], fs=12.5)
    PS.bracket(a, 2.4, 3.4, yk, r.p_sb_si, color=CC["SBI-553"], fs=12.5)
    PS.bracket(a, 1, 3.4, yk + 0.15 * span, wil(Z.si, Z.vi), fs=12.5)
    PS.bracket(a, 0, 2.4, yk + 0.28 * span, r.p_vb_sb, color="0.45", fs=12.5)
    a.set_xticks(list(xs.values())); a.set_xticklabels([T_(c, lg) for c in COND])
    a.axhline(0, color="0.6", lw=0.8)
    a.set_ylabel((f"response (z)\n{w0[2]:g}-{w0[3]:g} s vs the {w0[0]:g} s before") if not kr else
                 (f"반응 (z)\n{w0[2]:g}-{w0[3]:g} s vs 직전 {w0[0]:g} s"), linespacing=1.3)
    a.set_title(("A  each neuron = one thin line; thick black = median" if not kr else "A  뉴런 하나 = 가는 선 하나; 굵은 검은 선 = 중앙값"),
                loc="left")
    hb = [Line2D([], [], color=CC["vehicle"], lw=2, label="vehicle day" if not kr else "vehicle 날"),
          Line2D([], [], color="k", lw=2, label="after vs after" if not kr else "후 vs 후"),
          Line2D([], [], color=CC["SBI-553"], lw=2, label="SBI-553 day" if not kr else "SBI-553 날"),
          Line2D([], [], color="0.45", lw=2, label="the two baselines" if not kr else "두 기준")]
    a.legend(handles=hb, loc="upper center", bbox_to_anchor=(0.5, -0.145), ncol=2, frameon=False, handlelength=1.6,
             columnspacing=1.6, borderpad=0.2, title=("brackets (Wilcoxon, paired by neuron):" if not kr else
                                                       "괄호 (Wilcoxon, 뉴런별 짝지음):"))
    b = fig.add_axes([0.468, yb, 0.235, hh])
    for m in MO:
        q = Z[Z.mouse == m]
        b.scatter(q.chg_veh, q.chg_sbi, marker=MK[m], color=MC[m], s=30, alpha=0.75, edgecolor="white", lw=0.4)
    lim = np.nanpercentile(np.abs(np.r_[Z.chg_veh, Z.chg_sbi]), 99) * 1.1
    b.plot([-lim, lim], [-lim, lim], color="0.3", ls="--", lw=1.2)
    b.axhline(0, color="0.75", lw=0.8); b.axvline(0, color="0.75", lw=0.8)
    b.set_xlim(-lim, lim); b.set_ylim(-lim, lim)
    b.set_xlabel(T_("dv", lg)); b.set_ylabel(T_("ds", lg))
    b.set_title(("B  change from the same-day baseline" if not kr else "B  같은 날 기준 대비 변화"), loc="left")
    # 10-24 audit: the stats box and the legend sat on the points -> below the axes
    b.legend(handles=[Line2D([], [], color="0.3", ls="--", lw=1.2,
                             label=("equal change on both days\n(below = smaller on the SBI-553 day)" if not kr else
                                    "두 날 변화가 같음\n(아래 = SBI-553 날 변화가 더 작음)"))],
             loc="upper left", bbox_to_anchor=(-0.02, -0.118), frameon=False, handlelength=1.8, borderpad=0.2)
    b.text(0.0, -0.268, ((f"median paired difference {r.med_norm:+.3f} z, Wilcoxon {ptxt(r.p_norm)}\n"
                          f"mouse-level t-test (4 mice) {ptxt(r.p_mouse_norm)}") if not kr else
                         (f"짝지은 차이 중앙값 {r.med_norm:+.3f} z, Wilcoxon {ptxt(r.p_norm)}\n"
                          f"마우스 단위 t 검정 (4마리) {ptxt(r.p_mouse_norm)}")), transform=b.transAxes, va="top", ha="left",
           fontsize=12, linespacing=1.25)
    c = fig.add_axes([0.805, yb, 0.18, hh])
    for j, m in enumerate(MO):                                 # 10-23 review: the same quantity as the count in the key sentence
        q = Z[Z.mouse == m]
        if len(q):
            c.scatter(j, q.d_norm.median(), marker=MK[m], color=MC[m], s=95, edgecolor="white", lw=1.0, zorder=4, clip_on=False)
    c.axhline(0, color="0.5", lw=0.9, zorder=2)
    c.set_xticks(range(len(MO)))
    c.set_xticklabels([f"{m}\nDay {SBI_DAY[m]}" for m in MO], fontsize=12)
    c.set_xlabel(("mouse\n(day of SBI-553)" if not kr else "마우스\n(SBI-553 투여일)"))
    c.set_xlim(-0.6, len(MO) - 0.4)
    c.set_ylabel(("paired difference, mouse median (z)\n> 0 = larger change on SBI-553 day" if not kr else
                  "짝지은 차이, 마우스 중앙값 (z)\n> 0 = SBI-553 날 변화가 더 큼"), linespacing=1.3)
    c.set_title(("C  per mouse" if not kr else "C  마우스별"), loc="left")
    # the mouse key (line colour in A, marker in B and C) shares the title row
    fig.legend(handles=[Line2D([], [], color=MC[m], marker=MK[m], lw=1.5, ms=8, label=f"{m} (n = {int((Z.mouse == m).sum())})")
                        for m in MO], loc="upper right", bbox_to_anchor=(0.995, 1.0), ncol=4, frameon=False, handlelength=1.8,
               columnspacing=1.3, title=None)
    fig.suptitle((f"{ev}: {len(Z)} neurons followed on both days (colour = mouse)") if not kr else
                 (f"{ev}: 두 날 모두 추적한 뉴런 {len(Z)}개 (색 = 마우스)"), fontsize=13.5, x=0.01, ha="left", y=0.985)
    foot(fig, (("A: brackets = Wilcoxon signed-rank paired by neuron (blue = vehicle day, red = SBI-553 day, black = after "
                "vehicle vs after SBI-553, grey = the two baselines); y axis 1st-99th percentile. " + PS.KEY) if lg != "kr" else
               ("A: 괄호 = 뉴런별 짝지은 Wilcoxon signed-rank (파랑 = vehicle 날, 빨강 = SBI-553 날, 검은색 = vehicle 후 vs SBI-553 후, "
                "회색 = 두 기준); y축 1-99 백분위. " + PS.KEY)), lg)
    fig.savefig(fn, dpi=150); plt.close(fig)
    return len(Z)


def fig_responsive(Rn, C, lg, fn):
    font(lg)
    fig, ax = plt.subplots(2, 1, figsize=FIG, sharex=True)    # 10-24: 15 x 6.3 (was 6.6) = the deck box, so fonts stay >= 9 pt
    x = np.arange(len(EVENTS))
    for a, k, lab in ((ax[0], "up", ("UP (activated)", "UP (활성화)")), (ax[1], "down", ("DOWN (inhibited)", "DOWN (억제)"))):
        Z = Rn[Rn.kind == k].set_index("event").loc[EVENTS]
        a.bar(x - 0.19, Z.chg_veh, 0.36, color=CC["vehicle"], label="vehicle day" if lg != "kr" else "vehicle 날")
        a.bar(x + 0.19, Z.chg_sbi, 0.36, color=CC["SBI-553"], label="SBI-553 day" if lg != "kr" else "SBI-553 날")
        for i, ev in enumerate(EVENTS):
            for m in MO:
                mv = Z.loc[ev, "mice"][m]
                for dx, kk in ((-0.19, "vehicle"), (0.19, "sbi")):
                    if np.isfinite(mv[kk]):
                        a.scatter(i + dx, mv[kk], marker=MK[m], color=MC[m], s=32, edgecolor="white", lw=0.5, zorder=4,
                                  clip_on=False)
        a.axhline(0, color="0.5", lw=0.8)
        lo, hi = a.get_ylim(); span = hi - lo
        a.set_ylim(lo, hi + 0.22 * span)
        for i, ev in enumerate(EVENTS):
            PS.bracket(a, i - 0.19, i + 0.19, hi + 0.04 * span, Z.loc[ev, "p"], fs=12.5)
        a.set_ylabel(("change in % of neurons\n(after - same-day baseline)" if lg != "kr" else "뉴런 % 변화\n(후 - 같은 날 기준)"),
                     fontsize=12, linespacing=1.3)
        a.set_title(lab[1 if lg == "kr" else 0], loc="left", fontsize=13.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels(EVENTS, fontsize=12.5)
    h = [plt.Rectangle((0, 0), 1, 1, color=CC["vehicle"], label="vehicle day (216 neurons)" if lg != "kr" else "vehicle 날 (뉴런 216)"),
         plt.Rectangle((0, 0), 1, 1, color=CC["SBI-553"], label="SBI-553 day (246 neurons)" if lg != "kr" else "SBI-553 날 (뉴런 246)")]
    h += [Line2D([], [], marker=MK[m], ls="", color=MC[m], label=m) for m in MO]
    fig.legend(handles=h, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.6,
               title=("bars = all neurons pooled, markers = one mouse each" if lg != "kr" else "막대 = 뉴런 합산, 표식 = 마우스 하나"))
    foot(fig, (("Every neuron of each day; bars = neurons pooled, markers = mice. Brackets = vehicle-day vs SBI-553-day neurons, the "
                "per-neuron change from the same-day baseline (+1 gained, 0, -1 lost), Mann-Whitney U. " + PS.KEY) if lg != "kr" else
               ("각 날의 모든 뉴런; 막대 = 뉴런 합산, 표식 = 마우스. 괄호 = vehicle 날 vs SBI-553 날 뉴런, 같은 날 기준 대비 뉴런별 "
                "변화 (+1 생김, 0, -1 사라짐), Mann-Whitney U. " + PS.KEY)), lg)
    fig.tight_layout(rect=(0, 0.035, 1, 0.895), h_pad=0.6)
    fig.savefig(fn, dpi=150); plt.close(fig)


def jconv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def main():
    os.makedirs(OUT, exist_ok=True)
    T, X = load()
    X.to_csv(os.path.join(OUT, "tracked_responses.csv"), index=False)
    S = tests(X); SA = tests(X, "A")
    pd.concat([S, SA]).to_csv(os.path.join(OUT, "tracked_tests.csv"), index=False)
    C, Rn = responsive_norm(T)
    Rn.to_csv(os.path.join(OUT, "responsive_norm.csv"), index=False)
    D = np.load(CACHE, allow_pickle=True)["D"].item()
    for lg in ("en", "kr"):
        sfx = "_kr" if lg == "kr" else ""
        fig_summary(S, lg, os.path.join(OUT, f"TX0_summary{sfx}.png"))
        fig_responsive(Rn, C, lg, os.path.join(OUT, f"TX3_responsive_norm{sfx}.png"))
        for ev in EVENTS:
            tag = ev.split()[0].lower()
            fig_heat(X, D, ev, lg, os.path.join(OUT, f"TX1_heat_{tag}{sfx}.png"))
            fig_neurons(X, S, ev, lg, os.path.join(OUT, f"TX2_neurons_{tag}{sfx}.png"))
            fig_heatnorm(X, D, ev, lg, os.path.join(OUT, f"TX4_heatnorm_{tag}{sfx}.png"))
    J = dict(n_tracked=int(X[["mouse", "day1_id"]].drop_duplicates().shape[0]),
             n_tracked_A=int(X[X.track_class == "A"][["mouse", "day1_id"]].drop_duplicates().shape[0]),
             per_mouse={m: int(X[X.mouse == m].day1_id.nunique()) for m in MO},
             tests=S.to_dict("records"), tests_A=SA.to_dict("records"), responsive=Rn.to_dict("records"),
             windows={ev: list(WW.of(ev)) for ev in EVENTS})
    json.dump(J, open(os.path.join(OUT, "tracked_summary.json"), "w", encoding="utf-8"), indent=1, default=jconv)
    pd.set_option("display.width", 250)
    print(S[["event", "n", "med_after", "p_after", "p_mouse_after", "bal_after", "day_after", "med_norm", "p_norm", "p_mouse_norm",
             "bal_norm", "day_norm", "p_g1_after", "p_g2_after", "mice_neg_norm"]].round(4).to_string())
    print(SA[["event", "n", "med_after", "p_after", "med_norm", "p_norm", "p_mouse_norm"]].round(4).to_string())
    print(Rn[["event", "kind", "base_veh", "base_sbi", "chg_veh", "chg_sbi", "p", "mice_lower"]].round(3).to_string())
    print("->", OUT)


if __name__ == "__main__":
    main()
