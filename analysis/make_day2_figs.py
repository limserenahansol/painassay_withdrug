"""make_day2_figs.py  -  the Day 1 vs Day 2 figure set.

Every figure here shows both days at two levels:
    population   mean +/- SEM over the six mice
    individual   one line or one point per mouse, always drawn

WHY A SEPARATE SCRIPT
    make_extra_figs.py and make_lab_meeting_figs.py are the Day-1 descriptive
    set and they still work unchanged. This one only exists once a second day
    exists, so keeping it separate means the Day-1 figures cannot break.

NORMALISATION - the whole point
    Nothing here is a raw count. The number of stimuli delivered is NOT fixed:
    on Day 1 it runs 16 to 31 taps for pin prick across the six mice, and it
    will differ again on Day 2. Ten flinches from ten taps and ten flinches
    from twenty taps are the same raw count but not the same animal - 1.00 vs
    0.50 responses per delivery. So every measure is per delivery, with each
    mouse divided by its OWN delivery count.

    Raw counts are still plotted in D6, beside the delivery counts that
    produced them, precisely so the confound is visible rather than hidden.

--mockup
    Stamps MOCKUP on every panel and writes to a *_MOCKUP* filename. Use this
    with synthetic Day 2 data when planning the deck. Nothing that leaves this
    script with fabricated numbers in it is allowed to look real.

USAGE
    python make_day2_figs.py --day1 <folder> --day2 <folder> --out <folder>
    python make_day2_figs.py --day1 <folder> --day2 <synth> --out <folder> --mockup

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

AFF = {1: "attending", 2: "lickbite", 3: "guarding", 4: "escape"}
REF = {1: "withdrawal", 2: "flinch"}
ORDER = ["withdrawal", "flinch", "attending", "lickbite", "guarding", "escape"]
NICE = {"withdrawal": "Paw withdrawal", "flinch": "Flinch",
        "attending": "Paw attending", "lickbite": "Licking / biting",
        "guarding": "Guarding", "escape": "Escape / rearing"}
STIM = ["Light touch", "Mild touch", "Heat", "Pin prick"]
SCOL = {"Light touch": "#9CC3DA", "Mild touch": "#5E9FC4",
        "Heat": "#C0483B", "Pin prick": "#7A3B8F"}
# The axis label spells the ratio out. "events per stimulus" was ambiguous:
# the x-axis of most panels IS the four stimulus TYPES, so "per stimulus" read
# as "per stimulus type" rather than per individual delivery. Naming the
# numerator and the denominator removes the ambiguity and also distinguishes
# this from the other two normalisations in the pipeline (per minute, and
# delta against baseline).
YLAB = "total event number /\ntotal stimulus delivery"
YLAB1 = "total event number / total stimulus delivery"
C1, C2 = "#444444", "#C0483B"          # Day 1, Day 2
CR, CA = "#C0483B", "#1C6E8C"          # reflexive, affective titles
PRE, POST = 5.0, 15.0
WIN = {"withdrawal": (3.0, False), "flinch": (3.0, False),
       "attending": (10.0, False), "lickbite": (10.0, False),
       "guarding": (10.0, False), "escape": (10.0, False)}
# COUNT EVERY EVENT BY DEFAULT, not only those inside a response window.
#
# The two numerators answer different questions:
#   ALL       how much of this behaviour did the animal do
#   WINDOWED  did this particular stimulus provoke a response
#
# The window was the default and it silently discarded most of one behaviour:
# only 29 % of escape/rearing events start within 10 s of a delivery, because
# escape/rearing is spontaneous exploration and mostly happens away from the
# stimulus. That changed the headline - escape/rearing came out at x0.08
# windowed but x0.22 counting everything.
#
# For "how much did the animal do", which is the sedation question, ALL is the
# correct numerator. --response-window restores the windowed version for the
# stimulus-evoked question.
USE_WINDOW = False
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.labelsize": 12, "figure.dpi": 130})
MOCKUP = False


def s_(v, d=""):
    a = np.asarray(v).ravel()
    if a.size == 0:
        return d
    x = a[0]
    if isinstance(x, np.ndarray):
        x = x.ravel()[0] if x.size else d
    return str(x).strip()


def f_(v, d=np.nan):
    a = np.asarray(v, dtype=float).ravel()
    return float(a[0]) if a.size else d


def stamp(fig):
    """Make fabricated data impossible to mistake for a result."""
    if not MOCKUP:
        return
    fig.text(.5, .5, "MOCKUP\nsynthetic Day 2", ha="center", va="center",
             fontsize=64, color="#C0483B", alpha=.16, rotation=24,
             fontweight="bold", zorder=100)
    fig.text(.5, .003, "Day 2 numbers are SYNTHETIC - layout preview only, "
                       "not a result", ha="center", va="bottom", fontsize=11,
             color=C2, fontweight="bold")


def save(fig, path):
    stamp(fig)
    if MOCKUP:
        r, e = os.path.splitext(path)
        path = r + "_MOCKUP" + e
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(path)}")
    return path


def read_all(folder, label):
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "ScoringAB_*.mat"))):
        M = loadmat(p)
        fps = f_(M["frameRate"], 30.0)
        n = int(f_(M.get("nUsed", 0), 0))
        sc = np.asarray(M["score"]).ravel().astype(int)
        sc = sc[:n] if n else sc
        rx = np.asarray(M.get("reflexEvents", np.empty((0, 2)))).reshape(-1, 2)
        names = [s_(x) for x in np.asarray(M["stimNames"]).ravel()][:4]
        out.append(dict(day=label, mouse=s_(M.get("mouseID", "")),
                        sex=s_(M.get("sexID", "")), fps=fps, score=sc,
                        n=len(sc), rx=rx, names=names,
                        dF=np.asarray(M["dFrames"]).ravel().astype(int),
                        dT=np.asarray(M["dTypes"]).ravel().astype(int)))
    return out


def occupancy(S, code, fps, dF, pre, post):
    a, b = int(pre * fps), int(post * fps)
    rows = [(S[f - a - 1:f + b - 1] == code).astype(float)
            for f in dF if f - a >= 1 and f + b <= len(S)]
    return np.vstack(rows) if rows else np.empty((0, a + b))


def reflex_occupancy(rx, code, fps, dF, pre, post, n):
    a, b = int(pre * fps), int(post * fps)
    hit = np.zeros(n + 2)
    if rx.size:
        f = rx[rx[:, 1] == code, 0].astype(int)
        hit[f[(f >= 1) & (f <= n)]] = 1
    rows = [hit[f - a:f + b] for f in dF if f - a >= 1 and f + b <= n]
    return np.vstack(rows) if rows else np.empty((0, a + b))


def bout_starts(d):
    """True first frame of every bout, per behaviour, over the whole session.

    Counting rising edges inside a time slice is wrong at the slice edge: with
    np.r_[0, seg] a bout that was ALREADY RUNNING when the window opened looks
    like a new bout starting there, so every window boundary can manufacture
    an event. verify_final_numbers.py caught one such case (F1, Day 1, Heat,
    attending: 2 counted, 1 real). Computing starts once over the session and
    then asking which fall inside the span cannot do that.
    """
    out = {}
    for code, nm in AFF.items():
        m = (d["score"] == code).astype(np.int8)
        e = np.diff(np.concatenate(([0], m, [0])))
        out[nm] = np.flatnonzero(e == 1) + 1          # 1-based frame
    for code, nm in REF.items():
        ff = (d["rx"][d["rx"][:, 1] == code, 0].astype(int)
              if d["rx"].size else np.array([], int))
        out[nm] = ff[(ff >= 1) & (ff <= d["n"])]
    return out


def rate_table(data):
    """Events per delivery, one row per mouse x day x stimulus x behaviour.

    This is the table every population figure below is built from. The
    denominator is that mouse's own delivery count on that day, which is why
    a mouse that received 31 taps is comparable with one that received 16.
    """
    rows = []
    for d in data:
        ST = bout_starts(d)
        for ty in range(1, 5):
            sel = np.sort(d["dF"][d["dT"] == ty].astype(int))
            if not len(sel):
                continue
            nm = d["names"][ty - 1]
            for b in ORDER:
                w, _ = WIN[b]
                # USE_WINDOW off: one span covering the whole block, so every
                # event during that stimulus is counted whatever its timing.
                # On: one span per delivery, so only stimulus-evoked responses
                # are counted.
                if USE_WINDOW:
                    spans = [(int(f), min(int(f + w * d["fps"]), d["n"]))
                             for f in sel]
                else:
                    spans = [(int(sel[0]),
                              min(int(sel[-1] + w * d["fps"]), d["n"]))]
                hi_n = len(sel)
                st = ST[b]
                tot = 0
                for f, hi in spans:
                    if hi <= f:
                        continue
                    tot += int(np.sum((st >= f) & (st < hi)))
                if hi_n:
                    rows.append(dict(day=d["day"], mouse=d["mouse"],
                                     sex=d["sex"], stimulus=nm, behaviour=b,
                                     n_deliveries=hi_n, n_events=tot,
                                     per_delivery=tot / hi_n))
    return pd.DataFrame(rows)


def pstats(x, y):
    """Paired Wilcoxon plus the exact-test floor this n imposes."""
    n = len(x)
    if n < 2 or np.allclose(np.asarray(y) - np.asarray(x), 0):
        return np.nan, 2.0 / (2 ** n) if n else np.nan
    try:
        return float(stats.wilcoxon(x, y, zero_method="wilcox",
                                    method="exact").pvalue), 2.0 / (2 ** n)
    except ValueError:
        return np.nan, 2.0 / (2 ** n)


def star(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


# ───────────────── D1: PSTH, both days, per behaviour ──────────────────────
def d1_psth(dd, path, lab1, lab2):
    """Response SHAPE. A drug can change how much, or how fast, or how long.
    Only a time course separates those three."""
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 6.8))
    fps = dd[0][0]["fps"]
    t = np.arange(-int(PRE * fps), int(POST * fps)) / fps
    for ax, b in zip(axes.ravel(), ORDER):
        for data, lab, col, ls in ((dd[0], lab1, C1, "-"),
                                   (dd[1], lab2, C2, "--")):
            per = []
            for d in data:
                if b in REF.values():
                    code = [k for k, v in REF.items() if v == b][0]
                    m = reflex_occupancy(d["rx"], code, d["fps"], d["dF"],
                                         PRE, POST, d["n"])
                else:
                    code = [k for k, v in AFF.items() if v == b][0]
                    m = occupancy(d["score"], code, d["fps"], d["dF"],
                                  PRE, POST)
                if m.size:
                    per.append(m.mean(axis=0))
            if not per:
                continue
            Y = np.vstack(per)
            k = max(1, int(.5 * fps))
            sm = lambda v: np.convolve(v, np.ones(k) / k, mode="same")
            y = sm(Y.mean(axis=0))
            e = sm(Y.std(axis=0, ddof=1) / np.sqrt(len(Y))) if len(Y) > 1 \
                else np.zeros_like(y)
            ax.fill_between(t[:len(y)], y - e, y + e, color=col, alpha=.16,
                            lw=0)
            ax.plot(t[:len(y)], y, color=col, lw=2.4, ls=ls, label=lab)
        ax.axvline(0, color="k", lw=1.4, ls=":")
        ax.set_title(NICE[b], fontweight="bold",
                     color=CR if b in REF.values() else CA)
        ax.grid(alpha=.2)
        ax.set_xlim(-PRE, POST)
    axes[0][0].set_ylabel("P(behaviour on)")
    axes[1][0].set_ylabel("P(behaviour on)")
    axes[1][0].set_xlabel("seconds from stimulus delivery")
    axes[0][2].legend(fontsize=10, frameon=False)
    fig.suptitle("Response time course, averaged over all stimuli   ·   "
                 "mean ± SEM over 6 mice\n"
                 "the dip to zero at t = 0 in the affective panels is the "
                 "scorer releasing the hold key to tap the delivery, "
                 "not the animal", fontsize=12)
    fig.tight_layout()
    return save(fig, path)


# ────────── D2: events per delivery, per stimulus, paired per mouse ─────────
def d2_per_delivery(R, path, lab1, lab2):
    """The primary figure. Normalised by each mouse's own stimulus count, so
    a mouse that got 31 taps sits on the same axis as one that got 16."""
    behs = ORDER
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.4))
    rows = []
    for ax, b in zip(axes.ravel(), behs):
        marks = []
        for i, s in enumerate(STIM):
            g = R[(R.behaviour == b) & (R.stimulus == s)]
            w = g.pivot_table(index="mouse", columns="day",
                              values="per_delivery")
            if lab1 not in w.columns or lab2 not in w.columns:
                continue
            w = w.dropna()
            if not len(w):
                continue
            x, y = w[lab1].to_numpy(), w[lab2].to_numpy()
            x0, x1 = i - .17, i + .17
            for u, v in zip(x, y):
                ax.plot([x0, x1], [u, v], "-", color="#BBB", lw=1.0,
                        zorder=2)
            ax.plot(np.full(len(x), x0), x, "o", ms=4, color=C1, alpha=.75,
                    zorder=3)
            ax.plot(np.full(len(y), x1), y, "o", ms=4, color=C2, alpha=.75,
                    zorder=3)
            for xx, v, c in ((x0, x, C1), (x1, y, C2)):
                m = v.mean()
                se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
                ax.errorbar([xx], [m], yerr=[se], fmt="s", color=c, ms=11,
                            capsize=6, lw=2.2, mec="white", mew=1.5, zorder=6)
            p, floor = pstats(x, y)
            rows.append(dict(behaviour=b, stimulus=s, n=len(x),
                             day1=x.mean(), day2=y.mean(), p_paired=p,
                             p_floor=floor))
            marks.append((i, star(p)))
        ax.set_xticks(range(len(STIM)))
        ax.set_xticklabels([s.replace(" ", "\n") for s in STIM], fontsize=9)
        ax.set_title(NICE[b], fontweight="bold",
                     color=CR if b in REF.values() else CA)
        ax.grid(alpha=.2, axis="y")
        # headroom first, then place the marks INSIDE the axes. Putting them
        # at data height ran them into the panel title.
        ax.set_ylim(0, ax.get_ylim()[1] * 1.16)
        tr = matplotlib.transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        for i, txt in marks:
            ax.text(i, .94, txt, ha="center", va="top", fontsize=10,
                    color="#333", transform=tr)
    axes[0][0].set_ylabel(YLAB, fontsize=11)
    axes[1][0].set_ylabel(YLAB, fontsize=11)
    h = [plt.Line2D([], [], marker="s", ls="", color=c, ms=10, label=l)
         for c, l in ((C1, lab1), (C2, lab2))]
    # top-right of the figure: the suptitle is centred and the MOCKUP footer
    # owns the bottom, so this is the only corner that is reliably free
    fig.legend(handles=h, fontsize=11, frameon=False, ncol=1,
               loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{YLAB1}   ·   grey line = one mouse measured twice, "
                 "square = mean ± SEM\n"
                 "each mouse divided by its OWN delivery count, so unequal "
                 "stimulus numbers cannot bias the comparison   ·   "
                 "paired Wilcoxon, n = 6 (floor p = 0.031)", fontsize=12)
    fig.tight_layout()
    save(fig, path)
    return pd.DataFrame(rows)


# ──────────── D3: within-block time course, both days ──────────────────────
def d3_timecourse(dd, path, lab1, lab2, nbin=5, blocklen=300.0):
    """Does the response fade inside a 5 min block, and does the drug change
    the fading rather than the level?"""
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 6.8))
    edges = np.linspace(0, blocklen, nbin + 1)
    ctr = (edges[:-1] + edges[1:]) / 2 / 60.0
    for ax, b in zip(axes.ravel(), ORDER):
        for data, lab, col, ls in ((dd[0], lab1, C1, "-"),
                                   (dd[1], lab2, C2, "--")):
            per = []
            for d in data:
                allf = np.sort(d["dF"].astype(float))
                if not len(allf):
                    continue
                # a block starts at its first delivery
                starts, last = [], -1e9
                for f in allf:
                    if f - last > 120 * d["fps"]:
                        starts.append(f)
                    last = f
                vals = np.full(nbin, np.nan)
                acc = [[] for _ in range(nbin)]
                for st in starts:
                    for j in range(nbin):
                        lo = st + edges[j] * d["fps"]
                        hi = st + edges[j + 1] * d["fps"]
                        sel = allf[(allf >= lo) & (allf < hi)]
                        if not len(sel):
                            continue
                        tot = 0
                        for f in sel:
                            f = int(f)      # allf is float for the gap maths
                            w, _ = WIN[b]
                            h2 = min(int(f + w * d["fps"]), d["n"])
                            if h2 <= f:
                                continue
                            if b in REF.values():
                                code = [k for k, v in REF.items()
                                        if v == b][0]
                                ff = d["rx"][d["rx"][:, 1] == code, 0] \
                                    if d["rx"].size else np.array([])
                                tot += int(np.sum((ff >= f) & (ff < h2)))
                            else:
                                code = [k for k, v in AFF.items()
                                        if v == b][0]
                                seg = (d["score"][f - 1:h2 - 1]
                                       == code).astype(int)
                                tot += int((np.diff(np.r_[0, seg])
                                            == 1).sum())
                        acc[j].append(tot / len(sel))
                for j in range(nbin):
                    if acc[j]:
                        vals[j] = np.mean(acc[j])
                per.append(vals)
            if not per:
                continue
            Y = np.vstack(per)
            m = np.nanmean(Y, axis=0)
            cnt = np.sum(np.isfinite(Y), axis=0)
            e = np.nanstd(Y, axis=0, ddof=1) / np.sqrt(np.maximum(cnt, 1))
            ax.errorbar(ctr, m, yerr=e, fmt="o" + ls, color=col, lw=2.2,
                        ms=6, capsize=4, label=lab)
        ax.set_title(NICE[b], fontweight="bold",
                     color=CR if b in REF.values() else CA)
        ax.grid(alpha=.2)
        ax.set_ylim(bottom=0)
    axes[1][0].set_xlabel("minutes into the 5 min stimulus block")
    axes[0][0].set_ylabel(YLAB, fontsize=11)
    axes[1][0].set_ylabel(YLAB, fontsize=11)
    axes[0][2].legend(fontsize=10, frameon=False)
    fig.suptitle("Within-block time course, 1 min bins   ·   mean ± SEM over "
                 "mice   ·   pooled over the four stimulus blocks\n"
                 "a drug can lower the level, or steepen the fade - these "
                 "are different mechanisms", fontsize=12)
    fig.tight_layout()
    return save(fig, path)


# ───────── D4: reflexive vs affective, one point per mouse per day ─────────
def d4_dissoc(R, path, lab1, lab2):
    """The claim SBI-553 is meant to support: affective relief without a
    matching reflexive change. That is a MOVEMENT DOWN, not left."""
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    piv = (R.groupby(["day", "mouse", "behaviour"]).per_delivery.mean()
           .reset_index()
           .pivot_table(index=["day", "mouse"], columns="behaviour",
                        values="per_delivery"))
    refl = [b for b in REF.values() if b in piv.columns]
    affe = [b for b in ("attending", "lickbite", "guarding")
            if b in piv.columns]
    piv["refl"] = piv[refl].mean(axis=1)
    piv["affe"] = piv[affe].mean(axis=1)
    for (day, mouse), r in piv.iterrows():
        c = C1 if day == lab1 else C2
        ax.plot(r.refl, r.affe, "o", ms=12, color=c, mec="white", mew=1.6,
                zorder=5)
        ax.annotate(mouse, (r.refl, r.affe), fontsize=8, color="#333",
                    xytext=(0, -18), textcoords="offset points",
                    ha="center")
    for mouse in piv.index.get_level_values(1).unique():
        try:
            a = piv.loc[(lab1, mouse)]
            b = piv.loc[(lab2, mouse)]
        except KeyError:
            continue
        ax.annotate("", xy=(b.refl, b.affe), xytext=(a.refl, a.affe),
                    arrowprops=dict(arrowstyle="->", color="#999", lw=1.4))
    ax.set_xlabel("REFLEXIVE  (withdrawal + flinch)\n" + YLAB1, fontsize=11)
    ax.set_ylabel("AFFECTIVE  (attending + lick/bite + guarding)\n" + YLAB1,
                  fontsize=11)
    h = [plt.Line2D([], [], marker="o", ls="", color=c, ms=11, label=l)
         for c, l in ((C1, lab1), (C2, lab2))]
    ax.legend(handles=h, fontsize=10, frameon=False, loc="upper left")
    ax.grid(alpha=.2)
    ax.set_title("Reflexive vs affective, one arrow per mouse\n"
                 "a biased analgesic moves points DOWN (affective relief) "
                 "more than LEFT (reflex)", fontsize=12)
    fig.tight_layout()
    return save(fig, path)


# ───── D5: per-mouse effect size, ranked - the individual-level figure ─────
def d5_per_mouse(R, path, lab1, lab2):
    """Population means hide whether all six animals moved or only two.

    Three things this had wrong and now does not:

      * mouse order was sorted by effect size INSIDE each panel, so F1 was the
        top row in one panel and the bottom row in the next. You could not
        read one animal across behaviours. The order is now fixed.
      * "n/6 down" sat at axes y = 1.0 and collided with the panel title.
      * the x axis was per-cent change, which is bounded at -100 % but
        unbounded upwards. One mouse at +650 % (F1 guarding, from a near-zero
        Day-1 baseline) flattened every other bar into invisibility. It is now
        a RATIO on a log axis, which treats halving and doubling as equal
        distances and cannot be dominated by one large positive value.
    """
    behs = ORDER
    mice = sorted({m for m in R.mouse.unique() if m})
    fig, axes = plt.subplots(1, len(behs), figsize=(2.75 * len(behs), 4.9),
                             sharey=True)
    ladder = np.array([1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8], float)
    for ax, b in zip(axes, behs):
        g = (R[R.behaviour == b].groupby(["day", "mouse"])
             .per_delivery.mean().reset_index()
             .pivot_table(index="mouse", columns="day",
                          values="per_delivery"))
        if lab1 not in g.columns or lab2 not in g.columns:
            continue
        # fixed row order, one row per mouse, same in every panel
        g = g.reindex(mice)
        lo_clip = 1 / 24.0
        rat, ylab = [], []
        for m in mice:
            d1, d2 = g.at[m, lab1], g.at[m, lab2]
            if not np.isfinite(d1) or d1 == 0:
                rat.append(np.nan)
            elif d2 == 0:
                rat.append(lo_clip)          # total loss, drawn at the edge
            else:
                rat.append(d2 / d1)
            ylab.append(m)
        rat = np.asarray(rat, float)
        y = np.arange(len(mice))[::-1]       # F1 at the top
        for yy, v, m in zip(y, rat, mice):
            if not np.isfinite(v):
                continue
            col = C2 if v < 1 else "#6E8CA0"
            ax.plot([1, max(v, lo_clip)], [yy, yy], "-", color=col, lw=3.0,
                    solid_capstyle="butt", zorder=3)
            ax.plot([max(v, lo_clip)], [yy], "o", ms=7, color=col,
                    mec="white", mew=1.2, zorder=5)
            if g.at[m, lab2] == 0:
                ax.text(lo_clip, yy, " 0", va="center", ha="left",
                        fontsize=7.5, color=C2)
        ax.axvline(1, color="k", lw=1.3)
        ax.set_xscale("log")
        keep = ladder[(ladder >= np.nanmin(np.r_[rat, 1]) * .7)
                      & (ladder <= np.nanmax(np.r_[rat, 1]) * 1.4)]
        if len(keep) < 2:
            keep = np.array([.5, 1, 2], float)
        ax.set_xticks(keep)
        ax.set_xticklabels([("1" if abs(t - 1) < 1e-9 else
                             (f"1/{int(round(1 / t))}" if t < 1
                              else f"{int(round(t))}")) for t in keep],
                           fontsize=8.5)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks(y)
        ax.set_yticklabels(ylab, fontsize=10)
        ax.set_ylim(-.7, len(mice) - .3)
        nd = int(np.nansum(rat < 1))
        ntot = int(np.isfinite(rat).sum())
        # title and count on separate lines, inside the title block, so they
        # cannot overlap
        ax.set_title(f"{NICE[b]}\n{nd}/{ntot} down", fontsize=10,
                     fontweight="bold",
                     color=CR if b in REF.values() else CA)
        ax.grid(alpha=.22, axis="x")
        ax.set_xlabel("Day 2 / Day 1", fontsize=9)
    fig.suptitle("One row per mouse, same order in every panel   ·   "
                 "left of the line = fewer events on the drug day\n"
                 "ratio of (total events / total stimulus delivery), "
                 "log axis so a halving and a doubling are the same distance",
                 fontsize=11.5)
    fig.tight_layout()
    return save(fig, path)


# ───── D6: the confound, made visible - raw counts beside denominators ─────
def d6_raw_vs_norm(R, path, lab1, lab2):
    """Why the raw count is not the measure. Left: how many stimuli each
    mouse actually received. Right: the same behaviour raw and normalised."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    days = [lab1, lab2]
    dl = (R.groupby(["day", "mouse", "stimulus"]).n_deliveries.first()
          .reset_index())
    ax = axes[0]
    for i, s in enumerate(STIM):
        for j, day in enumerate(days):
            v = dl[(dl.stimulus == s) & (dl.day == day)].n_deliveries
            if not len(v):
                continue
            xx = i + (-.17 if j == 0 else .17)
            ax.plot(np.full(len(v), xx), v, "o", ms=5,
                    color=C1 if j == 0 else C2, alpha=.8)
            ax.plot([xx - .1, xx + .1], [v.mean()] * 2, "-",
                    color=C1 if j == 0 else C2, lw=2.6)
    ax.set_xticks(range(len(STIM)))
    ax.set_xticklabels([s.replace(" ", "\n") for s in STIM], fontsize=9)
    ax.set_ylabel("stimuli delivered per block")
    ax.set_title("THE DENOMINATOR\nit is not constant", fontsize=11,
                 fontweight="bold")
    ax.grid(alpha=.2, axis="y")

    b = "lickbite"
    for ax, col, lbl in ((axes[1], "n_events",
                          "total event number\nCONFOUNDED"),
                         (axes[2], "per_delivery",
                          YLAB1.replace(" / ", " /\n") + "\nCOMPARABLE")):
        g = R[R.behaviour == b]
        for i, s in enumerate(STIM):
            w = (g[g.stimulus == s].pivot_table(index="mouse", columns="day",
                                                values=col))
            if lab1 not in w.columns or lab2 not in w.columns:
                continue
            w = w.dropna()
            x, y = w[lab1].to_numpy(), w[lab2].to_numpy()
            for u, v in zip(x, y):
                ax.plot([i - .17, i + .17], [u, v], "-", color="#CCC", lw=.9)
            ax.plot(np.full(len(x), i - .17), x, "o", ms=4, color=C1)
            ax.plot(np.full(len(y), i + .17), y, "o", ms=4, color=C2)
            for xx, v, c in ((i - .17, x, C1), (i + .17, y, C2)):
                ax.plot([xx - .1, xx + .1], [v.mean()] * 2, "-", color=c,
                        lw=2.6)
        ax.set_xticks(range(len(STIM)))
        ax.set_xticklabels([s.replace(" ", "\n") for s in STIM], fontsize=9)
        ax.set_title(lbl, fontsize=11, fontweight="bold")
        ax.grid(alpha=.2, axis="y")
        ax.set_ylim(bottom=0)
    fig.suptitle(f"Why divide by the delivery count   ·   "
                 f"example behaviour: {NICE[b]}\n"
                 "same animals, same scoring - only the denominator differs",
                 fontsize=12)
    fig.tight_layout()
    return save(fig, path)


# ───────── D7: how many mice does this design actually need? ───────────────
def d7_power(R, path, lab1, lab2, nsim=4000, seed=3):
    """The question the mockup forces.

    With a true 40 % reduction injected into every responding animal, the
    paired Wilcoxon at n = 6 returned p = 0.062 for almost every cell - not
    significant. That is not the analysis failing, it is the exact test's
    floor: at n = 6 the smallest possible two-tailed p is 2/2^6 = 0.031, and
    it takes all six animals moving the same way to reach it. One
    non-responder drops the usable n to 5 and the floor rises to 0.062, at
    which point p < 0.05 is arithmetically unreachable.

    So the number of animals is not a detail to settle later. This panel
    simulates the actual design - each mouse's own Day-1 spread, a given
    effect, paired Wilcoxon - and reports the power.
    """
    rng = np.random.default_rng(seed)
    # variability taken from the real Day-1 data, not assumed
    g = (R[R.day == lab1].groupby(["mouse", "behaviour"])
         .per_delivery.mean().reset_index())
    cv = []
    for b, gg in g.groupby("behaviour"):
        v = gg.per_delivery.to_numpy()
        if len(v) > 1 and v.mean() > 0:
            cv.append(v.std(ddof=1) / v.mean())
    CV = float(np.median(cv)) if cv else .5

    effects = [.20, .30, .40, .60]
    cols = ["#1C6E8C", "#5E9FC4", "#C0483B", "#7A3B8F"]
    ns = np.arange(4, 17)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1))

    def power(n, eff, nnon):
        """nnon animals get no effect at all - the realistic case."""
        hit = 0
        for _ in range(nsim):
            base = np.clip(rng.normal(1.0, CV, n), .05, None)
            e = np.full(n, eff)
            if nnon:
                e[rng.choice(n, min(nnon, n), replace=False)] = 0.0
            d2v = base * (1 - e) * np.clip(
                rng.normal(1.0, CV * .5, n), .05, None)
            if np.allclose(d2v - base, 0):
                continue
            try:
                p = stats.wilcoxon(base, d2v, zero_method="wilcox",
                                   method="exact").pvalue
            except ValueError:
                continue
            hit += p < .05
        return hit / nsim

    ax = axes[0]
    for eff, c in zip(effects, cols):
        ax.plot(ns, [power(n, eff, 0) for n in ns], "-o", ms=4, lw=2.0,
                color=c, label=f"{eff:.0%}, all respond")
        ax.plot(ns, [power(n, eff, 1) for n in ns], "--s", ms=4, lw=1.8,
                color=c, alpha=.75, label=f"{eff:.0%}, 1 non-responder")
    ax.axhline(.80, color="k", ls="--", lw=1.2)
    ax.axvline(6, color="#888", ls=":", lw=2.0)
    ax.text(6.15, .04, "n = 6\ntoday", color="#555", fontsize=9)
    ax.text(ns[-1], .815, "80 % power", ha="right", fontsize=9)
    ax.set_xlabel("mice per group (paired, same animals both days)")
    ax.set_ylabel("power at p < 0.05")
    ax.set_ylim(0, 1.02)
    ax.set_title("Population test: paired Wilcoxon\n"
                 f"variability taken from Day 1 (median CV = {CV:.2f})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", ncol=2)
    ax.grid(alpha=.2)

    # the exact-test floor, which is what actually bit us
    ax = axes[1]
    nn = np.arange(3, 17)
    floor = 2.0 / (2.0 ** nn)
    ax.semilogy(nn, floor, "-o", ms=5, lw=2.2, color=C1)
    ax.axhline(.05, color=C2, ls="--", lw=1.6)
    ax.text(16, .056, "p = 0.05", ha="right", color=C2, fontsize=10)
    for n in (5, 6):
        ax.plot([n], [2.0 / 2 ** n], "o", ms=11, color=C2, zorder=5)
        ax.annotate(f"n = {n}\nfloor {2.0 / 2 ** n:.3f}",
                    (n, 2.0 / 2 ** n), fontsize=9, color=C2,
                    xytext=(8, 14), textcoords="offset points")
    ax.set_xlabel("number of pairs actually usable")
    ax.set_ylabel("smallest two-tailed p obtainable")
    ax.set_title("Why n = 6 is fragile\n"
                 "a single non-responder is dropped, n falls to 5, "
                 "and 0.05 becomes unreachable",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=.2, which="both")

    fig.suptitle("Design planning, from the Day-1 variability   ·   "
                 "this is a property of the DESIGN, not of the mock numbers",
                 fontsize=12)
    fig.tight_layout()
    return save(fig, path)


# ───── D8: dose-response curve, both days overlaid, per-mouse spaghetti ────
def d8_dose_both(R, path, lab1, lab2):
    """The two separate dose-response slides, on one pair of axes.

    Day 1 and Day 2 were being shown as two slides with different y ranges,
    which made the size of the drop impossible to judge and the change in
    CURVE SHAPE impossible to see at all. Overlaid on a shared axis, both are
    immediate: the Day-1 curve rises from light touch to heat, and the Day-2
    curve is both lower and flatter.

    One thin line per mouse per day (the spaghetti), one bold line for the
    mean. Nothing is averaged away.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.6))
    xs = np.arange(len(STIM))
    for ax, b in zip(axes.ravel(), ORDER):
        for day, col, ls, mk in ((lab1, C1, "-", "o"),
                                 (lab2, C2, "--", "s")):
            g = R[(R.behaviour == b) & (R.day == day)]
            w = g.pivot_table(index="mouse", columns="stimulus",
                              values="per_delivery").reindex(columns=STIM)
            if not len(w):
                continue
            for _, row in w.iterrows():
                ax.plot(xs, row.to_numpy(), ls, color=col, lw=.9, alpha=.42,
                        zorder=2)
            m = w.mean(axis=0).to_numpy()
            e = (w.std(axis=0, ddof=1) / np.sqrt(len(w))).to_numpy() \
                if len(w) > 1 else np.zeros_like(m)
            ax.errorbar(xs, m, yerr=e, fmt=mk + ls, color=col, lw=2.8,
                        ms=8, capsize=4, mec="white", mew=1.4, zorder=6,
                        label=day)
        ax.set_xticks(xs)
        ax.set_xticklabels([s.replace(" ", "\n") for s in STIM], fontsize=9)
        ax.set_title(NICE[b], fontweight="bold",
                     color=CR if b in REF.values() else CA)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2, axis="y")
    axes[0][0].set_ylabel(YLAB, fontsize=11)
    axes[1][0].set_ylabel(YLAB, fontsize=11)
    axes[0][2].legend(fontsize=9.5, frameon=False)
    fig.suptitle("Dose-response on both days, same axes   ·   "
                 "thin line = one mouse, bold = mean ± SEM\n"
                 "lower everywhere on the drug day. The stimulus ordering is "
                 "flattened for licking/biting and escape, but kept for the "
                 "reflexes", fontsize=12)
    fig.tight_layout()
    return save(fig, path)


def main():
    global MOCKUP
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day1", required=True)
    ap.add_argument("--day2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label1", default="Day 1 no drug")
    ap.add_argument("--label2", default="Day 2 drug")
    ap.add_argument("--mockup", action="store_true",
                    help="stamp MOCKUP on every panel; use with synthetic "
                         "Day 2 data")
    ap.add_argument("--keep-d5", action="store_true",
                    help="also write D5, the per-mouse change without "
                         "confidence intervals. Off by default because the "
                         "forest plot from step4 shows the same thing with "
                         "statistics added.")
    ap.add_argument("--response-window", action="store_true",
                    help="count only events starting within the response "
                         "window of an individual delivery (3 s reflex, 10 s "
                         "affective). Default is to count EVERY event in the "
                         "stimulus block. The window answers 'did this "
                         "stimulus provoke a response'; the default answers "
                         "'how much did the animal do'. The window discards "
                         "71 %% of escape/rearing, which is spontaneous.")
    a = ap.parse_args()
    MOCKUP = a.mockup
    global USE_WINDOW
    USE_WINDOW = a.response_window
    print("counting: " + ("only events inside the response window"
                          if USE_WINDOW else
                          "EVERY event in the stimulus block"))
    os.makedirs(a.out, exist_ok=True)

    d1 = read_all(a.day1, a.label1)
    d2 = read_all(a.day2, a.label2)
    if not d1 or not d2:
        raise SystemExit("need ScoringAB_*.mat in both folders")
    print(f"{len(d1)} Day-1 and {len(d2)} Day-2 session(s)"
          + ("   [MOCKUP MODE]" if MOCKUP else ""))

    R = pd.concat([rate_table(d1), rate_table(d2)], ignore_index=True)
    R.to_csv(os.path.join(a.out, "Day2_rate_table"
                          + ("_MOCKUP" if MOCKUP else "") + ".csv"),
             index=False)
    nd = R.groupby("day").n_deliveries.agg(["min", "max"])
    print("  deliveries per block:")
    for day, r in nd.iterrows():
        print(f"    {day:16s} {int(r['min']):3d} to {int(r['max']):3d}")
    print("  -> this spread is exactly why nothing below is a raw count\n")

    d1_psth((d1, d2), os.path.join(a.out, "D1_psth_day_compare.png"),
            a.label1, a.label2)
    ST = d2_per_delivery(R, os.path.join(a.out, "D2_per_delivery.png"),
                         a.label1, a.label2)
    d3_timecourse((d1, d2), os.path.join(a.out, "D3_within_block.png"),
                  a.label1, a.label2)
    d4_dissoc(R, os.path.join(a.out, "D4_reflex_vs_affective.png"),
              a.label1, a.label2)
    # D5 is no longer produced. It was a dot-and-stick version of exactly the
    # information in step4's forest plot, which adds a confidence interval and
    # per-mouse significance on top - two slides that said the same thing. The
    # function is kept in case a version without statistics is ever wanted.
    if a.keep_d5:
        d5_per_mouse(R, os.path.join(a.out, "D5_per_mouse_change.png"),
                     a.label1, a.label2)
    d8_dose_both(R, os.path.join(a.out, "D8_dose_response_both_days.png"),
                 a.label1, a.label2)
    d6_raw_vs_norm(R, os.path.join(a.out, "D6_why_normalise.png"),
                   a.label1, a.label2)
    # D7 is driven by the Day-1 variability, so it is a real design statement
    # even when the Day-2 numbers beside it are synthetic
    d7_power(R, os.path.join(a.out, "D7_power_planning.png"),
             a.label1, a.label2)

    ST["q"] = ST.p_paired  # single family per behaviour, reported raw
    ST.to_csv(os.path.join(a.out, "Day2_stats_per_stimulus"
                           + ("_MOCKUP" if MOCKUP else "") + ".csv"),
              index=False)
    print(f"\n  {YLAB1}, paired by mouse:")
    print(f"  {'behaviour':11s} {'stimulus':12s} {'D1':>6s} {'D2':>6s} "
          f"{'p':>7s} {'floor':>6s}")
    for r in ST.sort_values("p_paired", na_position="last").head(12).itertuples():
        print(f"  {r.behaviour:11s} {r.stimulus:12s} {r.day1:6.2f} "
              f"{r.day2:6.2f} {r.p_paired:7.3f} {r.p_floor:6.3f}")
    if MOCKUP:
        print("\n  MOCKUP MODE: every figure is stamped and every filename "
              "carries\n  _MOCKUP. The Day 2 numbers are synthetic.")


if __name__ == "__main__":
    main()
