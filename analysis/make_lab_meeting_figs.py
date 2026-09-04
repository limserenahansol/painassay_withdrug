"""make_lab_meeting_figs.py  -  slide-ready figures for the lab meeting.

Purpose-built for projection: large fonts, few colours, one message per panel.
The dense analysis figures from step1/step3 stay where they are; these are the
ones to put on a screen.

USAGE
    python make_lab_meeting_figs.py                       # Day 1 only
    python make_lab_meeting_figs.py --day2 <folder>       # once Day 2 exists

OUTPUT  ->  lab_meeting\\figs\\
    F1_design.png            the session timeline
    F2_dose_response.png     response vs stimulus intensity - the main result
    F3_per_delivery.png      the same, per stimulus delivered
    F4_baseline_vs_block.png why escape/rearing runs the other way
    F5_per_mouse.png         all six mice, one line each
    F6_day1_vs_day2.png      the drug comparison (placeholder until Day 2)

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

DAY1 = (r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos"
        r"\output_corrected")
OUT = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\lab_meeting\figs"

# innocuous -> noxious, which is the axis the whole story hangs on
STIM = ["Light touch", "Mild touch", "Heat", "Pin prick"]
REFLEX = ["withdrawal", "flinch"]
AFFECT = ["attending", "lickbite", "guarding", "escape"]
NICE = {"withdrawal": "Paw withdrawal", "flinch": "Flinch",
        "attending": "Paw attending", "lickbite": "Licking / biting",
        "guarding": "Guarding", "escape": "Escape / rearing"}
CR, CA = "#C0483B", "#1C6E8C"          # reflexive red, affective blue
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15,
                     "axes.labelsize": 13, "figure.dpi": 130})


def load(folder, label):
    L = pd.read_csv(os.path.join(folder, "BlockMeasures_long.csv"))
    key = ["mouse", "behaviour"]
    base = L[L.kind == "baseline"].set_index(key)["rate_per_min"].rename("base")
    base = base[~base.index.duplicated()]
    B = L[L.kind == "stimulus"].join(base, on=key)
    B["delta"] = B.rate_per_min - B.base
    B["day"] = label
    L["day"] = label
    return L, B


# ───────────────────────── F1: the design ──────────────────────────────────
def f1_design(path):
    fig, ax = plt.subplots(figsize=(13, 3.4))
    segs = [(0, 5, "BASELINE\nno stimulus", "#D9D9D9"),
            (5, 1, "", "#FFFFFF"),
            (6, 5, "stimulus 1", "#BBD5E5"),
            (11, 1, "", "#FFFFFF"),
            (12, 5, "stimulus 2", "#9CC3DA"),
            (17, 1, "", "#FFFFFF"),
            (18, 5, "stimulus 3", "#7DB1CF"),
            (23, 1, "", "#FFFFFF"),
            (24, 5, "stimulus 4", "#5E9FC4")]
    for x, w, lab, col in segs:
        ax.add_patch(Rectangle((x, 0), w, 1, facecolor=col,
                               edgecolor="#444", lw=1.4))
        if lab:
            ax.text(x + w / 2, .5, lab, ha="center", va="center",
                    fontsize=12, fontweight="bold" if "BASE" in lab else "normal")
    for x in (5, 11, 17, 23):
        ax.text(x + .5, -.22, "rest", ha="center", fontsize=9, color="#777")
    ax.set_xlim(-.4, 29.4)
    ax.set_ylim(-.55, 1.5)
    ax.set_yticks([])
    ax.set_xticks(range(0, 30, 5))
    ax.set_xlabel("minutes")

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ─────────────── F2: dose-response, the main result ────────────────────────
def f2_dose(B, path, value="delta",
            ylab="events / min above baseline"):
    behs = REFLEX + AFFECT
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 6.3))
    for ax, b in zip(axes.ravel(), behs):
        col = CR if b in REFLEX else CA
        med, lo, hi, pts = [], [], [], []
        for s in STIM:
            v = B[(B.behaviour == b) & (B.stimulus == s)][value].astype(float)
            v = v[np.isfinite(v)]
            pts.append(v.to_numpy())
            med.append(np.median(v) if len(v) else np.nan)
        x = np.arange(len(STIM))
        for i, v in enumerate(pts):
            ax.plot(np.full(len(v), i) + np.linspace(-.13, .13, max(len(v), 1)),
                    v, "o", ms=6, color=col, alpha=.45, mec="none")
        ax.plot(x, med, "-o", color=col, lw=2.6, ms=11, mfc="white", mew=2.4,
                zorder=5)
        ax.axhline(0, color="#888", lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(["Light\ntouch", "Mild\ntouch", "Heat", "Pin\nprick"])
        ax.set_title(NICE[b], color=col, fontweight="bold")
        ax.grid(alpha=.22, axis="y")
        if b in ("withdrawal", "attending"):
            ax.set_ylabel(ylab)
    # No suptitle: the slide already carries the message, and a second title
    # only shrinks the panels. A compact colour key is enough.
    axes[0][0].text(.02, .97, "reflexive", transform=axes[0][0].transAxes,
                    color=CR, fontsize=12, fontweight="bold", va="top")
    axes[1][0].text(.02, .97, "affective", transform=axes[1][0].transAxes,
                    color=CA, fontsize=12, fontweight="bold", va="top")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ─────────────── F4: baseline vs block, the escape story ───────────────────
def f4_baseline(L, B, path):
    behs = REFLEX + AFFECT
    fig, ax = plt.subplots(figsize=(14, 5.2))
    w = 0.36
    x = np.arange(len(behs))
    bs, blk = [], []
    for b in behs:
        bs.append(L[(L.kind == "baseline") & (L.behaviour == b)]
                  ["rate_per_min"].median())
        blk.append(B[B.behaviour == b]["rate_per_min"].median())
    ax.bar(x - w / 2, bs, w, label="baseline (no stimulus)",
           color="#BFBFBF", edgecolor="#444")
    ax.bar(x + w / 2, blk, w, label="during stimulus blocks",
           color="#1C6E8C", edgecolor="#444")
    for i, b in enumerate(behs):
        if bs[i] > blk[i]:
            ax.annotate("goes DOWN", (i, max(bs[i], blk[i]) + .25),
                        ha="center", fontsize=12, color="#C0483B",
                        fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[b] for b in behs], rotation=18, ha="right")
    ax.set_ylabel("events / min")
    ax.legend(frameon=False)
    ax.grid(alpha=.22, axis="y")

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ─────────────── F5: every mouse, one line each ────────────────────────────
def f5_mice(B, path):
    behs = REFLEX + AFFECT
    mice = sorted(B.mouse.dropna().unique())
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 6.3))
    for ax, b in zip(axes.ravel(), behs):
        for j, m in enumerate(mice):
            y = [B[(B.behaviour == b) & (B.stimulus == s) & (B.mouse == m)]
                 ["delta"].mean() for s in STIM]
            ax.plot(range(len(STIM)), y, "-o", color=cmap(j % 10), lw=1.8,
                    ms=6, label=m if b == "withdrawal" else None, alpha=.85)
        ax.axhline(0, color="#888", lw=1, ls="--")
        ax.set_xticks(range(len(STIM)))
        ax.set_xticklabels(["Light", "Mild", "Heat", "Pin"], fontsize=11)
        ax.set_title(NICE[b], fontweight="bold")
        ax.grid(alpha=.22, axis="y")
    axes[0][0].set_ylabel("events / min above baseline")
    axes[1][0].set_ylabel("events / min above baseline")
    axes[0][0].legend(fontsize=10, ncol=3, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ─────────────── F6: Day 1 vs Day 2 ────────────────────────────────────────
def f6_days(S, path, have2, measure="n_events",
            ylab="events in the whole session"):
    """Day 1 vs Day 2, showing the paired structure AND the group summary.

    Both are wanted: the paired lines are the actual experiment (same six
    animals twice), and the group mean is what a reader looks for first. They
    are drawn together so nobody has to guess which one the p value came from.
    """
    behs = REFLEX + AFFECT
    days = sorted(S.day.unique())
    fig, axes = plt.subplots(1, len(behs), figsize=(16, 4.8))
    for ax, b in zip(axes, behs):
        data = [S[(S.behaviour == b) & (S.day == d)][measure]
                .astype(float).dropna().to_numpy() for d in days]
        # group summary: mean +/- SEM, drawn as a heavy marker
        for i, v in enumerate(data):
            if not len(v):
                continue
            m = np.mean(v)
            sem = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            ax.errorbar([i + 1], [m], yerr=[sem], fmt="s",
                        color="#C0483B" if i else "#444", ms=12, capsize=7,
                        lw=2.4, zorder=6, mec="white", mew=1.6)
        # paired structure: one thin line per mouse
        if len(days) == 2:
            w = S[S.behaviour == b].pivot_table(index="mouse", columns="day",
                                                values=measure)
            for _, rr in w.iterrows():
                if np.isfinite(rr[days[0]]) and np.isfinite(rr[days[1]]):
                    ax.plot([1, 2], [rr[days[0]], rr[days[1]]], "-o",
                            color="#777", lw=1.2, ms=5, alpha=.8, zorder=3)
        else:
            v = data[0]
            ax.plot(np.full(len(v), 1) + np.linspace(-.10, .10, max(len(v), 1)),
                    v, "o", color="#777", ms=6, alpha=.8, zorder=3)
        ax.set_xlim(.55, len(days) + .45)
        ax.set_xticks(range(1, len(days) + 1))
        ax.set_xticklabels([d.replace(" ", chr(10)) for d in days], fontsize=11)
        ax.set_title(NICE[b], fontsize=12, fontweight="bold")
        ax.grid(alpha=.22, axis="y")
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel(ylab)
    axes[0].text(.03, .97, "grey lines = individual mice" + chr(10)
                 + "square = mean ± SEM",
                 transform=axes[0].transAxes, fontsize=9, va="top",
                 color="#444")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def session_measures(folder, label):
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "s3", os.path.join(here, "step3_day_comparison.py"))
    s3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s3)
    return s3.session_measures(folder, label)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day1", default=DAY1,
                    help="the folder to describe. Point it at the Day 2 "
                         "corrected folder with --label to get the same "
                         "descriptive set for the drug day on its own.")
    ap.add_argument("--label", default="Day 1 no drug",
                    help="what to call the --day1 folder in the figures")
    ap.add_argument("--day2", default=None)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    L1, B1 = load(a.day1, a.label)
    L, B = L1, B1
    S = session_measures(a.day1, a.label)
    have2 = False
    if a.day2 and os.path.exists(os.path.join(a.day2, "BlockMeasures_long.csv")):
        L2, B2 = load(a.day2, "Day 2 drug")
        L = pd.concat([L1, L2], ignore_index=True)
        B = pd.concat([B1, B2], ignore_index=True)
        S = pd.concat([S, session_measures(a.day2, "Day 2 drug")],
                      ignore_index=True)
        have2 = True

    print(f"{B.mouse.nunique()} mice, {'2 days' if have2 else 'Day 1 only'}")
    f1_design(os.path.join(a.out, "F1_design.png"))
    f2_dose(B1, os.path.join(a.out, "F2_dose_response.png"))
    # Raw counts, no normalisation. Every block is a fixed 300 s, so the
    # whole number of events in a block is directly comparable between
    # stimuli and between days - no per-minute conversion needed.
    f2_dose(B1, os.path.join(a.out, "F2b_event_counts.png"),
            value="n_bouts",
            ylab="events in the 5 min block")
    f2_dose(B1, os.path.join(a.out, "F3_per_delivery.png"),
            value="n_per_delivery",
            ylab="events per stimulus delivered")
    f4_baseline(L1, B1, os.path.join(a.out, "F4_baseline_vs_block.png"))
    f5_mice(B1, os.path.join(a.out, "F5_per_mouse.png"))
    f6_days(S, os.path.join(a.out, "F6_day1_vs_day2_counts.png"), have2,
            measure="n_events", ylab="events in the whole session")
    f6_days(S, os.path.join(a.out, "F6b_day1_vs_day2_rate.png"), have2,
            measure="events_per_min", ylab="events / min")
    S.to_csv(os.path.join(a.out, "SessionMeasures.csv"), index=False)


if __name__ == "__main__":
    main()
