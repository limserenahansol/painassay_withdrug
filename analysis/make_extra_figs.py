"""make_extra_figs.py  -  the comparisons that the block summaries cannot show.

These need the frame-level scoring, not just per-block totals.

  E1  PERI-STIMULUS TIME HISTOGRAM
      Align every delivery and plot the probability that a behaviour is
      happening, second by second, from -5 s to +15 s. This is the figure that
      most directly parallels the peri-event dF/F you will make from the mini1p
      data, so the behaviour and the imaging can be put side by side.

  E2  RESPONSE PROBABILITY PER DELIVERY
      For each delivery, did the behaviour occur within a response window?
      That is one Bernoulli trial per delivery - about 22 per block instead of
      one rate - which is far better powered than a block mean. This is the
      measure to use if you want a real test out of six animals.

  E3  WITHIN-BLOCK TIME COURSE
      Each 5 min block split into 1 min bins. Answers whether the response
      habituates or builds up, and the drug may change the shape rather than
      the level.

  E4  BLOCK POSITION CONTROL
      Response by position in the session (1st, 2nd, 3rd, 4th block) rather
      than by stimulus. The order was randomised, so if the response tracks
      stimulus and not position, the randomisation did its job.

  E5  REFLEXIVE vs AFFECTIVE
      One point per mouse. Does an animal that withdraws a lot also lick a
      lot? If the two classes dissociate, scoring them separately was the
      right call. Sex is marked.

USAGE
    python make_extra_figs.py
    python make_extra_figs.py --day2 <folder>

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
from scipy.io import loadmat

DAY1 = (r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos"
        r"\output_corrected")
OUT = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\lab_meeting\figs"

AFF = {1: "attending", 2: "lickbite", 3: "guarding", 4: "escape"}
REF = {1: "withdrawal", 2: "flinch"}
ORDER = ["withdrawal", "flinch", "attending", "lickbite", "guarding", "escape"]
NICE = {"withdrawal": "Paw withdrawal", "flinch": "Flinch",
        "attending": "Paw attending", "lickbite": "Licking / biting",
        "guarding": "Guarding", "escape": "Escape / rearing"}
STIM = ["Light touch", "Mild touch", "Heat", "Pin prick"]
SCOL = {"Light touch": "#9CC3DA", "Mild touch": "#5E9FC4",
        "Heat": "#C0483B", "Pin prick": "#7A3B8F"}
CR, CA = "#C0483B", "#1C6E8C"
PRE, POST = 5.0, 15.0           # PSTH window around each delivery
RESP_S = 3.0                    # response window for E2
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.labelsize": 12, "figure.dpi": 130})


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


def read_all(folder, label):
    """Frame-level scoring plus delivery times, one entry per session."""
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
    """Per-delivery matrix of 'is this behaviour on', aligned to delivery."""
    a, b = int(pre * fps), int(post * fps)
    rows = []
    for f in dF:
        lo, hi = f - a, f + b
        if lo < 1 or hi > len(S):
            continue
        rows.append((S[lo - 1:hi - 1] == code).astype(float))
    return np.vstack(rows) if rows else np.empty((0, a + b))


def reflex_occupancy(rx, code, fps, dF, pre, post, n):
    a, b = int(pre * fps), int(post * fps)
    hit = np.zeros(n + 2)
    if rx.size:
        f = rx[rx[:, 1] == code, 0].astype(int)
        f = f[(f >= 1) & (f <= n)]
        hit[f] = 1
    rows = []
    for f in dF:
        lo, hi = f - a, f + b
        if lo < 1 or hi > n:
            continue
        rows.append(hit[lo:hi])
    return np.vstack(rows) if rows else np.empty((0, a + b))


# ─────────────────── E1: peri-stimulus time histogram ──────────────────────
def e1_psth(data, path):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 6.6))
    fps = data[0]["fps"]
    t = np.arange(-int(PRE * fps), int(POST * fps)) / fps
    for ax, b in zip(axes.ravel(), ORDER):
        for s in STIM:
            per_mouse = []
            for d in data:
                idx = [i + 1 for i, nm in enumerate(d["names"]) if nm == s]
                if not idx:
                    continue
                sel = d["dF"][np.isin(d["dT"], idx)]
                if not len(sel):
                    continue
                if b in REF.values():
                    code = [k for k, v in REF.items() if v == b][0]
                    m = reflex_occupancy(d["rx"], code, d["fps"], sel,
                                         PRE, POST, d["n"])
                else:
                    code = [k for k, v in AFF.items() if v == b][0]
                    m = occupancy(d["score"], code, d["fps"], sel, PRE, POST)
                if m.size:
                    per_mouse.append(m.mean(axis=0))
            if not per_mouse:
                continue
            y = np.mean(np.vstack(per_mouse), axis=0)
            # smooth to 0.5 s so the reflex marks are visible at all
            k = max(1, int(.5 * fps))
            y = np.convolve(y, np.ones(k) / k, mode="same")
            ax.plot(t[:len(y)], y, color=SCOL[s], lw=2.2, label=s)
        ax.axvline(0, color="k", lw=1.4, ls="--")
        ax.set_title(NICE[b], fontweight="bold",
                     color=CR if b in REF.values() else CA)
        ax.grid(alpha=.2)
        ax.set_xlim(-PRE, POST)
    axes[1][0].set_xlabel("seconds from stimulus delivery")
    axes[0][0].set_ylabel("probability behaviour is occurring")
    axes[1][0].set_ylabel("probability behaviour is occurring")
    axes[0][2].legend(fontsize=9, frameon=False)
    # The affective traces all drop to exactly zero at t = 0. That is the
    # SCORER, not the mouse: to tap a delivery key the held behaviour key has
    # to be released. Say so on the figure, or a reader will interpret it.
    fig.text(.5, -.02,
             "note: the affective traces dip to zero exactly at t = 0 because "
             "marking a delivery means releasing the held behaviour key — "
             "that notch is the scorer's hand, not the animal",
             ha="center", fontsize=11, color="#C0483B")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ────────────── E2: response probability per delivery ──────────────────────
def e2_prob(data, path, csvpath, resp_s=RESP_S, isolated=False):
    """P(behaviour occurs within resp_s of a delivery).

    isolated=False   every delivery is used. Cheap, but with a median
                     inter-delivery interval of 2.7 s another delivery falls
                     inside the window for most trials, so a response cannot
                     be attributed to one stimulus.
    isolated=True    only deliveries with resp_s of clearance before the next
                     one. Attribution is unambiguous. At 10 s that keeps 122
                     of 535 deliveries - still five times more observations
                     than the 24 block means.
    """
    rows = []
    for d in data:
        for s in STIM:
            idx = [i + 1 for i, nm in enumerate(d["names"]) if nm == s]
            sel = d["dF"][np.isin(d["dT"], idx)] if idx else np.array([])
            if not len(sel):
                continue
            if isolated:
                allf = np.sort(d["dF"].astype(float))
                keep = []
                for f in sel:
                    nxt = allf[allf > f]
                    if not len(nxt) or (nxt[0] - f) / d["fps"] >= resp_s:
                        keep.append(f)
                sel = np.array(keep, dtype=int)
                if not len(sel):
                    continue
            for b in ORDER:
                if b in REF.values():
                    code = [k for k, v in REF.items() if v == b][0]
                    m = reflex_occupancy(d["rx"], code, d["fps"], sel,
                                         0, resp_s, d["n"])
                else:
                    code = [k for k, v in AFF.items() if v == b][0]
                    m = occupancy(d["score"], code, d["fps"], sel, 0, resp_s)
                if not m.size:
                    continue
                hit = (m.sum(axis=1) > 0).astype(int)
                rows.append(dict(day=d["day"], mouse=d["mouse"], sex=d["sex"],
                                 stimulus=s, behaviour=b,
                                 n_deliveries=len(hit),
                                 n_responded=int(hit.sum()),
                                 p_response=float(hit.mean())))
    R = pd.DataFrame(rows)
    R.to_csv(csvpath, index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 6.6))
    for ax, b in zip(axes.ravel(), ORDER):
        for i, s in enumerate(STIM):
            v = R[(R.behaviour == b) & (R.stimulus == s)]["p_response"]
            v = v[np.isfinite(v)]
            ax.plot(np.full(len(v), i) + np.linspace(-.13, .13, max(len(v), 1)),
                    v, "o", ms=6, color=SCOL[s], alpha=.55, mec="none")
            if len(v):
                ax.plot([i], [v.median()], "o", ms=12, mfc="white",
                        mec=SCOL[s], mew=2.6, zorder=5)
        ax.set_xticks(range(len(STIM)))
        ax.set_xticklabels(["Light\ntouch", "Mild\ntouch", "Heat", "Pin\nprick"])
        ax.set_ylim(-.04, 1.04)
        ax.set_title(NICE[b], fontweight="bold",
                     color=CR if b in REF.values() else CA)
        ax.grid(alpha=.2, axis="y")
    axes[0][0].set_ylabel(f"P(response within {resp_s:g} s)")
    axes[1][0].set_ylabel(f"P(response within {resp_s:g} s)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")
    print(f"  {csvpath}")
    return R


# ────────────── E3: within-block time course ───────────────────────────────
def e3_timecourse(data, path, nbin=5, blocklen=300.0):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 6.6))
    for ax, b in zip(axes.ravel(), ORDER):
        for s in STIM:
            acc = []
            for d in data:
                idx = [i + 1 for i, nm in enumerate(d["names"]) if nm == s]
                if not idx:
                    continue
                sel = d["dF"][np.isin(d["dT"], idx)]
                if not len(sel):
                    continue
                f0 = sel.min()
                per = []
                for k in range(nbin):
                    a = int(f0 + k * blocklen / nbin * d["fps"])
                    z = int(f0 + (k + 1) * blocklen / nbin * d["fps"])
                    z = min(z, d["n"])
                    if z <= a:
                        per.append(np.nan)
                        continue
                    if b in REF.values():
                        code = [kk for kk, v in REF.items() if v == b][0]
                        cnt = (int(np.sum((d["rx"][:, 1] == code)
                                          & (d["rx"][:, 0] >= a)
                                          & (d["rx"][:, 0] < z)))
                               if d["rx"].size else 0)
                    else:
                        code = [kk for kk, v in AFF.items() if v == b][0]
                        seg = (d["score"][a - 1:z - 1] == code).astype(np.int8)
                        dd = np.diff(np.concatenate(([0], seg, [0])))
                        cnt = int((dd == 1).sum())
                    per.append(cnt / ((z - a) / d["fps"] / 60))
                acc.append(per)
            if acc:
                y = np.nanmean(np.array(acc, float), axis=0)
                ax.plot(np.arange(nbin) + .5, y, "-o", color=SCOL[s], lw=2.0,
                        ms=6, label=s)
        ax.set_xticks(np.arange(nbin) + .5)
        ax.set_xticklabels([f"{k+1}" for k in range(nbin)])
        ax.set_title(NICE[b], fontweight="bold",
                     color=CR if b in REF.values() else CA)
        ax.grid(alpha=.2, axis="y")
    axes[1][0].set_xlabel("minute within the 5 min block")
    axes[0][0].set_ylabel("events / min")
    axes[1][0].set_ylabel("events / min")
    axes[0][2].legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ────────────── E4: block position control ─────────────────────────────────
def e4_position(folder, path):
    L = pd.read_csv(os.path.join(folder, "BlockMeasures_long.csv"))
    S = L[L.kind == "stimulus"]
    fig, axes = plt.subplots(1, 6, figsize=(16, 3.8))
    for ax, b in zip(axes, ORDER):
        for p in sorted(S.pos.unique()):
            v = S[(S.behaviour == b) & (S.pos == p)]["n_bouts"].astype(float)
            v = v[np.isfinite(v)]
            ax.plot(np.full(len(v), p) + np.linspace(-.14, .14, max(len(v), 1)),
                    v, "o", ms=5, color="#888", alpha=.6, mec="none")
            if len(v):
                ax.plot([p], [v.median()], "s", ms=10, color="#1F2937",
                        zorder=5)
        ax.set_xticks(sorted(S.pos.unique()))
        ax.set_title(NICE[b], fontsize=11, fontweight="bold")
        ax.grid(alpha=.2, axis="y")
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("events in block")
    axes[0].set_xlabel("block position in session")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ────────────── E5: reflexive vs affective ─────────────────────────────────
def e5_dissoc(folder, path):
    L = pd.read_csv(os.path.join(folder, "BlockMeasures_long.csv"))
    S = L[L.kind == "stimulus"]
    tot = S.groupby(["mouse", "sex", "behaviour"]).n_bouts.sum().reset_index()
    w = tot.pivot_table(index=["mouse", "sex"], columns="behaviour",
                        values="n_bouts").reset_index()
    w["reflexive"] = w.get("withdrawal", 0) + w.get("flinch", 0)
    w["affective"] = (w.get("attending", 0) + w.get("lickbite", 0)
                      + w.get("guarding", 0))
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for _, r in w.iterrows():
        mk = "o" if str(r["sex"]).upper().startswith("F") else "^"
        ax.plot(r["reflexive"], r["affective"], mk, ms=14, color=CA,
                alpha=.85, mec="white", mew=1.6)
        ax.annotate(r["mouse"], (r["reflexive"], r["affective"]),
                    textcoords="offset points", xytext=(11, -4), fontsize=11)
    if len(w) > 2:
        rho = np.corrcoef(w["reflexive"], w["affective"])[0, 1]
        ax.set_title(f"Pearson r = {rho:+.2f}   (n = {len(w)} mice)",
                     fontsize=13)
    ax.set_xlabel("reflexive events, whole session\n(withdrawal + flinch)")
    ax.set_ylabel("affective events, whole session\n"
                  "(attending + lick/bite + guarding)")
    ax.grid(alpha=.25)
    ax.plot([], [], "o", color=CA, ms=10, label="female")
    ax.plot([], [], "^", color=CA, ms=10, label="male")
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day1", default=DAY1)
    ap.add_argument("--day2", default=None)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d1 = read_all(a.day1, "Day 1 no drug")
    if not d1:
        raise SystemExit(f"no ScoringAB_*.mat in {a.day1}")
    print(f"{len(d1)} Day-1 session(s)")
    e1_psth(d1, os.path.join(a.out, "E1_psth.png"))
    # 3 s over every delivery, and 10 s over the isolated ones. Neither is
    # "the" answer: 3 s catches the reflexes but cuts the affective response
    # short, 10 s catches the affective response but only 23 % of deliveries
    # have 10 s of clearance.
    e2_prob(d1, os.path.join(a.out, "E2_response_probability.png"),
            os.path.join(a.out, "ResponseProbability_3s_all.csv"),
            resp_s=3.0, isolated=False)
    e2_prob(d1, os.path.join(a.out, "E2b_response_probability_10s.png"),
            os.path.join(a.out, "ResponseProbability_10s_isolated.csv"),
            resp_s=10.0, isolated=True)
    e3_timecourse(d1, os.path.join(a.out, "E3_within_block_timecourse.png"))
    e4_position(a.day1, os.path.join(a.out, "E4_block_position.png"))
    e5_dissoc(a.day1, os.path.join(a.out, "E5_reflex_vs_affective.png"))


if __name__ == "__main__":
    main()
