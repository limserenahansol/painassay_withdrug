"""Behaviour analysis of one imaging day: baseline vs injection, per mouse, SBI-553 vs vehicle.

Reads the final scoring files (manual deliveries / reflexes / affective behaviours + automatic
general movement and grooming; behavior_final_move_groom.py). Camera frame n = Inscopix frame n.

Per session
  per stimulus  n deliveries; % deliveries with a withdrawal / a flinch (a tap from the delivery
                to 2 s later or the next delivery); withdrawal latency; % deliveries followed by
                movement / escape-rearing / grooming / attending / lick-bite starting within 0-3 s
  whole session and per schedule block: % time in each held behaviour (attending, lick/bite,
                guarding, escape/rearing, general movement, grooming) and bouts per minute
Event-locked  probability of each behaviour around the deliveries (-2..+5 s, 0.1-s bins)
Statistics    per mouse and stimulus, baseline vs injection on the deliveries (Fisher exact test,
              two-sided); n = 2 mice per arm, so arm effects are shown, not tested

Outputs (--out-dir): behav_<day>_per_session_stimulus.csv, behav_<day>_held.csv,
behav_<day>_change.csv, behav_<day>_reflex.png, behav_<day>_affective.png,
behav_<day>_eventlocked.png

  python behavior_day_analysis.py --final-dir DIR --day d1 --out-dir OUT
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from behavior_supervised import BLOCKS  # noqa: E402

FPS = 30.0
ARM = {"5941": "SBI-553", "5939": "SBI-553", "5937": "vehicle", "1191": "vehicle"}
ARMC = {"SBI-553": "#c0392b", "vehicle": "#2471a3"}
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]
HELD = [("attending", "score", 1), ("lick/bite", "score", 2), ("guarding", "score", 3),
        ("escape/rearing", "score", 4), ("general movement", "moveVec", None), ("grooming", "groomVec", None)]


def spans(m):
    d = np.diff(np.r_[0, np.asarray(m, int), 0])
    return np.where(d == 1)[0], np.where(d == -1)[0]


def load(p):
    S = sio.loadmat(p, squeeze_me=True)
    n = int(S["nUsed"])
    d = dict(mouse=str(S["mouseID"]), phase="baseline" if str(S["phase"]).lower().startswith("base") else "injection",
             n=n, dF=np.atleast_1d(S["dFrames"]).astype(int) - 1, dT=np.atleast_1d(S["dTypes"]).astype(int),
             stim=[str(x) for x in np.atleast_1d(S["stimNames"])], file=p)
    re = np.atleast_2d(S["reflexEvents"]).astype(int).reshape(-1, 2)
    d["W"], d["F"] = np.sort(re[re[:, 1] == 1, 0] - 1), np.sort(re[re[:, 1] == 2, 0] - 1)
    sc = np.atleast_1d(S["score"]).astype(int)[:n]
    d["held"] = {}
    for nm, fld, code in HELD:
        v = sc == code if fld == "score" else np.asarray(S[fld]).astype(bool).ravel()[:n]
        d["held"][nm] = v
    o = np.argsort(d["dF"])
    d["dF"], d["dT"] = d["dF"][o], d["dT"][o]
    return d


def per_delivery(d):
    rows = []
    for i, (f, k) in enumerate(zip(d["dF"], d["dT"])):
        nxt = d["dF"][i + 1] if i + 1 < len(d["dF"]) else d["n"]
        hi = min(f + int(2 * FPS), nxt - 1)
        w = d["W"][(d["W"] >= f) & (d["W"] <= hi)]
        r = dict(delivery=i + 1, frame=int(f) + 1, t_s=f / FPS, stim=d["stim"][k - 1] if 1 <= k <= 4 else "?",
                 withdrawal=int(len(w) > 0), flinch=int(np.any((d["F"] >= f) & (d["F"] <= hi))),
                 wd_latency_s=(w[0] - f) / FPS if len(w) else np.nan)
        for nm, v in d["held"].items():
            on, _ = spans(v)
            r[f"{nm} starts 0-3 s"] = int(np.any((on >= f) & (on <= f + 3 * FPS)))
        rows.append(r)
    return pd.DataFrame(rows)


def held_table(d):
    rows = []
    for b, s0, s1 in [("whole session", 0, d["n"] / FPS)] + BLOCKS:
        sl = slice(int(s0 * FPS), min(d["n"], int(s1 * FPS)))
        dur = (sl.stop - sl.start) / FPS
        label = b if b in ("whole session", "baseline") else f"{b} ({d['stim'][int(b[-1]) - 1]})"
        for nm, v in d["held"].items():
            on, _ = spans(v[sl])
            rows.append(dict(block=label, behaviour=nm, pct_time=100 * v[sl].mean(), bouts_per_min=len(on) / dur * 60))
    return pd.DataFrame(rows)


def event_locked(d, behav, lo=-2.0, hi=5.0, step=0.1):
    edges = np.arange(lo, hi + step / 2, step)
    out = {}
    for nm in behav:
        if nm in ("withdrawal", "flinch"):
            ev = d["W"] if nm == "withdrawal" else d["F"]
            M = np.array([np.histogram((ev - f) / FPS, edges)[0] > 0 for f in d["dF"]], float)
        else:
            v = d["held"][nm]
            M = np.array([[v[int(np.clip(f + t * FPS, 0, d["n"] - 1))] for t in edges[:-1] + step / 2] for f in d["dF"]], float)
        out[nm] = M
    return edges[:-1] + step / 2, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-dir", required=True)
    ap.add_argument("--day", default="d1")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", default="", help="treatment of this day, e.g. 5941=vehicle,5939=vehicle,5937=SBI-553,1191=SBI-553 "
                                               "(Day 2 crossover); default = the Day-1 groups")
    a = ap.parse_args(argv)
    if a.arms:
        ARM.clear(); ARM.update(dict(kv.split("=") for kv in a.arms.split(",")))
    os.makedirs(a.out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = {}
    for p in sorted(glob.glob(os.path.join(a.final_dir, "ScoringAB_*.mat"))):
        d = load(p)
        if len(d["dF"]) == 0:
            continue
        D[(d["mouse"], d["phase"])] = d
    mice = sorted({m for m, _ in D}, key=lambda m: (ARM.get(m, ""), m))
    PD, HT = [], []
    for (m, ph), d in D.items():
        x = per_delivery(d); x.insert(0, "phase", ph); x.insert(0, "mouse", m); x.insert(1, "arm", ARM.get(m, "")); PD.append(x)
        h = held_table(d); h.insert(0, "phase", ph); h.insert(0, "mouse", m); h.insert(1, "arm", ARM.get(m, "")); HT.append(h)
    PD, HT = pd.concat(PD), pd.concat(HT)
    PD.to_csv(os.path.join(a.out_dir, f"behav_{a.day}_per_delivery.csv"), index=False)
    agg = {"n_deliveries": ("delivery", "size"), "withdrawal_pct": ("withdrawal", lambda x: 100 * x.mean()),
           "flinch_pct": ("flinch", lambda x: 100 * x.mean()), "wd_latency_median_s": ("wd_latency_s", "median")}
    for nm, _, _ in HELD:
        agg[f"{nm}_after_pct"] = (f"{nm} starts 0-3 s", lambda x: 100 * x.mean())
    T = PD.groupby(["mouse", "arm", "phase", "stim"]).agg(**agg).round(2).reset_index()
    # Fisher exact baseline vs injection, per mouse and stimulus
    st = []
    for m in mice:
        for s in STIM:
            b = PD[(PD.mouse == m) & (PD.phase == "baseline") & (PD.stim == s)]
            j = PD[(PD.mouse == m) & (PD.phase == "injection") & (PD.stim == s)]
            r = dict(mouse=m, arm=ARM.get(m, ""), stim=s, n_base=len(b), n_inj=len(j))
            for c in ("withdrawal", "flinch"):
                if len(b) and len(j):
                    tab = [[b[c].sum(), len(b) - b[c].sum()], [j[c].sum(), len(j) - j[c].sum()]]
                    r[f"{c}_base_pct"], r[f"{c}_inj_pct"] = 100 * b[c].mean(), 100 * j[c].mean()
                    r[f"{c}_change_pts"] = r[f"{c}_inj_pct"] - r[f"{c}_base_pct"]
                    r[f"{c}_fisher_p"] = fisher_exact(tab)[1]
            st.append(r)
    ST = pd.DataFrame(st).round(4)
    T.to_csv(os.path.join(a.out_dir, f"behav_{a.day}_per_session_stimulus.csv"), index=False)
    HT.round(3).to_csv(os.path.join(a.out_dir, f"behav_{a.day}_held.csv"), index=False)
    ST.to_csv(os.path.join(a.out_dir, f"behav_{a.day}_change.csv"), index=False)

    def pair_panel(ax, getv, title, ylab, ptxt=None):
        labels = []
        for m in mice:
            b, j = getv(m, "baseline"), getv(m, "injection")
            if b is None:
                continue
            arm = ARM.get(m, "")
            lab = f"{m} ({arm})" + ("" if j is not None else ", injection not scored yet")
            ax.plot([0, 1] if j is not None else [0], [b, j] if j is not None else [b], "o-", color=ARMC.get(arm, "k"),
                    label=lab, lw=2, ms=6)
            if ptxt and j is not None:
                p = ptxt(m)
                if p is not None and np.isfinite(p):
                    labels.append([j, f"{m}: p={p:.2g}", ARMC.get(arm, "k")])
        # spread the p labels so they never overlap (>= 7 % of the axis apart)
        labels.sort(key=lambda x: x[0])
        lo, hi = ax.get_ylim() if ax.get_ylim()[1] > 1 else (0, 100)
        gap = 0.07 * (hi - lo)
        for i in range(1, len(labels)):
            labels[i][0] = max(labels[i][0], labels[i - 1][0] + gap)
        for y, t, c in labels:
            ax.text(1.06, y, t, fontsize=6.5, va="center", color=c)
        ax.set_xticks([0, 1], ["baseline", "injection"]); ax.set_xlim(-0.3, 1.8)
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylab, fontsize=8)

    # ---- figure 1: reflexes per stimulus --------------------------------------------------------
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.5))
    for r, c in enumerate(("withdrawal", "flinch")):
        for k, s in enumerate(STIM):
            def gv(m, ph, c=c, s=s):
                x = T[(T.mouse == m) & (T.phase == ph) & (T.stim == s)]
                return float(x[f"{c}_pct"].iloc[0]) if len(x) else None

            def pv(m, c=c, s=s):
                x = ST[(ST.mouse == m) & (ST.stim == s)]
                return float(x[f"{c}_fisher_p"].iloc[0]) if len(x) and f"{c}_fisher_p" in x and np.isfinite(x[f"{c}_fisher_p"].iloc[0]) else None
            nb = T[(T.stim == s) & (T.phase == "baseline")].n_deliveries.sum()
            nj = T[(T.stim == s) & (T.phase == "injection")].n_deliveries.sum()
            pair_panel(ax[r, k], gv, f"{s}: {c} (deliveries: {nb} baseline, {nj} injection)", f"% deliveries with a {c}", pv)
            ax[r, k].set_ylim(-5, 105)
    ax[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(f"Day {a.day[-1]}: reflexes per stimulus, baseline vs injection (each line = one mouse; red SBI-553, blue vehicle)", fontsize=12)
    fig.text(0.5, 0.005, "A withdrawal / flinch = a tap from the delivery to 2 s later (or the next delivery). p = Fisher exact test on the "
             "deliveries of that mouse, baseline vs injection. n = 2 mice per arm: arms are compared by eye, not tested.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(os.path.join(a.out_dir, f"behav_{a.day}_reflex.png"), dpi=110); plt.close(fig)

    # ---- figure 2: held / affective behaviours ----------------------------------------------------
    names = [nm for nm, _, _ in HELD]
    fig, ax = plt.subplots(2, len(names), figsize=(20, 8))
    for k, nm in enumerate(names):
        for r, blk in enumerate(("whole session", "stimulus blocks")):
            def gv(m, ph, nm=nm, blk=blk):
                x = HT[(HT.mouse == m) & (HT.phase == ph) & (HT.behaviour == nm)]
                if blk == "whole session":
                    x = x[x.block == "whole session"]
                    return float(x.pct_time.iloc[0]) if len(x) else None
                x = x[x.block.str.startswith("stim")]
                return float(x.pct_time.mean()) if len(x) else None
            pair_panel(ax[r, k], gv, f"{nm}: {blk}", "% of time")
    ax[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Day {a.day[-1]}: affective / spontaneous behaviours, % of time, baseline vs injection "
                 "(attending, lick/bite, guarding, escape: your scoring; movement, grooming: automatic)", fontsize=11)
    fig.text(0.5, 0.005, "Top: whole 16.5-min session. Bottom: mean of the 4 stimulus blocks. Each line = one mouse; red SBI-553, blue vehicle.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(a.out_dir, f"behav_{a.day}_affective.png"), dpi=110); plt.close(fig)

    # ---- figure 3: event-locked behaviour probability -------------------------------------------
    beh = ["withdrawal", "flinch", "general movement", "escape/rearing", "grooming", "attending"]
    fig, ax = plt.subplots(2, len(beh), figsize=(21, 7.5), sharex=True)
    for r, arm in enumerate(("SBI-553", "vehicle")):
        for k, nm in enumerate(beh):
            for ph, ls in (("baseline", "-"), ("injection", "--")):
                Ms = []
                for m in mice:
                    if ARM.get(m) != arm or (m, ph) not in D:
                        continue
                    t, E = event_locked(D[(m, ph)], [nm])
                    Ms.append(E[nm].mean(0))
                if Ms:
                    y = np.mean(Ms, 0)
                    ax[r, k].plot(t, 100 * y, ls, color=ARMC[arm], lw=2 if ph == "baseline" else 1.5, alpha=1 if ph == "baseline" else 0.75,
                                  label=f"{ph} (mean of {len(Ms)} mice)")
            ax[r, k].axvline(0, color="k", lw=0.8, ls=":")
            ax[r, k].set_title(f"{arm}: {nm}", fontsize=9)
            if k == 0:
                ax[r, k].set_ylabel("% of deliveries", fontsize=8)
            if r == 1:
                ax[r, k].set_xlabel("time from delivery (s)")
    ax[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Day {a.day[-1]}: behaviour around every delivery (all stimuli), baseline (solid) vs injection (dashed)", fontsize=12)
    fig.text(0.5, 0.005, "Taps (withdrawal, flinch): % of deliveries with a tap in each 0.1-s bin. Held behaviours: % of deliveries in which the "
             "behaviour is on at that time. Mean of the mice of each arm.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(a.out_dir, f"behav_{a.day}_eventlocked.png"), dpi=110); plt.close(fig)
    print("sessions:", sorted(D)); print(ST.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
