"""behaviour_flow.py  -  sedation scales everything. What does it NOT change?

THE PROBLEM WITH EVERY NUMBER WE HAVE SO FAR
    Day 2 counts fell for all six behaviours, and baseline locomotion was
    already halved before any stimulus. A uniform reduction in motor output
    produces exactly that, so no MARGINAL count can separate:

        analgesia   the animal feels less, so the pain-specific part of the
                    response is missing
        sedation    the animal does less of everything, pain-specific or not

    Slide 10 of the deck already tried the affective-versus-reflexive plot
    and found both fell together. That compares two amounts, and two amounts
    both scale with activity.

WHAT IS INVARIANT TO A UNIFORM SCALING
    A conditional probability measured among the trials where the animal DID
    respond. If SBI-553 only turns the gain down, then GIVEN that a
    withdrawal happened, the rest of that response should look normal: the
    affective follow-on should still arrive, at the same latency, for the
    same duration. If the drug removes the affective component
    specifically, the follow-on should be missing even on trials where the
    reflex fired.

    This is the GrimACE argument applied to our own data - the pain axis and
    the movement axis are separate, and a transition structure says
    something a count cannot. It needs no new video processing: the
    per-frame score array is already in ScoringAB_*.mat.

WHAT IS COMPUTED, per session
    p_wd_given_del      withdrawal within RESP_S of a delivery / deliveries
    p_aff_given_wd      any affective state starting within FOLLOW_S of a
                        withdrawal / withdrawals        <- the key measure
    p_aff_given_del     the same not conditioned on the reflex, for
                        comparison: this one DOES scale with activity
    lat_aff_s           withdrawal -> affective onset, median, given both
    dur_aff_s           affective bout duration, median, given it occurred
    trans               5 x 5 state transition matrix, row-normalised

    States are the scorer's own: 0 none, 1 paw attending, 2 licking/biting,
    3 sustained lifting/guarding, 4 escape/rearing. AFFECTIVE = {1,2,3}.
    Escape is kept out of it - the deck reads escape as exploration.

WHAT IT FOUND - THE TRANSITION GRAMMAR IS UNCHANGED, AND THAT IS THE ANSWER
    The row-normalised transition matrix is independent of how much the
    animal did, and it barely moves (fig_transition_matrix.png):

        none      -> none      0.94 -> 0.98   (+0.05)
        attending -> attending 0.29 -> 0.35   (+0.06)
        lick/bite -> lick/bite 0.29 -> 0.29   (+0.01)
        guarding  -> guarding  0.26 -> 0.26   (+0.00)
        escape    -> escape    0.39 -> 0.40   (+0.01)
        attending -> none      0.61 -> 0.61   (+0.00)
        guarding  -> none      0.65 -> 0.65   (-0.00)
        escape    -> none      0.59 -> 0.60   (+0.01)

    An analgesic should make the pain-driven states END sooner and be
    ENTERED less often from inside a response. Neither happens: persistence
    and exit probabilities are within 0.01-0.06 of vehicle. The animal's
    behavioural grammar survives the drug. There is simply less of it.

    THE ONE REAL CHANGE  lick/bite -> guarding  0.11 -> 0.24  (+0.12)
    Licking is an active movement, guarding is a held posture. On the drug
    day an animal in the active affective state is more than twice as likely
    to fall into the passive one. That is the direction motor slowing
    predicts, and it is supported by two other numbers pointing the same
    way: affective onset is LATER (1.70 -> 2.20 s) and affective bouts are
    LONGER (0.50 -> 0.83 s) once they start. A drug that simply removed the
    pain would not lengthen the bouts it failed to prevent.

    THE CONDITIONAL PROBABILITY IS AMBIGUOUS, AND HONESTLY SO
        P(withdrawal | delivery)   0.81 -> 0.70   p = 0.562
        P(affective | delivery)    0.55 -> 0.32   p = 0.062
        P(affective | withdrawal)  0.56 -> 0.37   p = 0.094   (5 of 6 down)
    The key measure falls to 0.67 of vehicle, which leans
    affective-selective, but does not reach significance and points the
    opposite way from the grammar result. Taken together the drug looks
    predominantly sedating with a possible additional affective component
    that this n cannot resolve.

    READ P(withdrawal | delivery) WITH CARE. It is a probability, capped at
    1, and on Day 1 five of six mice sit at 0.75-0.88 - close to the
    ceiling, with little room to fall. The deck's events-per-delivery
    measure is the more sensitive one for the reflex.

    ALSO Day 2 had 27 % fewer deliveries than Day 1 (388 against 534), and
    Day 2 affective bout counts per mouse are 7-47, so those medians are
    noisy.

WHAT THIS STILL CANNOT DO
    Day 1 was vehicle for every mouse and Day 2 was drug for every mouse, so
    day, order and treatment are one variable. Nothing here separates them;
    that needs a counterbalanced cohort.

OUTPUT  ->  qc/flow/
    Flow_session.csv, Flow_report.txt, fig_behaviour_flow.png,
    fig_transition_matrix.png

USAGE
    python behaviour_flow.py

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import glob
import os

import matplotlib
import numpy as np
import pandas as pd
import scipy.io as sio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
SCO = {1: os.path.join(ROOT, "videos", "output_corrected"),
       2: os.path.join(ROOT, "videos", "output_day2_corrected")}
OUT = os.path.join(ROOT, "qc", "flow")
BLOCKS = [(300, 600), (660, 960), (1020, 1320), (1380, 1680)]
RESP_S = 2.0            # a delivery counts as answered within this
FOLLOW_S = 5.0          # an affective follow-on must start within this
AFF = (1, 2, 3)
NAMES = ["none", "attending", "lick/bite", "guarding", "escape/rear"]
D2C, INK = "#C1272D", "#1A1A1A"
plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.2,
                     "axes.titlesize": 12, "axes.labelsize": 11.5})


def one(mat, day):
    s = sio.loadmat(mat, squeeze_me=True, struct_as_record=False)
    fps = float(s["frameRate"])
    sc = np.asarray(s["score"]).astype(int)
    ev = np.asarray(s["reflexEvents"], float)
    dfr = np.asarray(s["dFrames"], float)
    n = len(sc)
    t = np.arange(n) / fps

    # inside the stimulus blocks only: the baseline has no deliveries, and
    # including it would mix a no-stimulus period into a stimulus measure
    inb = np.zeros(n, bool)
    for a, b in BLOCKS:
        inb |= (t >= a) & (t <= b)
    keep = np.array([inb[int(min(f, n - 1))] for f in dfr], bool) \
        if len(dfr) else np.zeros(0, bool)
    dfr = dfr[keep]
    wd = ev[ev[:, 1] == 1][:, 0] if len(ev) else np.zeros(0)
    if len(wd):
        wd = wd[np.array([inb[int(min(f, n - 1))] for f in wd], bool)]

    aff = np.isin(sc, AFF)
    is_wd = np.zeros(n, bool)
    is_wd[wd.astype(int)] = True
    rs, fs = int(RESP_S * fps), int(FOLLOW_S * fps)

    def any_in(mask, f0, span):
        a, b = int(f0), int(min(f0 + span, n))
        return bool(mask[a:b].any()) if b > a else False

    n_del, n_wd = len(dfr), len(wd)
    wd_after_del = sum(any_in(is_wd, f, rs) for f in dfr)
    aff_after_wd = sum(any_in(aff, f, fs) for f in wd)
    aff_after_del = sum(any_in(aff, f, fs) for f in dfr)

    # latency withdrawal -> affective onset, and that bout's own duration
    on = np.nonzero(np.diff(np.concatenate([[0], aff.astype(np.int8)]))
                    == 1)[0]
    off = np.nonzero(np.diff(np.concatenate([aff.astype(np.int8), [0]]))
                     == -1)[0]
    lat, dur = [], []
    for f in wd:
        c = on[(on >= f) & (on <= f + fs)]
        if len(c):
            lat.append((c[0] - f) / fps)
            e = off[off >= c[0]]
            if len(e):
                dur.append((e[0] - c[0] + 1) / fps)

    # transitions at 1 Hz, so a transition means a change of state rather
    # than a frame of jitter
    ss = sc[inb][::int(round(fps))]
    T = np.zeros((5, 5))
    for a, b in zip(ss[:-1], ss[1:]):
        T[a, b] += 1
    return dict(mouse=str(s["mouseID"]), day=day,
                treatment=str(s["treatment"]), n_del=n_del, n_wd=n_wd,
                p_wd_given_del=wd_after_del / n_del if n_del else np.nan,
                p_aff_given_wd=aff_after_wd / n_wd if n_wd else np.nan,
                p_aff_given_del=aff_after_del / n_del if n_del else np.nan,
                lat_aff_s=float(np.median(lat)) if lat else np.nan,
                dur_aff_s=float(np.median(dur)) if dur else np.nan,
                n_lat=len(lat)), T


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    TT = {1: np.zeros((5, 5)), 2: np.zeros((5, 5))}
    for day in (1, 2):
        for m in sorted(glob.glob(os.path.join(SCO[day],
                                               "ScoringAB_*.mat"))):
            r, T = one(m, day)
            rows.append(r)
            TT[day] += T
            print(f"  day{day} {r['mouse']}: {r['n_del']} deliveries, "
                  f"{r['n_wd']} withdrawals, "
                  f"p(aff|wd) = {r['p_aff_given_wd']:.2f}")
    D = pd.DataFrame(rows).sort_values(["mouse", "day"])
    D.to_csv(os.path.join(OUT, "Flow_session.csv"), index=False)
    print()
    print(D.to_string(index=False))
    report(D, TT)
    return D


def report(D, TT):
    L = ["", "===== behaviour flow, Day 2 (SBI-553) vs Day 1 (vehicle) "
             "=====",
         "paired exact Wilcoxon across the 6 mice; the floor at n = 6 is "
         "p = 0.031", ""]
    for col, what, note in (
            ("p_wd_given_del", "P(withdrawal | delivery)",
             "scales with activity"),
            ("p_aff_given_del", "P(affective | delivery)",
             "scales with activity"),
            ("p_aff_given_wd", "P(affective | withdrawal)",
             "INVARIANT to a uniform scaling  <- key measure"),
            ("lat_aff_s", "latency withdrawal -> affective (s)",
             "given both occurred"),
            ("dur_aff_s", "affective bout duration (s)",
             "given it occurred")):
        w = D.pivot(index="mouse", columns="day", values=col).dropna()
        if len(w) < 3:
            L.append(f"  {what:<38s} only {len(w)} mice have both days")
            continue
        try:
            _, p = wilcoxon(w[1], w[2])
        except ValueError:
            p = float("nan")
        L.append(f"  {what:<38s} {w[1].median():6.3f} -> "
                 f"{w[2].median():6.3f}   ratio "
                 f"{w[2].median() / max(w[1].median(), 1e-9):5.2f}   "
                 f"p = {p:.3f}   {note}")
    L += ["", "HOW TO READ IT",
          "  The two 'scales with activity' rows falling tells you only "
          "that the animal did less.",
          "  P(affective | withdrawal) conditions on the animal having "
          "responded at all:",
          "    UNCHANGED  -> the response structure survives. The drop is "
          "consistent with a gain",
          "                  reduction - sedation - and not with a missing "
          "affective component.",
          "    FALLS      -> the affective part is absent even on trials "
          "where the reflex fired,",
          "                  which is what an analgesic should do.",
          "",
          "  CAVEAT Day 1 was vehicle for every mouse and Day 2 drug for "
          "every mouse, so day,",
          "  order and treatment are one variable here."]
    txt = "\n".join(L)
    with open(os.path.join(OUT, "Flow_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    figure(D)
    fig_trans(TT)


def figure(D):
    ks = [("p_wd_given_del", "P(withdrawal | delivery)"),
          ("p_aff_given_del", "P(affective | delivery)"),
          ("p_aff_given_wd", "P(affective | withdrawal)"),
          ("lat_aff_s", "latency wd -> affective (s)"),
          ("dur_aff_s", "affective bout duration (s)")]
    fig, ax = plt.subplots(1, len(ks), figsize=(3.2 * len(ks), 4.6))
    for A, (col, ttl) in zip(np.atleast_1d(ax), ks):
        w = D.pivot(index="mouse", columns="day", values=col).dropna()
        for mo in w.index:
            A.plot([0, 1], [w.loc[mo, 1], w.loc[mo, 2]], color="#BBBBBB",
                   lw=1.3, marker="o", ms=5.5, mfc="white", mec=INK,
                   zorder=2)
        if len(w):
            A.plot([0, 1], [w[1].median(), w[2].median()], color=D2C, lw=3,
                   zorder=3)
        try:
            _, p = wilcoxon(w[1], w[2])
        except ValueError:
            p = float("nan")
        key = col == "p_aff_given_wd"
        A.set_xticks([0, 1])
        A.set_xticklabels(["Day 1\nvehicle", "Day 2\nSBI-553"])
        A.set_xlim(-.35, 1.35)
        A.set_title(f"{'>> ' if key else ''}{ttl}\n"
                    f"{w[1].median():.2f} -> {w[2].median():.2f},  "
                    f"p = {p:.3f}   (n = {len(w)})",
                    fontweight="bold" if key else "normal",
                    color=D2C if key else INK)
        A.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Sedation scales every count down. A probability measured AMONG "
        "THE TRIALS THAT GOT A RESPONSE does not scale.\n"
        "the first two panels fall with overall activity; the third "
        "conditions on the reflex having fired, so it asks whether the "
        "affective part is specifically missing.\n"
        "one line per mouse, red = median", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, .83])
    p_ = os.path.join(OUT, "fig_behaviour_flow.png")
    fig.savefig(p_, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p_}")


def fig_trans(TT):
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))
    N = {}
    for A, day, ttl in ((ax[0], 1, "Day 1  vehicle"),
                        (ax[1], 2, "Day 2  SBI-553")):
        T = TT[day]
        N[day] = T / np.maximum(T.sum(1, keepdims=True), 1e-9)
        im = A.imshow(N[day], cmap="magma", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                A.text(j, i, f"{N[day][i, j]:.2f}", ha="center",
                       va="center", fontsize=9,
                       color="white" if N[day][i, j] < .55 else "black")
        A.set_title(f"{ttl}\nP(next state | this state), 1 Hz")
        fig.colorbar(im, ax=A, fraction=.046)
    dif = N[2] - N[1]
    im = ax[2].imshow(dif, cmap="coolwarm", vmin=-.3, vmax=.3)
    for i in range(5):
        for j in range(5):
            ax[2].text(j, i, f"{dif[i, j]:+.2f}", ha="center", va="center",
                       fontsize=9, color=INK)
    ax[2].set_title("Day 2 minus Day 1\nred = more likely on the drug")
    fig.colorbar(im, ax=ax[2], fraction=.046)
    for A in ax:
        A.set_xticks(range(5))
        A.set_xticklabels(NAMES, rotation=40, ha="right", fontsize=9)
        A.set_yticks(range(5))
        A.set_yticklabels(NAMES, fontsize=9)
        A.set_xlabel("next state")
        A.set_ylabel("this state")
    fig.suptitle("Behaviour transition structure, pooled over the 6 mice. "
                 "Each ROW sums to 1, so this is independent of how much "
                 "the animal did overall.", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .91])
    p_ = os.path.join(OUT, "fig_transition_matrix.png")
    fig.savefig(p_, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p_}")


if __name__ == "__main__":
    main()
