"""make_synthetic_day2.py  -  a FAKE Day 2, for planning and for validation.

WHAT THIS IS FOR
    Two jobs, both legitimate, neither of them producing a result:

    1. PLANNING.  Build the lab-meeting deck before Day 2 exists, so the slide
       layout, the axis labels and the statistics are all settled in advance
       and Day 2 only needs the real folder swapped in.

    2. VALIDATION.  Inject a known effect and check the analysis recovers it.
       This is how the events-per-delivery measure was shown to be far more
       sensitive than a yes/no response: with a true 40 % reduction, the rate
       recovered a ratio of 0.64 while the binary readout only reached 0.82,
       because a binary hit saturates once a window holds two or three bouts.

WHAT THIS IS NOT
    Not data. Every downstream script that touches this folder must be run
    with --mockup, which stamps MOCKUP across every panel and appends _MOCKUP
    to every filename. Do not remove that. The real Day 2 scoring replaces
    this folder wholesale; nothing here is ever merged with real output.

TRUTH INJECTED  (change with the flags, printed on every run)
    affective  attending / lickbite / guarding   -40 % of bouts
    escape / rearing                             unchanged - it is
                                                 exploration, not pain, so a
                                                 pain drug should NOT move it
    reflexive  withdrawal / flinch               -15 % of events
    mouse F3                                     unchanged in everything, a
                                                 deliberate non-responder
    delivery count                               jittered per mouse, so the
                                                 two days differ in how many
                                                 stimuli were given

    That last one matters. The whole reason the analysis normalises is that
    the experimenter delivers a different number of taps each session - ten
    on one day, twenty on another. If the synthetic Day 2 copied Day 1's
    delivery counts exactly, the mockup would hide the very confound the
    normalisation exists to remove.

USAGE
    python make_synthetic_day2.py <day1_folder> <output_folder>
    python make_synthetic_day2.py <day1> <out> --aff-drop 0.4 --ref-drop 0.15

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from scipy.io import loadmat, savemat

AFF_CODES = (1, 2, 3)      # attending, lickbite, guarding
ESCAPE = 4
REF_CODES = (1, 2)         # withdrawal, flinch


def s_(v, d=""):
    a = np.asarray(v).ravel()
    if a.size == 0:
        return d
    x = a[0]
    if isinstance(x, np.ndarray):
        x = x.ravel()[0] if x.size else d
    return str(x).strip()


def bouts(sc, code):
    m = (sc == code).astype(int)
    e = np.diff(np.r_[0, m, 0])
    return list(zip(np.where(e == 1)[0], np.where(e == -1)[0]))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="Day 1 corrected folder (read only)")
    ap.add_argument("dst", help="where to write the synthetic Day 2")
    ap.add_argument("--aff-drop", type=float, default=.40)
    ap.add_argument("--ref-drop", type=float, default=.15)
    ap.add_argument("--deliv-jitter", type=float, default=.25,
                    help="fractional change in delivery count per mouse, "
                         "drawn uniformly in +/- this range")
    ap.add_argument("--nonresponder", default="F3",
                    help="mouse left completely unchanged")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    if os.path.abspath(a.src) == os.path.abspath(a.dst):
        raise SystemExit("refusing to write the synthetic day over the real "
                         "Day 1 folder")
    os.makedirs(a.dst, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    drop = {c: a.aff_drop for c in AFF_CODES}
    drop[ESCAPE] = 0.0

    print(f"reading  {a.src}")
    print(f"writing  {a.dst}   (SYNTHETIC - not data)")
    print(f"truth: affective -{a.aff_drop:.0%}, escape unchanged, "
          f"reflex -{a.ref_drop:.0%},\n       {a.nonresponder} unchanged, "
          f"delivery count jittered +/-{a.deliv_jitter:.0%}\n")

    for p in sorted(glob.glob(os.path.join(a.src, "ScoringAB_*.mat"))):
        M = loadmat(p)
        mouse = s_(M.get("mouseID", ""))
        resp = not mouse.upper().startswith(a.nonresponder.upper())
        scale = 1.0 if resp else 0.0

        # ---- thin the behaviour ----
        sc = np.asarray(M["score"]).ravel().astype(int).copy()
        for code, pr in drop.items():
            for lo, hi in bouts(sc, code):
                if rng.random() < pr * scale:
                    sc[lo:hi] = 0
        M["score"] = sc.reshape(np.asarray(M["score"]).shape)

        rx = np.asarray(M.get("reflexEvents", np.empty((0, 2)))).reshape(-1, 2)
        if rx.size:
            keep = np.array([
                not (int(c) in REF_CODES
                     and rng.random() < a.ref_drop * scale)
                for _, c in rx])
            M["reflexEvents"] = rx[keep]

        # ---- jitter the delivery count, per stimulus ----
        # Dropping a delivery is honest here: it mimics the experimenter
        # giving fewer taps. It is NOT allowed to invent taps at times the
        # scorer never marked, so the count can only go down, and the jitter
        # is applied by thinning within each stimulus block.
        dF = np.asarray(M["dFrames"]).ravel().astype(int)
        dT = np.asarray(M["dTypes"]).ravel().astype(int)
        keep = np.ones(len(dF), bool)
        for ty in np.unique(dT):
            idx = np.flatnonzero(dT == ty)
            frac = rng.uniform(0, a.deliv_jitter)
            ndrop = int(round(len(idx) * frac))
            if ndrop:
                keep[rng.choice(idx, ndrop, replace=False)] = False
        M["dFrames"] = dF[keep].reshape(1, -1)
        M["dTypes"] = dT[keep].reshape(1, -1)

        for k in list(M):
            if k.startswith("__"):
                M.pop(k)
        savemat(os.path.join(a.dst, os.path.basename(p)), M)
        print(f"  {mouse:4s} deliveries {len(dF):3d} -> {int(keep.sum()):3d}"
              f"{'' if resp else '    (non-responder by design)'}")

    print("\nRun every downstream script on this folder with --mockup.")


if __name__ == "__main__":
    main()
