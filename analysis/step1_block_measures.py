"""step1_block_measures.py  -  turn scored sessions into per-block measures.

THE DESIGN THIS ASSUMES
    5 min baseline (no stimulus)
    1 min rest
    5 min block of one stimulus, variable ITI within the block
    1 min rest
    ... four stimulus blocks, order randomised per session

PERIOD DEFINITION
    baseline    frames 1 .. BASELINE_S seconds
    each block  the block's FIRST delivery .. + BLOCK_S seconds,
                clipped so it never reaches the next block's first delivery

TWO BLOCK LENGTHS, BOTH REPORTED
    --block-s 300   the 5 min stimulus period only
    --block-s 360   the 5 min stimulus period PLUS the 1 min rest that
                    follows it

    The 6 min version exists because behaviour does not stop when the
    stimulus does. Attending, licking/biting and guarding continue into the
    rest minute, and a 300 s window cuts that tail off. Attributing the rest
    to the block that preceded it is the correct reading: nothing else caused
    it.

    Note what happens to the two denominators:
        n_per_delivery   denominator UNCHANGED - the rest minute contains no
                         deliveries, so the extra behaviour is counted
                         without diluting the divisor. This is the measure to
                         compare between the two versions.
        rate_per_min     denominator changes 5 -> 6 min, so the rate falls
                         mechanically even if nothing about the animal
                         changed. Do not compare rate_per_min across the two
                         versions.

    Run it twice into two folders and report both.

    Fixed windows, so every block has the same denominator and the numbers are
    directly comparable. Measured on female1 the real first-to-last delivery
    spans were 247-299 s against a nominal 300 s, so a fixed window includes a
    little stimulus-free time at the end of the shorter blocks. That is the
    price of a common denominator and it is the right trade.

WHY NOT PER-DELIVERY WINDOWS
    86 % of ITIs in female1 were under 30 s (median 4.7 s, min 0.9 s), so a
    per-delivery observation window is truncated almost every time and its
    length varies wildly. The 5 min block is the analysable unit here.
    Normalized_<vid>.csv from the scorer is kept as a secondary view only.

TWO DENOMINATORS, because both vary between blocks
    per minute     count / minutes of the window
                   -> the only thing comparable to baseline, which has no
                      deliveries at all
    per delivery   count / number of that stimulus delivered
                   -> the fair way to compare Heat against Pin prick, since
                      delivery counts differed by 46 % in female1 (16 vs 26)

USAGE
    python step1_block_measures.py <folder with ScoringAB_*.mat> [--out DIR]

OUTPUT
    BlockMeasures_long.csv   one row per session x period x behaviour
    BlockMeasures_wide.csv   one row per session x period
    Step1_QC.csv             per-session sanity checks - read this first

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

BASELINE_S = 300.0        # within-session no-stimulus reference
BLOCK_S = 300.0           # fixed 5 min from the block's first delivery

# affective codes written by score_AB_dual_view.m
AFF = {1: "attending", 2: "lickbite", 3: "guarding", 4: "escape"}
REF = {1: "withdrawal", 2: "flinch"}
MIN_DUR = {3: 2.0}        # guarding > 2 s; overridden by guardMin in the file


# ── .mat helpers: loadmat wraps everything in arrays ────────────────────────
def s_(v, default=""):
    """A scalar string out of whatever loadmat produced."""
    a = np.asarray(v).ravel()
    if a.size == 0:
        return default
    x = a[0]
    if isinstance(x, np.ndarray):
        x = x.ravel()[0] if x.size else default
    return str(x).strip()


def f_(v, default=np.nan):
    a = np.asarray(v, dtype=float).ravel()
    return float(a[0]) if a.size else default


def vec(v):
    return np.asarray(v).ravel()


def cellstr(v, n=4):
    a = np.asarray(v).ravel()
    out = []
    for i in range(min(n, a.size)):
        x = a[i]
        out.append(str(x.ravel()[0] if isinstance(x, np.ndarray) and x.size
                       else x).strip())
    while len(out) < n:
        out.append(f"Stim {len(out) + 1}")
    return out


def bouts(binary, fps, min_s=0.0):
    """(n_bouts, total_seconds) with bouts shorter than min_s discarded."""
    b = np.asarray(binary, bool).ravel().astype(np.int8)
    d = np.diff(np.concatenate(([0], b, [0])))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (en - st) >= max(int(round(min_s * fps)), 1)
    st, en = st[keep], en[keep]
    return int(len(st)), float((en - st).sum() / fps)


# ── one session ─────────────────────────────────────────────────────────────
def read_session(path, baseline_s, block_s):
    M = loadmat(path, squeeze_me=False)
    fps = f_(M["frameRate"], 30.0)
    score = vec(M["score"]).astype(int)
    nUsed = int(f_(M.get("nUsed", len(score)), len(score)))
    score = score[:nUsed]
    unc = vec(M["uncert"]).astype(bool)[:nUsed] if "uncert" in M \
        else np.zeros(nUsed, bool)
    dF = vec(M["dFrames"]).astype(int)
    dT = vec(M["dTypes"]).astype(int)
    names = cellstr(M["stimNames"])
    guard = f_(M.get("guardMin", 2.0), 2.0)

    rx = np.asarray(M["reflexEvents"]) if "reflexEvents" in M else np.empty((0, 2))
    rx = rx.reshape(-1, 2) if rx.size else np.empty((0, 2))

    meta = dict(
        file=os.path.basename(path),
        session=s_(M.get("sessionNo", "")),
        mouse=s_(M.get("mouseID", "")),
        sex=s_(M.get("sexID", "")),
        day=s_(M.get("dayNo", "")),
        phase=s_(M.get("phase", "")),
        fps=fps, nUsed=nUsed, guardMin=guard,
        session_s=nUsed / fps,
    )

    # ---- periods: baseline, then each stimulus block in temporal order ----
    periods = [dict(period="baseline", kind="baseline", pos=0,
                    stim_code=0, stimulus="none",
                    f0=1, f1=int(round(baseline_s * fps)), n_del=0)]
    seen, pos, starts = set(), 0, []
    for f, ty in zip(dF, dT):
        if ty in seen or ty < 1 or ty > 4:
            continue
        seen.add(ty)
        starts.append((int(f), int(ty)))
    for k, (f, ty) in enumerate(starts):
        pos += 1
        f1 = int(f + round(block_s * fps) - 1)
        # Never let a window run into the next block. At block_s = 300 this
        # never bites, but the 360 s version (5 min stimulus + 1 min rest)
        # reaches the next block's start whenever the first delivery was late,
        # and double-counting one block's behaviour into another would be
        # worse than losing the last second of rest.
        clipped = 0
        if k + 1 < len(starts):
            nxt = starts[k + 1][0]
            if f1 >= nxt:
                clipped = f1 - (nxt - 1)
                f1 = nxt - 1
        else:
            f1 = min(f1, nUsed)
        periods.append(dict(
            period=f"block{pos}_{names[ty - 1]}", kind="stimulus", pos=pos,
            stim_code=int(ty), stimulus=names[ty - 1],
            f0=int(f), f1=f1, clipped_frames=clipped,
            # the denominator stays the deliveries of THIS stimulus inside the
            # window; the 1 min rest contains none, so it adds behaviour to the
            # numerator without inflating the denominator - which is the point
            n_del=int(np.sum((dT == ty) & (dF >= f) & (dF <= f1)))))

    rows, qc = [], dict(meta)
    qc["n_periods"] = len(periods)
    qc["n_deliveries"] = int(len(dF))
    qc["n_unknown_type"] = int(np.sum(dT == 0))

    for p in periods:
        f0 = max(1, p["f0"])
        f1 = min(p["f1"], nUsed)
        if f1 < f0:
            continue
        sl = slice(f0 - 1, f1)
        seg = score[sl]
        mins = len(seg) / fps / 60.0
        trunc = p["f1"] > nUsed
        upct = 100.0 * float(unc[sl].mean()) if len(seg) else np.nan

        base = dict(meta)
        base.pop("file")
        base.update({k: p[k] for k in
                     ("period", "kind", "pos", "stim_code", "stimulus", "n_del")})
        base.update(dur_s=len(seg) / fps, dur_min=mins,
                    truncated=int(trunc), uncertain_pct=upct)

        for code, bn in AFF.items():
            n, dsec = bouts(seg == code, fps, MIN_DUR.get(code, 0.0)
                            if code != 3 else guard)
            rows.append({**base, "behaviour": bn, "class": "affective",
                         "n_bouts": n, "total_dur_s": dsec,
                         "rate_per_min": n / mins if mins else np.nan,
                         "dur_pct_time": 100 * dsec / (len(seg) / fps)
                         if len(seg) else np.nan,
                         "n_per_delivery": n / p["n_del"] if p["n_del"] else np.nan,
                         "dur_s_per_delivery": dsec / p["n_del"]
                         if p["n_del"] else np.nan})

        for code, bn in REF.items():
            if rx.size:
                k = int(np.sum((rx[:, 1] == code) & (rx[:, 0] >= f0) &
                               (rx[:, 0] <= f1)))
            else:
                k = 0
            rows.append({**base, "behaviour": bn, "class": "reflexive",
                         "n_bouts": k, "total_dur_s": np.nan,
                         "rate_per_min": k / mins if mins else np.nan,
                         "dur_pct_time": np.nan,
                         "n_per_delivery": k / p["n_del"] if p["n_del"] else np.nan,
                         "dur_s_per_delivery": np.nan})

    # QC fields
    st = [p for p in periods if p["kind"] == "stimulus"]
    qc["n_blocks"] = len(st)
    qc["block_order"] = " > ".join(p["stimulus"] for p in st)
    qc["n_del_per_block"] = " / ".join(str(p["n_del"]) for p in st)
    qc["any_truncated"] = int(any(p["f1"] > nUsed for p in periods))
    qc["baseline_covered_s"] = min(round(baseline_s * fps), nUsed) / fps
    # do the fixed windows collide with the next block?
    ov = 0
    for a, b in zip(st, st[1:]):
        if a["f1"] >= b["f0"]:
            ov += 1
    qc["block_window_overlaps"] = ov
    qc["uncertain_pct_session"] = 100.0 * float(unc.mean()) if nUsed else np.nan
    return rows, qc


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder containing ScoringAB_*.mat")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline-s", type=float, default=BASELINE_S)
    ap.add_argument("--block-s", type=float, default=BLOCK_S)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.folder, "ScoringAB_*.mat")))
    if not files:
        raise SystemExit(f"no ScoringAB_*.mat in {a.folder}\n"
                         "Score sessions with score_AB_dual_view.m first "
                         "(mode 2 or 3).")
    outdir = a.out or a.folder
    os.makedirs(outdir, exist_ok=True)

    print(f"baseline window {a.baseline_s:.0f} s, block window {a.block_s:.0f} s "
          f"from each block's first delivery\n")
    allrows, allqc = [], []
    for f in files:
        try:
            r, q = read_session(f, a.baseline_s, a.block_s)
        except Exception as e:                                  # noqa: BLE001
            print(f"  SKIPPED {os.path.basename(f)}: {type(e).__name__}: {e}")
            continue
        allrows += r
        allqc.append(q)
        print(f"  {q['mouse'] or '?':4s} day{q['day'] or '?'} "
              f"{q['phase'][:14]:14s} {q['n_blocks']} block(s)  "
              f"n/block {q['n_del_per_block']:16s}  {q['block_order']}")

    if not allrows:
        raise SystemExit("nothing readable")

    L = pd.DataFrame(allrows)
    Q = pd.DataFrame(allqc)
    lp = os.path.join(outdir, "BlockMeasures_long.csv")
    L.to_csv(lp, index=False)

    idx = ["session", "day", "mouse", "sex", "phase", "period", "kind", "pos",
           "stim_code", "stimulus", "n_del", "dur_min", "truncated",
           "uncertain_pct"]
    W = L.pivot_table(index=idx, columns="behaviour",
                      values=["rate_per_min", "dur_pct_time", "n_per_delivery"],
                      aggfunc="first")
    W.columns = [f"{b}_{m}" for m, b in W.columns]
    W = W.reset_index()
    wp = os.path.join(outdir, "BlockMeasures_wide.csv")
    W.to_csv(wp, index=False)
    qp = os.path.join(outdir, "Step1_QC.csv")
    Q.to_csv(qp, index=False)

    print(f"\n  {len(files)} file(s) -> {L['mouse'].nunique()} mouse/mice, "
          f"{len(Q)} session(s), {len(L)} long rows")
    print(f"  wrote {lp}")
    print(f"  wrote {wp}")
    print(f"  wrote {qp}")

    print("\n  ---- QC flags ----")
    bad = Q[(Q.n_blocks != 4) | (Q.any_truncated == 1) |
            (Q.block_window_overlaps > 0) | (Q.n_unknown_type > 0)]
    if bad.empty:
        print("  none: every session has 4 blocks, no truncation, no overlap, "
              "no untyped deliveries.")
    else:
        for _, r in bad.iterrows():
            msgs = []
            if r.n_blocks != 4:
                msgs.append(f"{r.n_blocks} blocks not 4")
            if r.any_truncated:
                msgs.append("a window runs past the end of the video")
            if r.block_window_overlaps:
                msgs.append(f"{r.block_window_overlaps} block window(s) overlap "
                            "the next block")
            if r.n_unknown_type:
                msgs.append(f"{r.n_unknown_type} untyped delivery/deliveries")
            print(f"  {r['file']}: " + "; ".join(msgs))
    print("\n  NEXT: python step2_block_stats.py " + f'"{outdir}"')


if __name__ == "__main__":
    main()
