"""repair_merge_fast_clicks.py  -  merge rapid repeated clicks into one event.

WHAT IT FIXES
    Two things produced the same artefact in the first sessions:
      1. a code bug - MATLAB fired KeyRelease during Windows key auto-repeat,
         so one held key was chopped into a train of 2-3 frame fragments
         (now fixed in score_AB_dual_view.m with a wall-clock debounce);
      2. clicking a key several times quickly instead of holding it.
    Both turn ONE behavioural episode into MANY bouts, which inflates counts.

    Measured in female1: lick/bite fragments repeated with a period of
    0.225 +/- 0.056 s (CV 0.25) in chains of up to 9. A tight period repeating
    nine times is not a human finger - it is auto-repeat.

WHAT IT DOES NOT DO
    It never touches your original scoring. Everything is written to a NEW
    folder. The originals in videos\\output stay exactly as you scored them.

WHAT IS AND IS NOT RECOVERABLE
    COUNTS      recoverable - merging collapses each fragment train back to
                one event.
    DURATIONS   NOT recoverable when the key was tapped rather than held. The
                merged span of a few taps is the span of your clicking, not of
                the mouse's behaviour. The corrected files therefore carry
                guardMin = 0 and you should report counts and rates only,
                never "% of time".

    The report lists the count at several merge thresholds. Where the count
    PLATEAUS the choice is safe; where it keeps falling there is no clean
    boundary between artefact and real gaps, and that behaviour's count should
    be treated as approximate. In female1/female2 only "attending" plateaued.

USAGE
    python repair_merge_fast_clicks.py "..\\videos\\output"
           [--out "..\\videos\\output_corrected"] [--gap 0.35]

OUTPUT (into the new folder)
    ScoringAB_<vid>.mat        corrected score vector - step1 reads this
    TrainingLabels_<vid>.csv   corrected frame-level labels
    Repair_report.md / .csv    what changed, and the threshold sensitivity
    DeliveryTimes_* etc.       copied through unchanged

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

AFF = {1: "attending", 2: "lickbite", 3: "guarding", 4: "escape"}
SENS = [0.0, 0.13, 0.20, 0.35, 0.50, 1.00]     # thresholds for the report


def runs(mask):
    b = np.asarray(mask, bool).astype(np.int8)
    d = np.diff(np.concatenate(([0], b, [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def merge_code(score, code, gap_frames):
    """Fill gaps <= gap_frames between bouts of `code`, but ONLY where the gap
    is empty. A gap containing a different behaviour is left alone, so we can
    never overwrite something you actually scored."""
    st, en = runs(score == code)
    filled = 0
    for i in range(len(st) - 1):
        g0, g1 = en[i], st[i + 1]              # [g0, g1) is the gap
        if (g1 - g0) > gap_frames:
            continue
        if np.any(score[g0:g1] != 0):          # someone else is in there
            continue
        score[g0:g1] = code
        filled += 1
    return filled


def count_at(score, code, gap_frames):
    s = score.copy()
    merge_code(s, code, gap_frames)
    st, _ = runs(s == code)
    return len(st)


def md_table(df):
    """Markdown table without pulling in tabulate."""
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for rec in df.to_dict("records"):
        out.append("| " + " | ".join(str(rec[c]) for c in cols) + " |")
    return "\n".join(out)


def s_(v, default=""):
    a = np.asarray(v).ravel()
    if a.size == 0:
        return default
    x = a[0]
    if isinstance(x, np.ndarray):
        x = x.ravel()[0] if x.size else default
    return str(x).strip()


def f_(v, d=np.nan):
    a = np.asarray(v, dtype=float).ravel()
    return float(a[0]) if a.size else d


def delivery_qc(dF, dT, names, fps, dup_s=0.5):
    """Flag DELIVERY marks that look like slips rather than real deliveries.

    Two patterns, both seen in female2:
      * two marks within dup_s of each other. One delivery marked twice.
      * of those, pairs with DIFFERENT stimulus types - a mis-key, e.g. 2
        pressed instead of 1. Six such pairs sat 0.0-0.2 s apart in female2
        and they turned a clean Heat block into 16 interleaved runs.

    These are NOT auto-corrected. Which of the two marks carries the right
    stimulus is ambiguous from the video alone, and you have the written
    record. They are reported so you can fix them deliberately.
    """
    order = np.argsort(dF)
    f, ty = dF[order], dT[order]
    out = []
    win = dup_s * fps
    for i in range(len(f) - 1):
        if (f[i + 1] - f[i]) > win:
            continue
        same = ty[i] == ty[i + 1]
        out.append(dict(
            t1=(f[i] - 1) / fps, t2=(f[i + 1] - 1) / fps,
            dt_s=(f[i + 1] - f[i]) / fps,
            stim1=names[ty[i] - 1] if 1 <= ty[i] <= 4 else "UNKNOWN",
            stim2=names[ty[i + 1] - 1] if 1 <= ty[i + 1] <= 4 else "UNKNOWN",
            kind="duplicate (same stimulus)" if same
                 else "MIS-KEY? (different stimulus)"))
    return pd.DataFrame(out)


def block_runs(dF, dT, names, fps):
    """How many runs of consecutive same-stimulus marks? 4 means a clean
    blocked session; many more means the blocks are interleaved or mis-keyed."""
    order = np.argsort(dF)
    f, ty = dF[order], dT[order]
    if not len(f):
        return pd.DataFrame()
    rows, s = [], 0
    for i in range(1, len(f) + 1):
        if i == len(f) or ty[i] != ty[s]:
            rows.append(dict(
                stimulus=names[ty[s] - 1] if 1 <= ty[s] <= 4 else "UNKNOWN",
                first_mark=s + 1, last_mark=i, n=i - s,
                t_start=(f[s] - 1) / fps, t_end=(f[i - 1] - 1) / fps,
                span_s=(f[i - 1] - f[s]) / fps))
            s = i
    return pd.DataFrame(rows)


def repair_one(path, outdir, gap_s):
    M = loadmat(path, squeeze_me=False)
    fps = f_(M["frameRate"], 30.0)
    gap_f = int(round(gap_s * fps))
    score = np.asarray(M["score"]).ravel().astype(int)
    orig = score.copy()
    vid = os.path.basename(path)[len("ScoringAB_"):-4]

    rows = []
    for code, name in AFF.items():
        st0, en0 = runs(orig == code)
        sens = {f"n_at_{g:g}s": count_at(orig, code, int(round(g * fps)))
                for g in SENS}
        nfilled = merge_code(score, code, gap_f)
        st1, en1 = runs(score == code)
        # does the count plateau over the top half of the thresholds?
        vals = [sens[f"n_at_{g:g}s"] for g in SENS[2:]]
        plateau = len(set(vals)) == 1
        rows.append(dict(
            file=vid, behaviour=name, code=code,
            raw_events=len(st0), merged_events=len(st1),
            gaps_filled=nfilled,
            raw_frames=int((en0 - st0).sum()), merged_frames=int((en1 - st1).sum()),
            plateau=int(plateau), counts_trustworthy=("yes" if plateau
                                                      else "approximate"),
            **sens))

    os.makedirs(outdir, exist_ok=True)

    # ---- corrected .mat, same field names so step1 reads it unchanged ----
    out = {k: v for k, v in M.items() if not k.startswith("__")}
    out["score"] = score.reshape(-1, 1).astype(float)
    out["guardMin"] = 0.0          # durations are not meaningful after tapping
    out["repaired"] = 1.0
    out["mergeGapS"] = float(gap_s)
    out["repairNote"] = ("fast repeated clicks merged into single events; "
                         "counts usable, durations are NOT")
    savemat(os.path.join(outdir, f"ScoringAB_{vid}.mat"), out)

    # ---- corrected TrainingLabels, if the original exists ----
    tl = os.path.join(os.path.dirname(path), f"TrainingLabels_{vid}.csv")
    if os.path.exists(tl):
        T = pd.read_csv(tl)
        n = min(len(T), len(score))
        lab = {0: "None"}
        lab.update({c: n_ for c, n_ in [(1, "Paw attending"),
                                        (2, "Licking or biting"),
                                        (3, "Sustained lifting / guarding"),
                                        (4, "Escape / rearing")]})
        T = T.iloc[:n].copy()
        T["affective_code"] = score[:n]
        T["affective_label"] = [lab.get(int(c), "?") for c in score[:n]]
        # redo the ML shift from the corrected trace
        lagf = int(round(0.250 * fps))
        ml = np.concatenate([score[lagf:n], np.zeros(min(lagf, n), int)])[:n]
        T["affective_code_ml"] = ml
        T["affective_label_ml"] = [lab.get(int(c), "?") for c in ml]
        T["repaired"] = 1
        T.to_csv(os.path.join(outdir, f"TrainingLabels_{vid}.csv"), index=False)

    # ---- delivery-mark QC and block structure ----
    dF = np.asarray(M.get("dFrames", np.empty(0))).ravel().astype(int)
    dT = np.asarray(M.get("dTypes", np.empty(0))).ravel().astype(int)
    names = []
    arr = np.asarray(M.get("stimNames", np.empty(0))).ravel()
    for i in range(4):
        if i < arr.size:
            x = arr[i]
            names.append(str(x.ravel()[0] if isinstance(x, np.ndarray) and x.size
                             else x).strip())
        else:
            names.append(f"Stim {i + 1}")
    dq = delivery_qc(dF, dT, names, fps)
    br = block_runs(dF, dT, names, fps)
    if not dq.empty:
        dq.insert(0, "file", vid)
    if not br.empty:
        br.insert(0, "file", vid)

    meta_missing = [k for k in ("sessionNo", "mouseID", "sexID", "dayNo")
                    if s_(M.get(k, "")) == ""]

    return (pd.DataFrame(rows), dq, br,
            dict(file=vid, missing_metadata=";".join(meta_missing),
                 n_runs=len(br)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="the ORIGINAL videos\\output folder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gap", type=float, default=0.35,
                    help="merge bouts separated by <= this many seconds "
                         "(default 0.35, which covers the 0.15-0.29 s "
                         "auto-repeat periods measured in female1/female2)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.folder, "ScoringAB_*.mat")))
    if not files:
        raise SystemExit(f"no ScoringAB_*.mat in {a.folder}")
    outdir = a.out or os.path.join(os.path.dirname(a.folder.rstrip("\\/")),
                                   "output_corrected")
    if os.path.abspath(outdir) == os.path.abspath(a.folder):
        raise SystemExit("refusing to write into the original folder")

    print(f"originals : {a.folder}   (never modified)")
    print(f"corrected : {outdir}")
    print(f"merge gap : {a.gap:.2f} s\n")

    got = [repair_one(f, outdir, a.gap) for f in files]
    R = pd.concat([g[0] for g in got], ignore_index=True)
    dqs = [g[1] for g in got if not g[1].empty]
    brs = [g[2] for g in got if not g[2].empty]
    DQ = pd.concat(dqs, ignore_index=True) if dqs else pd.DataFrame()
    BR = pd.concat(brs, ignore_index=True) if brs else pd.DataFrame()
    MD = pd.DataFrame([g[3] for g in got])

    # pass the delivery files through untouched
    ncopy = 0
    for pat in ("DeliveryTimes_*", "DeliveryCounts_*"):
        for f in glob.glob(os.path.join(a.folder, pat)):
            shutil.copy2(f, os.path.join(outdir, os.path.basename(f)))
            ncopy += 1

    R.to_csv(os.path.join(outdir, "Repair_report.csv"), index=False)

    print(f"  {'file':26s} {'behaviour':10s} {'raw':>5s} {'merged':>7s} "
          f"{'filled':>7s}  {'counts':>11s}")
    for r in R.itertuples():
        print(f"  {r.file[:26]:26s} {r.behaviour:10s} {r.raw_events:5d} "
              f"{r.merged_events:7d} {r.gaps_filled:7d}  "
              f"{r.counts_trustworthy:>11s}")

    print(f"\n  threshold sensitivity (events at each merge gap)")
    cols = [f"n_at_{g:g}s" for g in SENS]
    print(f"  {'file':26s} {'behaviour':10s} " + "".join(f"{c[5:]:>8s}" for c in cols))
    for r in R.to_dict("records"):
        print(f"  {r['file'][:26]:26s} {r['behaviour']:10s} "
              + "".join(f"{r[c]:8d}" for c in cols))

    md = os.path.join(outdir, "Repair_report.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("# Repair report - fast repeated clicks merged\n\n")
        fh.write(f"- merge gap: **{a.gap:.2f} s**\n")
        fh.write(f"- originals: `{a.folder}` (unmodified)\n")
        fh.write(f"- sessions repaired: {len(files)}, "
                 f"delivery files copied: {ncopy}\n\n")
        fh.write("## Counts before and after\n\n")
        fh.write(md_table(R[["file", "behaviour", "raw_events",
                             "merged_events", "gaps_filled",
                             "counts_trustworthy"]]))
        fh.write("\n\n## Threshold sensitivity\n\n")
        fh.write("Where the count plateaus across thresholds the merge choice "
                 "is safe. Where it keeps falling there is no clean boundary "
                 "between artefact and real gaps, so treat that count as "
                 "approximate.\n\n")
        fh.write(md_table(R[["file", "behaviour"] + cols]))
        fh.write("\n\n## Read this before using the corrected files\n\n"
                 "- **Counts are the usable measure.** `guardMin` is set to 0 "
                 "in the corrected `.mat`, so no duration filtering is "
                 "applied.\n"
                 "- **Durations are NOT usable.** When a key is tapped rather "
                 "than held, the merged span is the span of the clicking, not "
                 "of the behaviour. Do not report `% of time`.\n"
                 "- The `> 2 s` guarding criterion is no longer applied by "
                 "code. It is the scorer's judgement and must be stated as "
                 "such in the methods.\n")

    # ---- delivery / metadata warnings ----
    if not MD.empty:
        bad = MD[MD.missing_metadata != ""]
        if not bad.empty:
            print("\n  *** MISSING METADATA - step2 cannot join treatment ***")
            for r in bad.itertuples():
                print(f"  {r.file[:40]:40s} empty: {r.missing_metadata}")
            print("  Fill Session / Mouse ID / Sex / Day in the dialog when "
                  "you re-score.")
        odd = MD[MD.n_runs > 4]
        if not odd.empty:
            print("\n  *** BLOCK STRUCTURE NOT CLEAN ***")
            for r in odd.itertuples():
                print(f"  {r.file[:40]:40s} {r.n_runs} runs of consecutive "
                      f"same-stimulus marks, expected 4")
            print("  A blocked session gives exactly 4 runs. More means the "
                  "blocks are")
            print("  interleaved, or some marks carry the wrong stimulus key.")
    if not DQ.empty:
        DQ.to_csv(os.path.join(outdir, "Delivery_QC.csv"), index=False)
        mis = DQ[DQ.kind.str.startswith("MIS-KEY")]
        print(f"\n  {len(DQ)} delivery mark pair(s) within 0.5 s "
              f"({len(mis)} with DIFFERENT stimulus types)")
        for r in mis.head(12).itertuples():
            print(f"    {r.file[:24]:24s} {r.t1:8.1f}s {r.stim1:12s} -> "
                  f"{r.t2:8.1f}s {r.stim2:12s}  dt {r.dt_s:.2f}s")
        if len(mis) > 12:
            print(f"    ... {len(mis) - 12} more, see Delivery_QC.csv")
        print("  These are NOT auto-corrected: which mark carries the right")
        print("  stimulus is ambiguous from the video. Check your written record.")
    if not BR.empty:
        BR.to_csv(os.path.join(outdir, "Block_runs.csv"), index=False)

    print(f"\n  wrote {os.path.join(outdir, 'Repair_report.csv')}")
    print(f"  wrote {md}")
    print(f"  copied {ncopy} delivery file(s) through unchanged")
    print(f"\n  NEXT: python step1_block_measures.py \"{outdir}\"")
    print("  (the originals are untouched - run step1 on either folder)")


if __name__ == "__main__":
    main()
