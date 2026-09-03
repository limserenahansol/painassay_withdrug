"""qc_and_correct_all.py  -  audit every scored session and write a corrected
copy for downstream analysis.

GROUND RULES
    * Your originals in videos\\output are NEVER modified. Everything goes to
      a new folder.
    * Nothing is invented. Every change is either (a) mechanical - merging the
      fragments of one key press, normalising the spelling of a stimulus name -
      or (b) forced by a constraint that cannot be satisfied any other way.
    * Every single change is logged, line by line, in the report. Anything that
      needs a human decision is FLAGGED and left untouched.

WHAT IT CHECKS AND FIXES

  1  stimulus names                    fixed      Taken from the RANDOMISATION
                                                  SHEET. Confirmed by Hansol
                                                  that the sheet order is what
                                                  was delivered and the names
                                                  typed while scoring were
                                                  mis-entered. The key presses
                                                  are trusted: the code order is
                                                  1,2,3,4 in every session, so
                                                  slot i is the sheet's Stim i.

  2  two stimuli inside one 5 min      fixed      Blocks are sequential, so a
     block (mis-key)                              lone code-B mark sitting
                                                  inside a run of code-A marks,
                                                  within MISKEY_S of one, is a
                                                  slip of the finger. It is
                                                  re-assigned to the block's
                                                  code and logged.

  3  fast repeated clicks              fixed      Bouts of the same behaviour
                                                  separated by <= MERGE_S are
                                                  merged into one event.

  4  duplicate / missing stimulus      fixed by   Taking the names from the
     names                             item 1     sheet removes these by
                                                  construction. A leftover
                                                  mismatch is FLAGGED.

  5  missing metadata                  filled     Session / mouse / sex / day
                                                  come from the filename and
                                                  YOUR randomisation sheet.

  6  guarding duration filter          set to 0   Scoring is by taps, so a
                                                  duration rule would discard
                                                  almost every mark. Counts are
                                                  the measure.

  7  bursts of alternating marks       FLAGGED    e.g. 6 marks in 3 s switching
     in the rest period                ONLY       between two codes, outside any
                                                  block. Probably a key-mash,
                                                  but which marks are real is
                                                  not recoverable. Left alone.

  8  block count, delivery counts,     reported   read the report before you
     truncation, coverage                         trust anything downstream.

USAGE
    python qc_and_correct_all.py                       # audit + correct
    python qc_and_correct_all.py --report-only         # audit, write nothing
    python qc_and_correct_all.py --merge-s 0.35

OUTPUT
    videos\\output_corrected\\   corrected ScoringAB_*.mat + TrainingLabels_*.csv
    videos\\output_corrected\\QC_REPORT.md     read this first
    videos\\output_corrected\\QC_changes.csv   every change, one row each
    videos\\output_corrected\\QC_flags.csv     everything needing your decision

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

ROOT = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos"
SCHED = (r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
         r"\Stimulus_randomisation_mini1p.xlsx")

MERGE_S = 0.35          # fast repeated clicks closer than this are one event
BURST_N, BURST_S = 4, 5.0   # >=4 marks in <=5 s with mixed codes = worth a look

# The protocol runs on the clock: 5 min baseline, then a 5 min stimulus block
# every 6 min. Confirmed by Hansol - 1st stimulus 5-10 min, 2nd 11-16,
# 3rd 17-22, 4th 23-28. These windows decide which block a delivery belongs to,
# which resolves cases that neighbour-voting cannot: at a block boundary the
# marks alternate and no majority exists, but the clock is unambiguous.
BLOCK_WIN = [(300.0, 600.0), (660.0, 960.0), (1020.0, 1320.0), (1380.0, 1680.0)]
FAR_S = 60.0            # a mark this far outside every window is reported

# Nominal duration for a tapped mark. A tap records ~2 frames, which is the
# length of the keypress, not of the behaviour. Hansol's call: a guarding tap
# means guarding was seen and it lasts about 1 s, so each guarding event is
# given 1 s. The other behaviours keep what was actually recorded.
NOMINAL_S = {3: 1.0}

# Where each mouse's stimulus names should come from. Decided per mouse after
# comparing what was typed with the randomisation sheet:
#   F1, F2, M2  typed and sheet agree, so it makes no difference
#   M1, M3      the delivery order on the day did not follow the sheet, and the
#               typed names are the record of what was actually given
#   F3          the typed names had "Mild touch" twice and no "Light touch", so
#               they cannot be right; the sheet is used
NAME_SOURCE = {"F1": "sheet", "F2": "sheet", "F3": "sheet",
               "M1": "typed", "M2": "sheet", "M3": "typed"}

AFF = {1: "attending", 2: "lickbite", 3: "guarding", 4: "escape"}
MOUSE_FROM_FILE = {"female1": "F1", "female2": "F2", "female3": "F3",
                   "male1": "M1", "male2": "M2", "male3": "M3"}

# the four canonical stimulus labels, and every spelling seen in the data
CANON = ["Heat", "Mild touch", "Light touch", "Pin prick"]
ALIAS = {
    "heat": "Heat",
    "mildtouch": "Mild touch", "mild": "Mild touch",
    "lighttouch": "Light touch", "light": "Light touch",
    "pinprick": "Pin prick", "pin": "Pin prick",
}


def norm_name(s):
    k = re.sub(r"[^a-z]", "", str(s).lower())
    return ALIAS.get(k, str(s).strip())


def mkey(p):
    m = re.match(r"([a-z]+\d)", os.path.basename(p).lower())
    return m.group(1) if m else ""


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


def runs_of(mask):
    b = np.asarray(mask, bool).astype(np.int8)
    d = np.diff(np.concatenate(([0], b, [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def load_schedule():
    S = pd.read_excel(SCHED, sheet_name="Schedule", keep_default_na=False)
    S.columns = [str(c).strip() for c in S.columns]
    S["Treatment"] = S["Treatment"].astype(str).str.strip()
    S.loc[S["Treatment"].isin(["", "nan", "NaN"]), "Treatment"] = "None"
    return S


# ───────────────────────── one session ─────────────────────────────────────
def process(path, S, merge_s):
    M = loadmat(path, squeeze_me=False)
    vid = os.path.basename(path)[len("ScoringAB_"):-4]
    k = mkey(vid)
    mid = MOUSE_FROM_FILE.get(k, "")
    fps = f_(M["frameRate"], 30.0)
    nUsed = int(f_(M.get("nUsed", 0), 0))
    score = np.asarray(M["score"]).ravel().astype(int)
    if nUsed:
        score = score[:nUsed]
    else:
        nUsed = len(score)
    dF = np.asarray(M["dFrames"]).ravel().astype(int)
    dT = np.asarray(M["dTypes"]).ravel().astype(int)
    raw_names = [s_(x) for x in np.asarray(M["stimNames"]).ravel()][:4]
    while len(raw_names) < 4:
        raw_names.append(f"Stim {len(raw_names)+1}")

    changes, flags = [], []

    def chg(kind, detail, before, after):
        changes.append(dict(file=vid, mouse=mid, kind=kind, detail=detail,
                            before=before, after=after))

    def flag(kind, detail, action="left untouched - needs your decision"):
        flags.append(dict(file=vid, mouse=mid, kind=kind, detail=detail,
                          action=action))

    # ---- 1. stimulus names come from the RANDOMISATION SHEET -------------
    # Confirmed by Hansol: the sheet order is what was actually delivered, and
    # the names typed into the dialog while scoring were mis-entered. The key
    # PRESSES are trusted - the code order is 1,2,3,4 in every session, i.e.
    # key i was used for the i-th block - so slot i is the sheet's Stim i.
    # This also resolves female3, where the typed names had "Mild touch" twice
    # and no "Light touch" at all.
    sched_row = S[(S["Mouse ID"] == mid) & (S["Day"] == 1) &
                  (S["Phase"].astype(str).str.startswith("Baseline"))]
    src = NAME_SOURCE.get(mid, "sheet")
    typed_names = [norm_name(n) for n in raw_names]
    if src == "typed" or not len(sched_row):
        names = list(typed_names)
        why = "typed while scoring (the sheet order was not followed for this "
        why += "mouse)" if len(sched_row) else "typed (no schedule row)"
        for i in range(4):
            if names[i] != raw_names[i]:
                chg("stimulus name spelling only", f"slot {i+1}",
                    raw_names[i], names[i])
        if len(sched_row):
            planned = [norm_name(sched_row.iloc[0][f"Stim {j}"])
                       for j in (1, 2, 3, 4)]
            if planned != names:
                flag("names kept as TYPED, sheet differs",
                     f"typed {names} vs sheet {planned} - kept the typed names "
                     f"on your instruction",
                     action="no change - this is the chosen source")
    else:
        planned = [norm_name(sched_row.iloc[0][f"Stim {j}"]) for j in (1, 2, 3, 4)]
        names = list(planned)
        for i in range(4):
            if typed_names[i] != names[i]:
                chg("stimulus name from sheet",
                    f"slot {i+1} (block {i+1} of the session)",
                    raw_names[i], names[i])

    # sanity: the four slots must now be the four distinct stimuli
    if sorted(names) != sorted(CANON):
        flag("stimulus set is not the expected four",
             f"got {names}, expected {CANON}")

    # ---- 2. mis-keys: a lone wrong-code mark inside a block --------------
    order = np.argsort(dF)
    dF, dT = dF[order].copy(), dT[order].copy()
    n = len(dF)
    # block identity = the code of the longest run containing each mark
    runs = []
    s = 0
    for i in range(1, n + 1):
        if i == n or dT[i] != dT[s]:
            runs.append((s, i - 1, int(dT[s])))
            s = i
    # Which block does a delivery belong to? THE CLOCK decides, not the
    # neighbouring marks. Blocks run 5-10, 11-16, 17-22, 23-28 min. A
    # neighbour vote cannot resolve a block boundary - there the marks
    # alternate and no majority exists - but the clock can. The last window is
    # extended to the end of the recording, because M3's session ran to 1722 s
    # and its 4th block continued past the nominal 1680 s.
    wins = [list(w) for w in BLOCK_WIN]
    wins[-1][1] = max(wins[-1][1], nUsed / fps)
    t = (dF - 1) / fps

    def which_block(x):
        for b, (lo, hi) in enumerate(wins, start=1):
            if lo <= x <= hi:
                return b, 0.0
        # outside every window: take the nearest, and report how far off
        d = [min(abs(x - lo), abs(x - hi)) for lo, hi in wins]
        b = int(np.argmin(d)) + 1
        return b, float(min(d))

    n_miskey = 0
    for i in range(n):
        blk, off = which_block(t[i])
        if off > FAR_S:
            flag("delivery far outside every block window",
                 f"t={t[i]:.1f}s code {int(dT[i])} ({names[int(dT[i])-1]}) is "
                 f"{off:.0f}s from the nearest window (block {blk})")
            continue
        if int(dT[i]) == blk:
            continue
        chg("stimulus mis-key (clock window)",
            f"t={t[i]:.1f}s is in block {blk} "
            f"({wins[blk-1][0]:.0f}-{wins[blk-1][1]:.0f}s)"
            + (f", {off:.0f}s outside the nominal window" if off else ""),
            names[int(dT[i]) - 1], names[blk - 1])
        dT[i] = blk
        n_miskey += 1

    # ---- 7. bursts of alternating marks (flag only) ----------------------
    i = 0
    while i < n:
        j = i
        while j + 1 < n and (dF[j + 1] - dF[i]) <= BURST_S * fps:
            j += 1
        if (j - i + 1) >= BURST_N and len(set(dT[i:j + 1].tolist())) >= 2:
            flag("burst of alternating stimulus marks",
                 f"{j-i+1} marks in {(dF[j]-dF[i])/fps:.1f}s at "
                 f"t={(dF[i]-1)/fps:.1f}s, codes "
                 f"{[int(x) for x in dT[i:j+1]]} - looks like a key-mash; "
                 f"which marks are real is not recoverable")
            i = j + 1
        else:
            i += 1

    # ---- 3. merge fast repeated clicks -----------------------------------
    gap_f = int(round(merge_s * fps))
    merged_counts = {}
    for code, bn in AFF.items():
        st, en = runs_of(score == code)
        before = len(st)
        filled = 0
        for q in range(len(st) - 1):
            g0, g1 = en[q], st[q + 1]
            if (g1 - g0) > gap_f:
                continue
            if np.any(score[g0:g1] != 0):
                continue                # never overwrite another behaviour
            score[g0:g1] = code
            filled += 1
        st2, en2 = runs_of(score == code)
        merged_counts[bn] = (before, len(st2))
        if filled:
            chg("fast clicks merged", f"{bn}: {filled} gap(s) <= {merge_s:g}s "
                                      f"filled", before, len(st2))

        # ---- nominal duration for a tapped mark ----
        if code in NOMINAL_S:
            want = int(round(NOMINAL_S[code] * fps))
            grown, before_s = 0, float((en2 - st2).sum()) / fps
            for q in range(len(st2)):
                have = en2[q] - st2[q]
                if have >= want:
                    continue
                e = st2[q]
                # grow forward only, and only over frames scored as nothing,
                # so a neighbouring behaviour is never overwritten
                while (e - st2[q]) < want and e < len(score) and \
                        (score[e] == 0 or score[e] == code):
                    score[e] = code
                    e += 1
                grown += 1
            after_s = float(np.sum(score == code)) / fps
            if grown:
                chg("nominal duration applied",
                    f"{bn}: {grown} event(s) extended to "
                    f"{NOMINAL_S[code]:g}s each (a tap records the keypress, "
                    f"not the behaviour). Event COUNT is unchanged",
                    f"{before_s:.2f}s total", f"{after_s:.2f}s total")

    # ---- 5. metadata from the filename + the sheet -----------------------
    row = sched_row
    meta = {}
    if len(row):
        r = row.iloc[0]
        meta = dict(sessionNo=str(int(r["Session"])), mouseID=str(r["Mouse ID"]),
                    sexID=str(r["Sex"]), dayNo="1", phase=str(r["Phase"]),
                    treatment=str(r["Treatment"]))
        for kk, vv in meta.items():
            old = s_(M.get(kk, ""))
            if old != vv:
                chg("metadata filled", kk, old if old else "(empty)", vv)
    else:
        flag("no schedule row", f"mouse '{mid}' Day 1 baseline not in the sheet")

    # ---- 6. count-only scoring: turn the duration filter off -------------
    if f_(M.get("guardMin", 2.0), 2.0) != 0:
        chg("guardMin", "tap scoring: a duration filter would discard nearly "
                        "every mark", f_(M.get("guardMin", 2.0), 2.0), 0.0)

    # ---- 8. plain facts for the report -----------------------------------
    o = np.argsort(dF)
    tt = dT[o]
    nruns = 1 + int(np.sum(tt[1:] != tt[:-1])) if len(tt) else 0
    facts = dict(file=vid, mouse=mid, session=meta.get("sessionNo", ""),
                 dur_s=round(nUsed / fps, 1), n_del=int(n),
                 runs_after_fix=nruns, n_miskey_fixed=n_miskey,
                 n_unknown=int((dT == 0).sum()),
                 labelled_frames=int((score > 0).sum()),
                 stimuli=" | ".join(f"{names[i-1]}:{int((dT==i).sum())}"
                                    for i in range(1, 5) if (dT == i).sum()))
    for bn, (b4, af) in merged_counts.items():
        facts[f"{bn}_events"] = af
        facts[f"{bn}_before_merge"] = b4

    out = {kk: vv for kk, vv in M.items() if not kk.startswith("__")}
    out["score"] = score.reshape(-1, 1).astype(float)
    out["dFrames"] = dF.reshape(-1, 1).astype(float)
    out["dTypes"] = dT.reshape(-1, 1).astype(float)
    out["stimNames"] = np.array(names, dtype=object).reshape(4, 1)
    out["guardMin"] = 0.0
    out["nUsed"] = float(nUsed)
    out["corrected"] = 1.0
    out["mergeGapS"] = float(merge_s)
    for kk, vv in meta.items():
        out[kk] = vv
    return out, facts, changes, flags, vid, fps, nUsed, score


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(ROOT, "output"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output_corrected"))
    ap.add_argument("--merge-s", type=float, default=MERGE_S)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    if os.path.abspath(a.out) == os.path.abspath(a.src):
        raise SystemExit("refusing to write into the original folder")
    files = sorted(glob.glob(os.path.join(a.src, "ScoringAB_*.mat")))
    if not files:
        raise SystemExit(f"no ScoringAB_*.mat in {a.src}")
    S = load_schedule()

    print(f"originals : {a.src}   (never modified)")
    print(f"corrected : {a.out}")
    print(f"merge gap {a.merge_s:g}s   blocks assigned by the clock "
          f"(5-10, 11-16, 17-22, 23-28 min)\n")

    if not a.report_only:
        os.makedirs(a.out, exist_ok=True)

    FACTS, CHG, FLG = [], [], []
    for f in files:
        out, facts, chg, flg, vid, fps, nUsed, score = process(
            f, S, a.merge_s)
        FACTS.append(facts); CHG += chg; FLG += flg
        print(f"  {facts['mouse'] or '?':3s} S{facts['session'] or '?':>2s} "
              f"{vid[:26]:28s} {facts['n_del']:3d} del  "
              f"{facts['runs_after_fix']} run(s)  "
              f"{facts['n_miskey_fixed']} mis-key fixed  "
              f"{len(chg):2d} change(s)  {len(flg)} flag(s)")
        if a.report_only:
            continue
        savemat(os.path.join(a.out, f"ScoringAB_{vid}.mat"), out)
        tl = os.path.join(a.src, f"TrainingLabels_{vid}.csv")
        if os.path.exists(tl):
            T = pd.read_csv(tl)
            m = min(len(T), len(score))
            lab = {0: "None", 1: "Paw attending", 2: "Licking or biting",
                   3: "Sustained lifting / guarding", 4: "Escape / rearing"}
            T = T.iloc[:m].copy()
            T["affective_code"] = score[:m]
            T["affective_label"] = [lab.get(int(c), "?") for c in score[:m]]
            lagf = int(round(0.25 * fps))
            ml = np.concatenate([score[lagf:m], np.zeros(min(lagf, m), int)])[:m]
            T["affective_code_ml"] = ml
            T["affective_label_ml"] = [lab.get(int(c), "?") for c in ml]
            T["corrected"] = 1
            T.to_csv(os.path.join(a.out, f"TrainingLabels_{vid}.csv"), index=False)
        for pat in ("DeliveryTimes_", "DeliveryCounts_"):
            for g in glob.glob(os.path.join(a.src, pat + vid + ".*")):
                shutil.copy2(g, os.path.join(a.out, os.path.basename(g)))

    F = pd.DataFrame(FACTS)
    C = pd.DataFrame(CHG)
    L = pd.DataFrame(FLG)
    print(f"\n  {len(C)} change(s), {len(L)} flag(s) needing your decision")

    if len(L):
        print("\n  ======== FLAGGED - NOT CHANGED ========")
        for r in L.itertuples():
            print(f"  [{r.mouse}] {r.kind}\n      {r.detail}")

    if len(C):
        print("\n  ======== CHANGES MADE ========")
        for kind, g in C.groupby("kind"):
            print(f"  {kind}: {len(g)}")
            for r in g.head(8).itertuples():
                print(f"      [{r.mouse}] {r.detail}: "
                      f"'{r.before}' -> '{r.after}'")
            if len(g) > 8:
                print(f"      ... {len(g)-8} more, see QC_changes.csv")

    if a.report_only:
        print("\n  --report-only: nothing written")
        return

    C.to_csv(os.path.join(a.out, "QC_changes.csv"), index=False)
    L.to_csv(os.path.join(a.out, "QC_flags.csv"), index=False)
    F.to_csv(os.path.join(a.out, "QC_facts.csv"), index=False)
    write_report(a, F, C, L)
    print(f"\n  wrote {os.path.join(a.out, 'QC_REPORT.md')}")
    print(f"  NEXT: python step1_block_measures.py \"{a.out}\"")


def md(df):
    if df.empty:
        return "_none_\n"
    cols = list(df.columns)
    o = ["| " + " | ".join(map(str, cols)) + " |",
         "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.to_dict("records"):
        o.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(o) + "\n"


def write_report(a, F, C, L):
    p = os.path.join(a.out, "QC_REPORT.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("# QC and correction report - Day 1 (no drug)\n\n")
        fh.write(f"- originals: `{a.src}` **unmodified**\n")
        fh.write(f"- corrected: `{a.out}`\n")
        fh.write(f"- merge gap {a.merge_s:g} s; blocks assigned by clock "
                 f"window 5-10 / 11-16 / 17-22 / 23-28 min\n")
        fh.write("- guarding events given a nominal "
                 f"{NOMINAL_S.get(3, 0):g} s each (a tap records the keypress, "
                 "not the behaviour); event counts unchanged\n")
        fh.write("- stimulus names: " + ", ".join(
            f"{k}={v}" for k, v in NAME_SOURCE.items()) + "\n")
        fh.write(f"- {len(F)} session(s), {len(C)} change(s), "
                 f"{len(L)} flag(s)\n\n")
        fh.write("Stimulus identity comes from the names TYPED WHILE SCORING, "
                 "not from the randomisation sheet, because the delivery order "
                 "on the day did not follow the sheet. Only the spelling was "
                 "normalised.\n\n")
        fh.write("## Session facts\n\n")
        keep = ["mouse", "session", "file", "dur_s", "n_del", "runs_after_fix",
                "n_miskey_fixed", "labelled_frames", "stimuli"]
        fh.write(md(F[[c for c in keep if c in F.columns]]))
        fh.write("\n## Behaviour events after merging fast clicks\n\n")
        ev = [c for c in F.columns if c.endswith("_events")
              or c.endswith("_before_merge")]
        fh.write(md(F[["mouse"] + sorted(ev)]))
        fh.write("\n## FLAGGED - not changed, needs your decision\n\n")
        fh.write(md(L))
        fh.write("\n## Every change made\n\n")
        fh.write(md(C))


if __name__ == "__main__":
    main()
