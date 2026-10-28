"""ca_apply_confirmation.py - your answers in CONFIRM_DAY2_<date>\\CONFIRM_LIST_DAY2.xlsx -> the Day-2 (and, if asked, Day-1)
selection, then the list of mice to re-run.

your_answer (case and spaces do not matter):
  keep / 유지 / yes / y / o                -> the Day-2 neuron is kept (ca_exclude_rois.py --keep-ids, your decision)
  delete / 삭제 / no / n / x               -> deleted (--ids)
  not same / 다른 / 다른 뉴런               -> kept, but NOT the same neuron as the Day-1 one (ca_day2_decisions.py --unpair)
  ... day 1 / day1 / d1 / 'Day 1에도'       -> (D1 OFF rows) also added on Day 1 (ca_exclude_rois on <m>_d1 + provenance)
  empty                                   -> as it is now
A row you keep that the Day-2 rules had deleted (class C, e.g. a recovered neuron) becomes a Day 1 <-> Day 2 pair: you said
it is that neuron. Mice in --confirm get <m>_d2_CONFIRMED.flag (all four -> ALL_CONFIRMED.flag: no PROVISIONAL mark).

  python ca_apply_confirmation.py [--xlsx <file>] [--confirm 5941 5939 1191 5937] [--dry-run]
Prints what it did and the command for the re-run (RUN_after_confirmation.sh <mice>).
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import pandas as pd

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
PIPE = os.path.join(BASE, "mini1p_pipeline")
OUT2 = os.path.join(BASE, "mini1p_out", "20261021")
OUT1 = os.path.join(BASE, "mini1p_out", "20261019")


def parse(ans):
    a = str(ans).strip().lower()
    if not a or a == "nan":
        return None, False
    day1 = bool(re.search(r"day\s*1|\bd1\b|day1", a))
    if re.search(r"not\s*same|다른", a):
        return "unpair", day1
    if re.search(r"\bkeep\b|유지|^yes$|^y$|^o$|^ok$", a):
        return "keep", day1
    if re.search(r"delete|삭제|^no$|^n$|^x$", a):
        return "delete", day1
    return "?", day1


def run(cmd, dry):
    print("  $", " ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="")
    ap.add_argument("--confirm", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    x = a.xlsx or sorted(glob.glob(os.path.join(BASE, "CONFIRM_DAY2_*", "CONFIRM_LIST_DAY2.xlsx")))[-1]
    print("answers from", x)
    changed2, changed1, unknown = set(), set(), []
    for m in ("5941", "5939", "1191", "5937"):
        try:
            T = pd.read_excel(x, sheet_name=m, dtype=str)
        except ValueError:
            continue
        keep, dele, unp, d1keep = [], [], [], []
        for r in T.itertuples():
            act, day1 = parse(r.your_answer)
            if act is None:
                continue
            d2 = str(r.day2_id).strip()
            if act == "?":
                unknown.append(f"{m} {r.map_label}: '{r.your_answer}'"); continue
            if not d2 or d2 == "nan":
                unknown.append(f"{m} {r.map_label}: no Day-2 component (NOT FOUND) - tell me where"); continue
            d2 = int(float(d2))
            {"keep": keep, "delete": dele, "unpair": unp}[act].append(d2)
            if act == "unpair":
                keep.append(d2)
            if day1 and str(r.kind) == "D1 OFF" and act == "keep":
                d1keep.append(int(float(r.day1_id)))
        if keep or dele:
            cmd = [sys.executable, "-I", os.path.join(PIPE, "ca_exclude_rois.py"), "--day-root", os.path.join(OUT2, f"{m}_d2")]
            if dele:
                cmd += ["--ids"] + [str(i) for i in sorted(set(dele))]
            if keep:
                cmd += ["--keep-ids"] + [str(i) for i in sorted(set(keep))]
            run(cmd, a.dry_run); changed2.add(m)
        if unp:
            run([sys.executable, "-I", os.path.join(PIPE, "ca_day2_decisions.py"), "--mouse", m, "--unpair"] + [str(i) for i in unp], a.dry_run)
            changed2.add(m)
        if d1keep:
            run([sys.executable, "-I", os.path.join(PIPE, "ca_exclude_rois.py"), "--day-root", os.path.join(OUT1, f"{m}_d1"),
                 "--keep-ids"] + [str(i) for i in d1keep], a.dry_run)
            pv = os.path.join(BASE, "analysis", "crossday", f"{m}_day1_provenance.csv")
            if os.path.isfile(pv) and not a.dry_run:
                P = pd.read_csv(pv)
                P.loc[P.id.isin(d1keep), ["keep", "decision_source"]] = [1, "user"]
                P.to_csv(pv, index=False)
            changed1.add(m); changed2.add(m)
        print(f"{m}: keep {sorted(set(keep))}, delete {sorted(set(dele))}, not same {unp}, add on Day 1 {d1keep}")
    for m in a.confirm:
        if not a.dry_run:
            open(os.path.join(OUT2, f"{m}_d2_CONFIRMED.flag"), "w").close()
        print(f"{m}: confirmed")
    if not a.dry_run and len(glob.glob(os.path.join(OUT2, "*_d2_CONFIRMED.flag"))) == 4:
        open(os.path.join(OUT2, "ALL_CONFIRMED.flag"), "w").close()
        print("all four confirmed -> ALL_CONFIRMED.flag (no PROVISIONAL mark)")
    for u in unknown:
        print("  ? not applied:", u)
    mice = sorted(changed2 | set(a.confirm))
    print("re-run: bash", os.path.join(PIPE, "RUN_after_confirmation.sh").replace("\\", "/"),
          (" --day1 " + " ".join(sorted(changed1)) + " --" if changed1 else ""), " ".join(mice))


if __name__ == "__main__":
    main()
