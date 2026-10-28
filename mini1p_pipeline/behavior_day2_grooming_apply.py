"""behavior_day2_grooming_apply.py - your answer to the Day-2 "maybe grooming" clips -> grooming bouts.

  python behavior_day2_grooming_apply.py --yes 5 9 10 12        (clip numbers of day2_grooming_maybe.mp4 = grooming)

Each confirmed clip becomes a grooming bout: the run around the probability peak where the smoothed probability
stays >= half of the peak, at least 1 s, at most the clip (-2 s .. +4 s). The other clips are recorded as "not
grooming". Updates behavior_auto\\20261021\\<tag>_move_groom.npz (backup *_before_maybe.npz once), writes the answers
into day2_grooming_maybe.csv, then re-run behavior_day2_final.py and the behaviour analysis (printed).
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
AUTO = os.path.join(BASE, "behavior_auto", "20261021")
FPS = 30.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", nargs="*", type=int, default=[])
    a = ap.parse_args(argv)
    C = pd.read_csv(os.path.join(AUTO, "grooming", "day2_grooming_maybe.csv"))
    C["your_answer"] = np.where(C["clip"].isin(a.yes), "grooming", "not grooming")
    for tag, G in C[C["clip"].isin(a.yes)].groupby("session"):
        z = os.path.join(AUTO, f"{tag}_move_groom.npz")
        bk = z.replace(".npz", "_before_maybe.npz")
        if not os.path.isfile(bk):
            shutil.copy(z, bk)
        Z = dict(np.load(bk))
        g = Z["groom"].astype(bool).copy(); ps = Z["ps"].astype(float)
        for r in G.itertuples():
            p = int(r.peak_frame); half = 0.5 * ps[p]
            lo, hi = max(0, p - 60), min(len(g), p + 120)
            s = p
            while s > lo and ps[s - 1] >= half:
                s -= 1
            e = p + 1
            while e < hi and ps[e] >= half:
                e += 1
            if e - s < FPS:
                c = (s + e) // 2; s, e = max(lo, c - 15), min(hi, c + 15)
            g[s:e] = True
            print(f"{tag}: clip #{r.clip} -> grooming {s / FPS:.1f}-{e / FPS:.1f} s")
        Z["groom"] = g
        np.savez(z, **Z)
    C.to_csv(os.path.join(AUTO, "grooming", "day2_grooming_maybe.csv"), index=False)
    print("now: python behavior_day2_final.py  and  python behavior_day_analysis.py --final-dir "
          "C:\\Users\\hsollim\\Documents\\HEAL_mini1p_SBI553\\scoring_mini1p\\20261021_final --day d2 --out-dir "
          "C:\\Users\\hsollim\\Documents\\HEAL_mini1p_SBI553\\analysis\\behavior_day2 --arms "
          "5941=vehicle,5939=vehicle,5937=SBI-553,1191=SBI-553")


if __name__ == "__main__":
    main()
