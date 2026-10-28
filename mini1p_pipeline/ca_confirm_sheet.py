"""ca_confirm_sheet.py - ONE sheet per mouse with every Day-2 item that needs your eye (Day 1 | Day 2, max dF/F).

Rows, in this order (each = the Day-1 window and the same place on Day 2, 51 x 51 px, outlines, the reason):
  RECOVERED  Day-1 neuron that was not found: its Day-1 footprint placed on Day 2 (ca_day2_recover.py), the places
             you marked in yellow first                                                      -> the same neuron?
  D1 OFF     left out by MY Day-1 rule (not by you), but a cell on Day 2 (kept, yellow)       -> keep? add on Day 1?
  CHECK      Day-1 neuron followed to Day 2, kept, but some evidence weak (orange)            -> keep or delete?
  DELETED    Day-1 neuron at its place on Day 2, the Day-2 rules deleted it (magenta)         -> add back?
  NOT FOUND  Day-1 neuron with no Day-2 component, but its pattern is still there (magenta x)  -> (no trace without a component)
  NEW KEPT   new on Day 2, kept by the strict rule (cyan)                                     -> keep?
  NEW MAYBE  new, dropped by the strict rule, looks like your Day-1 keeps (red)               -> add?
  SPLIT      a bright body cut into >= 3 deleted pieces                                       -> one neuron?
  REDO       your Day-1 delete, but a bright neuron-like body on Day 2                         -> still delete?
  PAIR       two kept neighbours with anti-correlated traces                                  -> one neuron split?
Each item has its Day-2 id; answers go to ca_exclude_rois.py (--ids / --keep-ids) and ca_day2_decisions.py --unpair.
Writes analysis\\crossday\\<m>_CONFIRM_p<k>.png and <m>_confirm_items.csv

  python ca_confirm_sheet.py --mouse 5941
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
from ca_day2_decisions import BASE, inv, load_day, sample, window  # noqa: E402

COLS = {"RECOVERED": "#f012be", "D1 OFF": "#ffdc00", "CHECK": "#ff9f1a", "DELETED": "#f012be", "NOT FOUND": "#f012be", "NEW KEPT": "#00d0ff", "NEW MAYBE": "#ff4136",
        "SPLIT": "#ffdc00", "REDO": "#ffdc00", "PAIR": "#ff9f1a"}
ORDER = list(COLS)
H = 25


def items(m):
    cd = os.path.join(BASE, "analysis", "crossday")
    t2 = f"{m}_d2"
    root = os.path.join(BASE, "mini1p_out", "20261021", t2)
    R = pd.read_csv(os.path.join(root, "4_cell_review", f"{t2}_cell_review.csv"))
    att = R.get("attention", pd.Series([""] * len(R))).fillna("").astype(str)
    tc = R.get("track_class", pd.Series([""] * len(R))).fillna("").astype(str)
    out = []
    for j in range(len(R)):
        i2, d1 = int(R.id.iloc[j]), R.day1_id.iloc[j]
        kp = int(R.keep.iloc[j]) == 1
        base = dict(day2_id=i2, day1_id=int(d1) if pd.notna(d1) else 0, rc=(float(R.row.iloc[j]), float(R.col.iloc[j])))
        if att.iloc[j].startswith("recovered Day-1"):
            import re as _re
            mk = _re.search(r"your yellow mark (\d+)", att.iloc[j])
            out.append(dict(kind="RECOVERED", why=att.iloc[j][:160], mark=int(mk.group(1)) if mk else 0, **base))
            continue
        if "left out by my Day-1 rule" in att.iloc[j]:
            out.append(dict(kind="D1 OFF", why=att.iloc[j][:110], **base))
            continue
        if kp and tc.iloc[j] == "B":
            out.append(dict(kind="CHECK", why=att.iloc[j].replace(" - check", ""), **base))
        if (not kp) and tc.iloc[j] == "C":
            out.append(dict(kind="DELETED", why=str(R.note.iloc[j])[:90], **base))
        if kp and tc.iloc[j] == "new":
            out.append(dict(kind="NEW KEPT", why=att.iloc[j][:90], **base))
        if (not kp) and tc.iloc[j] == "new" and "your call" in att.iloc[j]:
            out.append(dict(kind="NEW MAYBE", why=att.iloc[j][:90], **base))
        if att.iloc[j].startswith("bright body split"):
            out.append(dict(kind="SPLIT", why=att.iloc[j][:90], **base))
        if att.iloc[j].startswith("re-check: your Day-1 delete"):
            out.append(dict(kind="REDO", why=att.iloc[j][:90], **base))
        if "anti-correlated" in att.iloc[j]:
            out.append(dict(kind="PAIR", why=att.iloc[j][att.iloc[j].find("overlaps"):][:90], **base))
    nf = os.path.join(cd, f"{m}_day1_not_found.csv")
    if os.path.isfile(nf) and os.path.getsize(nf) > 5:              # an empty file = every Day-1 neuron found
        for r in pd.read_csv(nf).itertuples():
            if True:                                  # every one (the 'visible' test disagreed with your eye, 10-22)
                out.append(dict(kind="NOT FOUND", why=f"no Day-2 component; pattern {r.pattern_there:.2f}, "
                                                     f"nearest component {r.nearest_day2_px:.0f} px",
                                day2_id=0, day1_id=int(r.day1_id), rc=(float(r.day2_row), float(r.day2_col))))
    out.sort(key=lambda d: (ORDER.index(d["kind"]), 0 if d.get("mark") else 1, d.get("mark", 0)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", required=True)
    ap.add_argument("--per-page", type=int, default=24)
    a = ap.parse_args(argv)
    m = a.mouse
    D1 = load_day(os.path.join(BASE, "mini1p_out", "20261019"), f"{m}_d1")
    D2 = load_day(os.path.join(BASE, "mini1p_out", "20261021"), f"{m}_d2")
    J = json.load(open(os.path.join(BASE, "analysis", "crossday", f"{m}_tracking.json")))
    A = np.array(J["transform"]["matrix_rowcol"]); Ai = inv(A)
    It = items(m)
    cd = os.path.join(BASE, "analysis", "crossday")
    pd.DataFrame([{k: v for k, v in d.items() if k != "rc"} | dict(row=d["rc"][0], col=d["rc"][1]) for d in It]).to_csv(
        os.path.join(cd, f"{m}_confirm_items.csv"), index=False)
    import glob
    for old in glob.glob(os.path.join(cd, f"{m}_CONFIRM_p*.png")):
        os.remove(old)
    lo1, hi1 = np.percentile(D1["maxdff"], [1, 99.8]); lo2, hi2 = np.percentile(D2["maxdff"], [1, 99.8])
    id1 = {int(v): k for k, v in enumerate(D1["R"].id.values)}
    for pg in range(max(1, int(np.ceil(len(It) / a.per_page)))):
        chunk = It[pg * a.per_page:(pg + 1) * a.per_page]
        ncol = 4
        nrow = max(1, int(np.ceil(len(chunk) / ncol)))
        fig, axs = plt.subplots(nrow, 2 * ncol, figsize=(2 * ncol * 2.2, nrow * 2.75))
        axs = np.atleast_2d(axs)
        for a_ in axs.ravel():
            a_.set_axis_off()
        for k, it in enumerate(chunk):
            r_, c_ = divmod(k, ncol)
            a1, a2 = axs[r_, 2 * c_], axs[r_, 2 * c_ + 1]
            rc = np.array(it["rc"])
            w1 = sample(D1["maxdff"], Ai, rc, H); w2 = window(D2["maxdff"], rc, H)
            a1.imshow(np.clip((w1 - lo1) / (hi1 - lo1), 0, 1) ** 0.7, cmap="gray")
            a2.imshow(np.clip((w2 - lo2) / (hi2 - lo2), 0, 1) ** 0.7, cmap="gray")
            col = COLS[it["kind"]]
            if it["day1_id"] and it["day1_id"] in id1:
                f1 = sample(D1["S"][:, :, id1[it["day1_id"]]], Ai, rc, H)
                if f1.max() > 0:
                    a1.contour(f1 >= 0.3 * f1.max(), levels=[0.5], colors=["#2ecc40"], linewidths=1.0)
            if it["day2_id"]:
                f2 = window(D2["S"][:, :, it["day2_id"] - 1], rc, H)
                if f2.max() > 0:
                    a2.contour(f2 >= 0.3 * f2.max(), levels=[0.5], colors=[col], linewidths=1.1)
            a1.set_title(f"{it['kind']}\nDay 1 " + (f"#{it['day1_id']}" if it["day1_id"] else "(none)"), fontsize=8, color=col)
            a2.set_title(f"Day 2 " + (f"#{it['day2_id']}" if it["day2_id"] else "-"), fontsize=8, color=col)
            words, lines, cur = it["why"].split(), [], ""
            for wd in words:
                if len(cur) + len(wd) > 34:
                    lines.append(cur); cur = wd
                else:
                    cur = (cur + " " + wd).strip()
            lines.append(cur)
            a2.text(-0.02, -0.04, "\n".join(lines[:3]), transform=a2.transAxes, ha="center", va="top", fontsize=6.3)
        n = {k: sum(1 for d in It if d["kind"] == k) for k in ORDER}
        fig.suptitle(f"{m} Day 2 - items for your decision (PROVISIONAL selection). " + ", ".join(f"{k} {v}" for k, v in n.items() if v)
                     + f"   (page {pg + 1}; green outline = the Day-1 neuron; max dF/F, 51 px)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.2)
        fig.savefig(os.path.join(cd, f"{m}_CONFIRM_p{pg + 1}.png"), dpi=100); plt.close(fig)
    print(f"{m}: {len(It)} items for your decision -> {cd}\\{m}_CONFIRM_p*.png")


if __name__ == "__main__":
    main()
