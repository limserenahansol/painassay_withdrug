"""ca_transfer_decisions.py - carry your keep / delete decisions over to the whole-day cells.

Your decisions were made on a short movie (the 5941 5-min test, or a 1-min preview) with
its own EXTRACT run. The whole-day run finds the same cells again, plus some new ones.
For every whole-day candidate:
  1. the two mean images are registered (normalised cross-correlation of the high-passed
     images; this covers the lens crop and any template shift between the two runs)
  2. each reference footprint is shifted into the day frame and matched one-to-one to the
     day footprints (cosine overlap >= --min-cos, Hungarian assignment)
  3. keep = YOUR decision of the matched reference cell (only rows you reviewed);
     a new candidate (no match, or a reference that was never reviewed) gets the review
     rules that reproduce your 5941 decisions 91/91: keep unless the verdict is not_cell
     --strict-new (2026-10-21, after your preview review "too many green, not neuron shapes"):
     a new candidate is kept only if it looks at least as good as the cells you kept - one
     compact body (mass in the biggest piece, solidity), normal size and axis, movie proof
     (event_image_corr), Ca-like transients; the bar = the --strict-q percentile of the day
     candidates matched to your keeps
     --marks-day: positions you circled (day coordinates, ca_marks_to_day.py); the candidate
     there is kept as your decision
The day review table is updated in place (keep, note, source columns) after a backup
<tag>_cell_review_rules.csv; apply_cell_keep then makes the kept file as usual.

Outputs in --out:
  <tag>_decisions.csv        one row per day candidate (match, cosine, source, keep)
  <tag>_decisions_map.png    day mean image: your keeps / your deletes / new keeps /
                             new deletes, and reference cells you kept that the day run
                             did not find (crosses)
  <tag>_decisions.json       counts and the registration shift
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_extract(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False,
                    variable_names=["spatial_weights", "mean_image"])
    S = np.asarray(m["spatial_weights"], dtype=np.float32)
    if S.ndim == 2:
        S = S[:, :, None]
    return S, np.asarray(m["mean_image"], dtype=np.float32)


def highpass(img):
    x = img.astype(np.float64)
    x = gaussian_filter(x, 1.5) - gaussian_filter(x, 12)
    return (x - x.mean()) / (x.std() + 1e-9)


def register(ref_mean, day_mean, margin=60, max_shift=120):
    """offset (dy, dx) such that day pixel (y, x) = ref pixel (y + dy, x + dx)"""
    import cv2
    a = highpass(ref_mean).astype(np.float32)
    b = highpass(day_mean).astype(np.float32)
    tpl = b[margin:-margin, margin:-margin]
    pad = max_shift
    big = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    r = cv2.matchTemplate(big, tpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(r)
    dx = loc[0] - pad - margin
    dy = loc[1] - pad - margin
    return int(dy), int(dx), float(peak)


def footprint_info(S):
    out = []
    for k in range(S.shape[2]):
        f = S[:, :, k]
        ys, xs = np.nonzero(f > 0.1 * max(f.max(), 1e-9))
        if len(ys) == 0:
            out.append(None)
            continue
        w = f[ys, xs]
        out.append(dict(y0=ys.min(), y1=ys.max() + 1, x0=xs.min(), x1=xs.max() + 1,
                        cy=float((ys * w).sum() / w.sum()), cx=float((xs * w).sum() / w.sum())))
    return out


def cosine_shifted(fr, fd, dy, dx, ir, idd):
    """cosine between reference footprint fr (shifted into the day frame) and day footprint fd"""
    y0 = min(ir["y0"] - dy, idd["y0"]); y1 = max(ir["y1"] - dy, idd["y1"])
    x0 = min(ir["x0"] - dx, idd["x0"]); x1 = max(ir["x1"] - dx, idd["x1"])
    H, W = fd.shape
    y0 = max(y0, 0); x0 = max(x0, 0); y1 = min(y1, H); x1 = min(x1, W)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    b = fd[y0:y1, x0:x1].astype(np.float64)
    a = np.zeros_like(b)
    ry0, ry1, rx0, rx1 = y0 + dy, y1 + dy, x0 + dx, x1 + dx
    Hr, Wr = fr.shape
    sy0, sx0 = max(ry0, 0), max(rx0, 0)
    sy1, sx1 = min(ry1, Hr), min(rx1, Wr)
    if sy1 > sy0 and sx1 > sx0:
        a[sy0 - ry0:sy1 - ry0, sx0 - rx0:sx1 - rx0] = fr[sy0:sy1, sx0:sx1]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a * b).sum() / (na * nb)) if na > 0 and nb > 0 else 0.0


# the envelope of the cells YOU kept on the 5941 5-min test (91 decisions): area <= 3.19 x the
# expected soma, <= 2 pieces, axis ratio <= 2.87. A rule-only candidate far outside it (a blob of
# several cells, neuropil, a streak along a vessel edge) is not kept by the rules alone.
MAX_AREA_RATIO, MAX_PIECES, MAX_AXIS = 3.5, 3, 3.5


def mark_footprint(img, r0, c0, radius=10):
    """the soma under a mark on the day max dF/F projection: the connected region above half way
    between the local background and the local peak, within 1.5 radii; a Gaussian if that fails"""
    from scipy.ndimage import label as ndlabel
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(yy - r0, xx - c0)
    near = d <= 0.6 * radius
    ring = (d >= 2 * radius) & (d <= 3 * radius)
    g = np.exp(-d ** 2 / (2 * (0.6 * radius) ** 2))
    if not near.any() or not ring.any():
        return g * (g > 0.05)
    pk = np.unravel_index(np.argmax(np.where(near, img, -np.inf)), img.shape)
    bgv = np.median(img[ring])
    thr = bgv + 0.5 * (img[pk] - bgv)
    bw = (img >= thr) & (d <= 1.5 * radius)
    L, _ = ndlabel(bw)
    if L[pk] == 0:
        return g * (g > 0.05)
    reg = L == L[pk]
    area = reg.sum()
    if area < 0.3 * np.pi * radius ** 2 or area > 3 * np.pi * radius ** 2:
        return g * (g > 0.05)
    f = np.where(reg, img - bgv, 0).clip(0)
    return f / max(f.max(), 1e-12)


def add_mark_footprints(a, Sd, mean_d, marks_missing, regs, missing):
    """marks with no day candidate: soma footprints from the day max projection, written as a
    reference EXTRACT-like file 'your_marks' so rescue_missing_cells puts them in (joint LS)"""
    proj = os.path.join(os.path.dirname(a.day_mat), a.tag + "_projections.mat")
    if os.path.isfile(proj):
        img = np.asarray(sio.loadmat(proj, variable_names=["max_dff"])["max_dff"], float)
    else:
        img = np.asarray(mean_d, float)
    if img.shape != Sd.shape[:2]:
        img = np.asarray(mean_d, float)
    F = np.stack([mark_footprint(img, r0, c0) for r0, c0, _ in marks_missing], 2).astype(np.float32)
    out = os.path.join(a.out, a.tag + "_your_marks_ref.mat")
    sio.savemat(out, {"spatial_weights": F, "mean_image": np.asarray(mean_d, np.float32)})
    regs.append(dict(label="your_marks", mat=os.path.abspath(out), csv="", dy=0, dx=0, ncc_peak=1.0,
                     n_ref=int(F.shape[2]), n_reviewed=int(F.shape[2])))
    Dn = Sd.reshape(-1, Sd.shape[2]); Dn = Dn / np.maximum(np.linalg.norm(Dn, axis=0), 1e-12)
    covered = []
    for k, (r0, c0, mk) in enumerate(marks_missing):
        f = F[:, :, k].ravel(); f = f / max(np.linalg.norm(f), 1e-12)
        cs = Dn.T @ f
        jb = int(np.argmax(cs))
        if cs[jb] >= 0.5:                      # an existing component already is that soma: keep it
            covered.append((jb, mk))
            continue
        missing.append(dict(ref="your_marks", ref_id=k + 1, cy=r0, cx=c0, best_cos=float(cs[jb]), best_day_id=int(jb + 1),
                            mark=int(mk) if np.isfinite(mk) else k + 1))
    print("[transfer] %d of your marks have no day candidate within 10 px: %d covered by an overlapping component (cos >= 0.5), "
          "%d soma footprints from the max projection -> rescue (%s)" % (len(marks_missing), len(covered),
                                                                           len(marks_missing) - len(covered), out))
    return covered


def rule_keep(verdict, area_ratio=np.nan, n_pieces=np.nan, axis_ratio=np.nan):
    if str(verdict) == "not_cell":
        return 0
    if np.isfinite(area_ratio) and area_ratio > MAX_AREA_RATIO:
        return 0
    if np.isfinite(n_pieces) and n_pieces > MAX_PIECES:
        return 0
    if np.isfinite(axis_ratio) and axis_ratio > MAX_AXIS:
        return 0
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-mat", required=True)
    ap.add_argument("--day-review", required=True, help="<tag>_cell_review.csv of the day run")
    ap.add_argument("--ref", nargs=3, action="append", metavar=("MAT", "CSV", "LABEL"), default=[],
                    help="reference EXTRACT mat + its reviewed table + a label (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--min-cos", type=float, default=0.5)
    ap.add_argument("--weak-cos", type=float, default=0.3)
    ap.add_argument("--no-write", action="store_true", help="do not update the day review table")
    ap.add_argument("--strict-new", action="store_true", help="new candidates only with a clear neuron shape")
    ap.add_argument("--strict-q", type=float, default=10.0)
    ap.add_argument("--marks-day", default="", help="csv with fov_row, fov_col in day coordinates")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    Sd, mean_d = load_extract(a.day_mat)
    info_d = footprint_info(Sd)
    R = pd.read_csv(a.day_review)
    K = Sd.shape[2]
    if len(R) != K:
        sys.exit("day review table has %d rows, EXTRACT file %d candidates" % (len(R), K))
    num = lambda c, k: float(pd.to_numeric(R[c], errors="coerce").iloc[k]) if c in R else np.nan
    rows = [dict(id=int(R.id[k]), verdict=R.verdict[k],
                 rule_keep=rule_keep(R.verdict[k], num("area_ratio", k), num("n_pieces", k), num("axis_ratio", k)),
                 ref="", ref_id=np.nan, cos=np.nan, user_keep=np.nan, weak=False) for k in range(K)]
    regs, missing, merged = [], [], []
    for mat, csv, label in a.ref:
        Sr, mean_r = load_extract(mat)
        T = pd.read_csv(csv)
        reviewed = T["reviewed"].fillna(0).astype(int).values if "reviewed" in T else np.zeros(len(T), int)
        if "decision_source" in T:           # a table this script wrote (e.g. Day 1 for Day 2): your decisions count
            reviewed = reviewed | (T["decision_source"].astype(str) == "user").values.astype(int)
        dy, dx, peak = register(mean_r, mean_d)
        regs.append(dict(label=label, mat=os.path.abspath(mat), csv=os.path.abspath(csv), dy=dy, dx=dx, ncc_peak=peak,
                         n_ref=int(Sr.shape[2]), n_reviewed=int(reviewed.sum())))
        print("[transfer] %s: shift day->ref (dy %d, dx %d), NCC peak %.2f, %d of %d reviewed"
              % (label, dy, dx, peak, reviewed.sum(), len(T)))
        info_r = footprint_info(Sr)
        C = np.zeros((Sr.shape[2], K))
        for i in range(Sr.shape[2]):
            if info_r[i] is None:
                continue
            for j in range(K):
                if info_d[j] is None:
                    continue
                if abs(info_r[i]["cy"] - dy - info_d[j]["cy"]) > 25 or abs(info_r[i]["cx"] - dx - info_d[j]["cx"]) > 25:
                    continue
                C[i, j] = cosine_shifted(Sr[:, :, i], Sd[:, :, j], dy, dx, info_r[i], info_d[j])
        ri, dj = linear_sum_assignment(-C)
        matched_r, matched_d = set(), set()
        for i, j in zip(ri, dj):
            if C[i, j] < a.min_cos:
                continue
            matched_r.add(i); matched_d.add(j)
            if not reviewed[i]:
                continue
            if rows[j]["ref"] and rows[j]["cos"] >= C[i, j]:
                continue                       # an earlier reference matched better
            rows[j].update(ref=label, ref_id=int(T.id[i]), cos=float(C[i, j]), user_keep=int(T.keep[i]), weak=False)
        # weak overlap (--weak-cos <= cos < --min-cos): a cell you KEPT whose footprint the day run
        # found in a different shape (e.g. split or merged with a neighbour) - that day candidate is
        # kept if the rules keep it too; deletions are not carried over this way (the rules decide)
        pairs = sorted(((C[i, j], i, j) for i in range(Sr.shape[2]) for j in range(K)
                        if a.weak_cos <= C[i, j] < a.min_cos), reverse=True)
        for cs, i, j in pairs:
            if i in matched_r or j in matched_d or not reviewed[i] or int(T.keep[i]) != 1 or rows[j]["ref"]:
                continue
            if rows[j]["rule_keep"] != 1:
                continue                       # a fragment the rules reject: rescue your footprint instead
            matched_r.add(i); matched_d.add(j)
            rows[j].update(ref=label, ref_id=int(T.id[i]), cos=float(cs), user_keep=1, weak=True)
        for i in range(Sr.shape[2]):
            if reviewed[i] and int(T.keep[i]) == 1 and i not in matched_r and info_r[i] is not None:
                jb = int(np.argmax(C[i]))
                q = dict(ref=label, ref_id=int(T.id[i]), cy=info_r[i]["cy"] - dy, cx=info_r[i]["cx"] - dx,
                         best_cos=float(C[i, jb]), best_day_id=int(R.id[jb]))
                if C[i, jb] >= a.min_cos:
                    # one day component already covers this footprint (it took another of your cells
                    # in the one-to-one match: two of your cells became ONE component). Adding the
                    # footprint again would split the signal between near-identical footprints
                    # (least squares then gives anti-correlated traces) - so it is reported, not rescued
                    merged.append(q)
                else:
                    missing.append(q)

    # your marks (day coordinates): the candidate under each mark
    marked = {}
    marks_missing = []
    if a.marks_day and os.path.isfile(a.marks_day) and os.path.getsize(a.marks_day) > 10:
        Mk = pd.read_csv(a.marks_day)
        mxd = Sd.reshape(-1, K).max(0)
        for _, mrow in Mk.iterrows():
            r0, c0 = int(round(mrow.fov_row)), int(round(mrow.fov_col))
            hit = []
            if 0 <= r0 < Sd.shape[0] and 0 <= c0 < Sd.shape[1]:
                hit = list(np.where(Sd[r0, c0, :] >= 0.3 * np.maximum(mxd, 1e-12))[0])
            if hit:
                j = max(hit, key=lambda q: Sd[r0, c0, q] / max(mxd[q], 1e-12))
            else:
                d = np.hypot(pd.to_numeric(R.row) - mrow.fov_row, pd.to_numeric(R.col) - mrow.fov_col)
                j = int(np.argmin(d))
                if d.iloc[j] > 10:
                    marks_missing.append((float(mrow.fov_row), float(mrow.fov_col), mrow.get("mark", len(marks_missing) + 1)))
                    continue
            marked[j] = mrow.get("mark", len(marked) + 1)
        print("[transfer] your marks: %d of %d on a day candidate" % (len(marked), len(Mk)))
        if marks_missing:
            for j, mk in add_mark_footprints(a, Sd, mean_d, marks_missing, regs, missing):
                marked[j] = mk                 # the soma under the mark is already this day component
    # the clear-neuron-shape bar for new candidates (strict mode)
    shape_ok = np.ones(K, bool)
    bar = {}
    if a.strict_new:
        from skimage.measure import label as sklabel, regionprops
        mf, so = np.zeros(K), np.zeros(K)
        for j in range(K):
            f = Sd[:, :, j]
            if f.max() <= 0:
                continue
            L = sklabel(f >= 0.3 * f.max(), connectivity=2)
            pr = regionprops(L, intensity_image=f)
            mass = np.array([q.intensity_mean * q.area for q in pr])
            k0 = int(np.argmax(mass))
            mf[j], so[j] = mass[k0] / mass.sum(), pr[k0].solidity
        ar = pd.to_numeric(R.get("area_ratio"), errors="coerce").values
        ax_ = pd.to_numeric(R.get("axis_ratio"), errors="coerce").values
        ev = pd.to_numeric(R.get("event_image_corr"), errors="coerce").values
        cl = pd.to_numeric(R.get("ca_like"), errors="coerce").fillna(0).values
        ref_keep = np.array([bool(r["ref"]) and r["user_keep"] == 1 for r in rows]) | np.isin(np.arange(K), list(marked))
        q = a.strict_q
        if ref_keep.sum() >= 5:
            # nanpercentile: rescued rows have no shape numbers in the review table
            bar = dict(main_min=np.nanpercentile(mf[ref_keep], q), sol_min=np.nanpercentile(so[ref_keep], q),
                       area_lo=np.nanpercentile(ar[ref_keep], q / 2), area_hi=np.nanpercentile(ar[ref_keep], 100 - q / 2),
                       axis_max=np.nanpercentile(ax_[ref_keep], 100 - q), ev_min=np.nanpercentile(ev[ref_keep], q))
        else:                                   # too few of your cells: the preview bar of 2026-10-21
            bar = dict(main_min=0.88, sol_min=0.6, area_lo=0.35, area_hi=2.3, axis_max=2.3, ev_min=0.4)
        shape_ok = ((mf >= bar["main_min"]) & (so >= bar["sol_min"]) & (ar >= bar["area_lo"]) & (ar <= bar["area_hi"])
                    & (ax_ <= bar["axis_max"]) & (ev >= bar["ev_min"]) & (cl == 1))
        print("[transfer] strict bar from %d of your cells: %s; new candidates passing %d"
              % (int(ref_keep.sum()), ", ".join("%s %.2f" % kv for kv in bar.items()),
                 int(sum(shape_ok[j] for j in range(K) if not rows[j]["ref"] and j not in marked))))
    day_rev = R["reviewed"].fillna(0).astype(int).values if "reviewed" in R else np.zeros(K, int)
    for k, r in enumerate(rows):
        if day_rev[k]:
            # you reviewed this candidate on the whole day itself: that decision wins
            r["keep"] = int(R.keep[k])
            r["source"] = "your decision on the whole day"
        elif k in marked:
            r["keep"] = 1
            r["source"] = "your decision (mark %s on the preview screenshot)" % marked[k]
        elif r["ref"]:
            r["keep"] = int(r["user_keep"])
            r["source"] = "your decision (%s #%d, cos %.2f%s)" % (r["ref"], r["ref_id"], r["cos"],
                                                                 ", weak overlap: keep only" if r.get("weak") else "")
        elif a.strict_new:
            r["keep"] = int(r["rule_keep"] == 1 and bool(shape_ok[k]))
            r["source"] = ("new candidate: clear neuron shape (strict rule)" if r["keep"]
                           else "new candidate: not a clear neuron shape (strict rule, %s)" % r["verdict"])
        else:
            r["keep"] = r["rule_keep"]
            r["source"] = ("new candidate: rule (%s)" % r["verdict"] if r["verdict"] == "not_cell" or r["rule_keep"]
                           else "new candidate: rule (%s) but larger / more pieces / longer than any cell you kept" % r["verdict"])
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, a.tag + "_decisions.csv"), index=False)

    # map
    fig, ax = plt.subplots(figsize=(11, 11 * mean_d.shape[0] / mean_d.shape[1] + 0.8))
    lo, hi = np.percentile(mean_d, [1, 99.7])
    ax.imshow(mean_d, cmap="gray", vmin=lo, vmax=hi)
    style = {(1, 1): ("lime", "-", "your keep"), (1, 0): ("red", "-", "your delete"),
             (0, 1): ("yellow", "--", "new: kept by rule"), (0, 0): ("orange", ":", "new: deleted by rule")}
    seen = set()
    for j, r in D.iterrows():
        key = (1 if r["ref"] else 0, int(r["keep"]))
        col, ls, lab = style[key]
        f = Sd[:, :, j]
        ax.contour(f, levels=[0.35 * f.max()], colors=[col], linestyles=ls, linewidths=1.0)
        if info_d[j] is not None:
            ax.text(info_d[j]["cx"] + 6, info_d[j]["cy"], str(r["id"]), color=col, fontsize=6)
        if key not in seen:
            ax.plot([], [], color=col, ls=ls, label=lab)
            seen.add(key)
    for m in missing:
        ax.plot(m["cx"], m["cy"], "x", color="magenta", ms=8, mew=2)
    if missing:
        ax.plot([], [], "x", color="magenta", label="you kept it, day run did not find it (rescued)")
    for m in merged:
        ax.plot(m["cx"], m["cy"], "+", color="cyan", ms=9, mew=2)
    if merged:
        ax.plot([], [], "+", color="cyan", label="you kept it, merged with another of yours into one day cell")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    n_user = int((D.ref != "").sum())
    ax.set_title("%s  whole day: %d candidates, %d kept  (%d by your decisions, %d new by rule)  |  %d of your keeps not found"
                 % (a.tag, K, int(D.keep.sum()), int(D[D.ref != ""].keep.sum()), int(D[D.ref == ""].keep.sum()),
                    len(missing)), fontsize=9)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, a.tag + "_decisions_map.png"), dpi=130)
    plt.close(fig)

    summ = dict(tag=a.tag, day_mat=a.day_mat, day_review=a.day_review, n_candidates=K,
                n_kept=int(D.keep.sum()), n_from_user=n_user,
                n_kept_from_user=int(D[D.ref != ""].keep.sum()),
                n_new=int((D.ref == "").sum()), n_new_kept=int(D[D.ref == ""].keep.sum()),
                registration=regs, user_keeps_not_found=missing, user_keeps_merged=merged, min_cos=a.min_cos)
    with open(os.path.join(a.out, a.tag + "_decisions.json"), "w") as fh:
        json.dump(summ, fh, indent=1, default=float)

    if not a.no_write:
        bk = a.day_review.replace("_cell_review.csv", "_cell_review_rules.csv")
        if not os.path.isfile(bk):
            shutil.copyfile(a.day_review, bk)
        R["keep"] = D["keep"].values
        R["note"] = D["source"].values
        R["transfer_ref"] = D["ref"].values
        R["transfer_ref_id"] = D["ref_id"].values
        R["transfer_cos"] = D["cos"].values
        R["decision_source"] = np.where(D["source"].str.startswith("your decision"), "user", "rule")
        R.to_csv(a.day_review, index=False)
    print("[transfer] %s: %d candidates -> %d kept (%d of yours matched, %d new kept by rule); "
          "%d of your keeps not found" % (a.tag, K, summ["n_kept"], summ["n_kept_from_user"],
                                         summ["n_new_kept"], len(missing)))


if __name__ == "__main__":
    main()
