"""R9  WHERE DO THE FAST 'DOWN' RESPONSES COME FROM? Straight from the motion-corrected movie, no
demixing: per cell the mean of the pixels inside its footprint (> 30 % of its peak) and of an
annulus around it (3-12 px out, every kept footprint removed), around each event; dF/F against
[-1.5, -0.5] s of the same trial, trial mean. Compared with the EXTRACT-based z of the same cell.
  raw ROI falls fast too         -> an optical change at the cell (focus / z, brightness)
  raw ROI rises with the annulus,
  only the EXTRACT trace falls   -> the demixing removes more background than the cell has
                                    (over-subtraction): not a real decrease
  raw ROI - 0.7 annulus          -> the classic neuropil-subtracted trace, for reference
Cells: DOWN and UP of the time-shift rule in that session / event, and 15 non-responsive cells.

  python ca_raw_roi_check.py --out-root <mini1p_out/20261019> --scoring-dir <final> --day1 <analysis/ca_day1>
      --out <analysis/ca_response_shape> --cases 5939:pre:Pin prick 5941:post:Light touch 1191:pre:Light touch
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
import ca_day1_analysis as d1  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = d1.FPS
K0, K1 = int(-3 * FPS), int(5 * FPS)
TT = np.arange(K0, K1) / FPS


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def masks(A, ids_all, sel):
    """ROI and annulus (movie orientation = footprint transposed) for the selected component ids"""
    allm = np.zeros(A.shape[:2][::-1], bool)
    for k in ids_all:
        f = A[:, :, k - 1].T
        allm |= f > 0.1 * f.max()
    allm = binary_dilation(allm, disk(2))
    out = {}
    for k in sel:
        f = A[:, :, k - 1].T
        roi = f > 0.3 * f.max()
        ann = binary_dilation(roi, disk(12)) & ~binary_dilation(roi, disk(3)) & ~allm
        out[k] = (roi, ann)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--scoring-dir", required=True)
    ap.add_argument("--day1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="+", default=["5939:pre:Pin prick", "5941:post:Light touch", "1191:pre:Light touch"])
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    ET = pd.read_csv(os.path.join(a.day1, f"ca_d{d1.DAY}_cells_event_tests.csv"))
    ET["mouse"] = ET.mouse.astype(str)
    rng = np.random.default_rng(0)
    rows, curves = [], {}
    Acache = {}
    for case in a.cases:
        m, ph, ev = case.split(":")
        tag = f"{m}_d{d1.DAY}"
        s = d1.load_session(a.out_root, m, ph, a.scoring_dir)
        if s is None or s["B"] is None:
            continue
        if m not in Acache:
            Acache.clear()
            Acache[m] = np.asarray(sio.loadmat(os.path.join(a.out_root, tag, "3_extract", f"{tag}_extract.mat"),
                                               variable_names=["spatial_weights"])["spatial_weights"], np.float32)
        A = Acache[m]
        ids = s["T"]["cell_ids"].astype(int)
        E = ET[(ET.mouse == m) & (ET.phase == ph) & (ET.event == ev)].set_index("cell")
        dn = [c for c in ids if c in E.index and E.at[c, "cls_shift"] == "DOWN"]
        up = [c for c in ids if c in E.index and E.at[c, "cls_shift"] == "UP"]
        no = [c for c in ids if c in E.index and E.at[c, "cls_shift"] == "no response"]
        no = list(rng.choice(no, min(15, len(no)), replace=False)) if no else []
        groups = {"DOWN": dn, "UP": up, "none": no}
        sel = dn + up + no
        if not sel:
            continue
        MK = masks(A, ids, sel)
        allpix = np.zeros_like(next(iter(MK.values()))[0])
        for roi, ann in MK.values():
            allpix |= binary_dilation(roi, disk(13))
        yy, xx = np.where(allpix)
        y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
        FT = pd.read_csv(os.path.join(a.out_root, tag, "1_movie", f"{tag}_mc_frames.csv"))
        FT = FT[(FT.recording_id == f"{tag}_{ph}") & FT.rec_h5_row.notna()]   # dropped camera frames: no h5 row
        row_of = dict(zip(FT.isx_frame.astype(int), FT.rec_h5_row.astype(int) - 1))
        X, tt, used = d1.peri(s, s["B"]["ev"].get(ev, []))
        fr = s["T"]["frame"].astype(int)
        # EXTRACT-based z of the same cells (trial mean, baseline [-1.5, -0.5])
        zt = np.nanmean(X, 0)
        zt = zt - np.nanmean(zt[(tt >= -1.5) & (tt < -0.5)], 0)
        col = {c: j for j, c in enumerate(ids)}
        roi_tr = {c: [] for c in sel}
        ann_tr = {c: [] for c in sel}
        with h5py.File(os.path.join(a.out_root, tag, "1_movie", f"{tag}_{ph}_mc.h5"), "r") as h:
            mov = h["mov"]
            for f in used:
                rws = [row_of.get(int(f) + k) for k in range(K0, K1)]
                if any(r is None for r in rws) or rws[-1] - rws[0] != K1 - K0 - 1:
                    continue
                blk = mov[rws[0]:rws[-1] + 1, y0:y1, x0:x1]                 # uint16, means per mask below
                for c in sel:
                    roi, ann = MK[c]
                    rm, am = roi[y0:y1, x0:x1], ann[y0:y1, x0:x1]
                    roi_tr[c].append(blk[:, rm].mean(1, dtype=np.float64))
                    ann_tr[c].append(blk[:, am].mean(1, dtype=np.float64) if am.any() else np.full(len(blk), np.nan))
        base = (TT >= -1.5) & (TT < -0.5)
        for g, cs in groups.items():
            for c in cs:
                if not roi_tr[c]:
                    continue
                R_ = np.vstack(roi_tr[c]); N_ = np.vstack(ann_tr[c])
                r0 = R_[:, base].mean(1, keepdims=True); n0 = N_[:, base].mean(1, keepdims=True)
                rd = ((R_ - r0) / r0).mean(0); nd = ((N_ - n0) / n0).mean(0)
                sub = R_ - 0.7 * N_
                s0 = sub[:, base].mean(1, keepdims=True)
                sd = ((sub - s0) / r0).mean(0)
                zc = zt[:, col[c]]
                # put the EXTRACT z on the time base of the movie windows
                zi = np.interp(TT, tt, zc)
                w1 = (TT >= 0.8) & (TT < 1.2)
                w0 = (TT >= 0.1) & (TT < 0.3)
                rows.append(dict(mouse=m, phase=ph, event=ev, group=g, cell=int(c), n_trials=len(roi_tr[c]),
                                 roi_dff_1s=float(rd[w1].mean()), annulus_dff_1s=float(nd[w1].mean()),
                                 roi_minus_ann_1s=float(sd[w1].mean()), extract_z_1s=float(zi[w1].mean()),
                                 roi_speed=float(rd[w0].mean() / rd[w1].mean()) if abs(rd[w1].mean()) > 1e-6 else np.nan,
                                 extract_speed=float(zi[w0].mean() / zi[w1].mean()) if abs(zi[w1].mean()) > 1e-6 else np.nan))
                curves.setdefault((case, g), []).append((rd, nd, sd, zi))
        print(f"{case}: DOWN {len(dn)}, UP {len(up)}, none {len(no)}; {len(used)} events")
    D = pd.DataFrame(rows)
    D.round(5).to_csv(os.path.join(a.out, "R9_raw_roi_check.csv"), index=False)
    cases = [c for c in a.cases if any(k[0] == c for k in curves)]
    fig, ax = plt.subplots(3, max(len(cases), 1), figsize=(5 * max(len(cases), 1), 9.5), squeeze=False, sharex=True)
    for j, case in enumerate(cases):
        for i, g in enumerate(("DOWN", "UP", "none")):
            a_ = ax[i, j]
            L = curves.get((case, g), [])
            if not L:
                a_.set_axis_off(); continue
            R_ = np.vstack([x[0] for x in L]); N_ = np.vstack([x[1] for x in L]); S_ = np.vstack([x[2] for x in L])
            Z_ = np.vstack([x[3] for x in L])
            a_.plot(TT, 100 * R_.mean(0), color="k", lw=1.4, label="raw ROI (footprint pixels)")
            a_.plot(TT, 100 * N_.mean(0), color="#e67e22", lw=1.2, label="annulus around it")
            a_.plot(TT, 100 * S_.mean(0), color="#8e44ad", lw=1.0, ls="--", label="ROI - 0.7 annulus")
            a2 = a_.twinx()
            a2.plot(TT, Z_.mean(0), color="#2471a3" if g == "DOWN" else "#c0392b", lw=1.0, alpha=0.8, label="EXTRACT z (right axis)")
            a2.tick_params(labelsize=7)
            a_.axvline(0, color="k", lw=0.6); a_.axhline(0, color="0.6", lw=0.5)
            a_.set_title(f"{case} - {g} cells (n={len(L)})", fontsize=9)
            if j == 0:
                a_.set_ylabel("dF/F (%) vs [-1.5,-0.5] s", fontsize=8)
            if i == 2:
                a_.set_xlabel("time from event (s)", fontsize=8)
            if i == 0 and j == 0:
                h1, l1 = a_.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
                a_.legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, loc="lower left")
    fig.suptitle("R9 raw movie pixels at the cells the analysis calls DOWN / UP (no demixing): footprint pixels, the annulus around "
                 "them, and the EXTRACT trace of the same cells", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(a.out, "R9_raw_roi_check.png"), dpi=130); plt.close(fig)
    if len(D):
        print(D.groupby(["event", "phase", "mouse", "group"])[["roi_dff_1s", "annulus_dff_1s", "roi_minus_ann_1s", "extract_z_1s",
                                                               "roi_speed", "extract_speed"]].median().round(4).to_string())
        print(D.groupby("group").apply(lambda X: pd.Series(dict(frac_roi_down=(X.roi_dff_1s < 0).mean(),
                                                                frac_ann_up=(X.annulus_dff_1s > 0).mean(),
                                                                frac_sub_down=(X.roi_minus_ann_1s < 0).mean()))).round(2).to_string())
    print("->", a.out)


if __name__ == "__main__":
    main()
