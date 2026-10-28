"""One QC sheet per recording of a mouse-day: every check of the Ca pipeline on one page.

  row 1  motion correction: shifts over the whole session (x, y), frame correlation with the
         template before / after, the template vs corrected mean image, the lens crop
  row 2  filtering / detection images (raw and corrected mean, temporal SD, filtered max, Cn,
         Cn x PNR - qc_stage_media) and the cells used (green) on the max dF/F and mean image
  row 3  decision transfer map (your decisions / rules) and the trace QC (EXTRACT trace, baseline,
         local dF/F, deconvolution, events)
  numbers  max / p95 shift, median correlation after MC, kept cells, median noise, decay tau,
           transients per cell per min, camera vs Inscopix frame counts
Writes <day-root>/5_analysis/<rid>_QC_sheet.png.

  python ca_qc_sheet.py --day-root <mini1p_out/20261019/5941_d1>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)


def img(ax, f, title):
    ax.axis("off")
    if f and os.path.isfile(f):
        ax.imshow(plt.imread(f))
        ax.set_title(title, fontsize=9)
    else:
        ax.text(0.5, 0.5, f"missing:\n{os.path.basename(f) if f else title}", ha="center", va="center", fontsize=8,
                color="red", transform=ax.transAxes)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-root", required=True)
    a = ap.parse_args(argv)
    tag = os.path.basename(os.path.normpath(a.day_root))
    mv, qc, an = (os.path.join(a.day_root, d) for d in ("1_movie", "2_qc", "5_analysis"))
    dj = os.path.join(mv, f"{tag}_inscopix.json")
    js = json.load(open(dj)) if os.path.isfile(dj) else {}
    recs = js.get("recordings", [])
    recs = recs if isinstance(recs, list) else [recs]
    mcj = {}
    for f in glob.glob(os.path.join(mv, "*_mc.json")) + glob.glob(os.path.join(mv, f"{tag}*mc*.json")):
        try:
            j = json.load(open(f))
            rr = j.get("recordings", [])
            for r in (rr if isinstance(rr, list) else [rr]):
                mcj[r["recording_id"]] = (j, r)
        except Exception:
            pass
    summ = pd.read_csv(os.path.join(an, "traces", f"{tag}_cell_summary.csv")) if os.path.isfile(
        os.path.join(an, "traces", f"{tag}_cell_summary.csv")) else pd.DataFrame()
    for rid in sorted({r.get("recording_id") for r in recs} | set(mcj)):
        if not rid:
            continue
        fig = plt.figure(figsize=(24, 17))
        gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 1.1, 1.2], hspace=0.12, wspace=0.04,
                              left=0.03, right=0.99, top=0.93, bottom=0.01)
        # shifts
        sh = os.path.join(mv, f"{rid}_mc_shifts.csv")
        ax = fig.add_subplot(gs[0, 0])
        if os.path.isfile(sh):
            S = pd.read_csv(sh)
            t = S.isx_frame / 30.0
            ax.plot(t, S.x_shift_px, lw=0.5, label="x"); ax.plot(t, S.y_shift_px, lw=0.5, label="y")
            ax.set_xlabel("time (s)"); ax.set_ylabel("rigid + piecewise shift (px)")
            ax.legend(fontsize=8, frameon=False)
            ax.set_title("motion correction: shift per frame", fontsize=9)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(t, S.corr_before, lw=0.4, color="0.6", label="before MC")
            ax2.plot(t, S.corr_after, lw=0.4, color="tab:green", label="after MC")
            ax2.set_xlabel("time (s)"); ax2.set_ylabel("frame vs template correlation")
            ax2.legend(fontsize=8, frameon=False)
            ax2.set_title("registration quality over the session", fontsize=9)
        else:
            ax.text(0.5, 0.5, "no shifts csv", ha="center"); ax.axis("off")
        img(fig.add_subplot(gs[0, 2]), os.path.join(mv, f"{rid}_mc_qc.png") if os.path.isfile(os.path.join(mv, f"{rid}_mc_qc.png"))
            else (glob.glob(os.path.join(mv, f"{tag}*mc_qc.png")) or [""])[0], "template vs corrected mean")
        img(fig.add_subplot(gs[0, 3]), (glob.glob(os.path.join(mv, f"*{rid}*crop_qc.png")) + glob.glob(os.path.join(mv, f"{tag}*crop_qc.png")) or [""])[0],
            "lens crop (kept area)")
        img(fig.add_subplot(gs[1, 0:2]), os.path.join(qc, f"{rid}_qc_stages.png"), "filtering and detection images")
        img(fig.add_subplot(gs[1, 2:4]), os.path.join(an, "figures", f"{tag}_cellmap.png"), "cells used (green) / rejected (red)")
        img(fig.add_subplot(gs[2, 0:2]), os.path.join(an, "decisions", f"{tag}_decisions_map.png"), "decision transfer")
        img(fig.add_subplot(gs[2, 2:4]), os.path.join(an, "traces", f"{rid}_trace_qc.png"), "trace QC")
        lines = [f"{rid}"]
        if rid in mcj:
            st = mcj[rid][1].get("stats", {})
            lines.append(f"shift max {st.get('abs_shift_max', np.nan):.1f} px, p95 {st.get('abs_shift_p95', np.nan):.1f} px, "
                         f"frame-to-frame p95 {st.get('frame_to_frame_p95', np.nan):.2f} px; corr after MC median "
                         f"{st.get('corr_after_median', np.nan):.3f} (p05 {st.get('corr_after_p05', np.nan):.3f}); "
                         f"mean image vs template r {st.get('mean_image_corr_with_template', np.nan):.4f}")
        rr = [r for r in recs if r.get("recording_id") == rid]
        if rr:
            v = rr[0].get("lossless_verification", {})
            lines.append(f"lossless copy: {v.get('n_frames_compared', '?')} frames compared, max |diff| {v.get('max_abs_diff', '?')}, "
                         f"passed {v.get('passed', '?')}")
        if len(summ):
            s = summ[summ.recording_id == rid]
            if len(s):
                lines.append(f"{len(s)} kept cells: noise (dF/F SD) median {s.sn_dff.median():.4f}, decay tau median "
                             f"{s.tau_s.median():.2f} s, transients {s.events_per_min.median():.2f} / cell / min (median), "
                             f"SNR (p95 z) median {s.snr_p95_z.median():.1f}")
        dp = os.path.join(an, "decisions", f"{tag}_duplicate_pairs.csv")
        if os.path.isfile(dp):
            nd = len(pd.read_csv(dp))
            dj = os.path.join(an, "decisions", f"{tag}_decisions.json")
            extra = ""
            if os.path.isfile(dj):
                jj = json.load(open(dj))
                extra = (f"; decisions: {jj.get('n_kept_from_user', '?')} kept by yours, {jj.get('n_new_kept', '?')} new by rule, "
                         f"{len(jj.get('user_keeps_merged', []))} of yours merged into one day cell")
            lines.append(f"possible duplicate pairs (same neuron twice): {nd}{extra}")
        fig.suptitle("   |   ".join(lines[:1]) + "\n" + "\n".join(lines[1:]), fontsize=11, y=0.995)
        out = os.path.join(an, f"{rid}_QC_sheet.png")
        os.makedirs(an, exist_ok=True)
        fig.savefig(out, dpi=95); plt.close(fig)
        print("QC sheet ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
