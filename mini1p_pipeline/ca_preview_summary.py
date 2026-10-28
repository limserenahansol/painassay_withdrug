"""Compare the detection previews (first 1 or 3 min) of all recordings (RUN_ca_preview3min_20261019.m):
noise, signal, motion and how many candidates / cells the recipe finds, side by side, plus a
sheet of every recording's cell map. A recording that stands out (noisier, dimmer, more
motion, far fewer or more candidates) is where the recipe may need adjusting.

  python ca_preview_summary.py --dir <...>/mini1p_out/20261019_preview3min
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, "*", "*_preview.json"))):
        j = json.load(open(f))
        rows.append(j)
    if not rows:
        print("no previews yet"); return 1
    T = pd.DataFrame(rows)
    T["mouse"] = T.recording_id.str.split("_").str[0]
    T["session"] = T.recording_id.str.split("_").str[-1].map({"pre": "baseline", "post": "injection"})
    T = T.sort_values(["mouse", "session"]).reset_index(drop=True)
    cols = ["recording_id", "brightness_median_counts", "noise_std_dff_median", "Cn_p99", "PNR_p99", "max_shift_px",
            "crop_fraction_kept", "n_pass1", "n_seeds", "n_candidates", "n_cell", "n_check", "n_not_cell",
            "snr_median_kept", "event_r_median_kept", "minutes"]
    T[[c for c in cols if c in T]].round(4).to_csv(os.path.join(a.dir, "preview_summary.csv"), index=False)
    lab = [f"{r.mouse}\n{r.session}\n{r.frames / 1800:.0f} min" for r in T.itertuples()]
    x = np.arange(len(T))
    panels = [("brightness_median_counts", "brightness in the lens (counts)", "signal level"),
              ("noise_std_dff_median", "noise: median pixel SD of dF/F (high-passed)", "noise"),
              ("PNR_p99", "peak-to-noise, 99th pct of pixels", "signal / noise"),
              ("Cn_p99", "local correlation Cn, 99th pct", "cell-like correlation"),
              ("max_shift_px", "largest motion shift (px)", "motion"),
              ("snr_median_kept", "SNR of the kept candidates (median)", "trace quality")]
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.5))
    ax = ax.ravel()
    for k, (c, t, short) in enumerate(panels):
        col = ["#c0392b" if s == "injection" else "#7f8c8d" for s in T.session]
        ax[k].bar(x, T[c], color=col)
        med = np.median(T[c])
        ax[k].axhline(med, color="k", ls=":", lw=1)
        for i, v in enumerate(T[c]):
            if med > 0 and (v > 1.5 * med or v < med / 1.5):
                ax[k].text(i, v, "!", ha="center", va="bottom", color="#e67e22", fontsize=14, fontweight="bold")
        ax[k].set_xticks(x, lab, fontsize=7); ax[k].set_title(t, fontsize=9)
    ax[6].bar(x, T.n_cell, color="#27ae60", label="cell")
    ax[6].bar(x, T.n_check, bottom=T.n_cell, color="#f1c40f", label="check")
    ax[6].bar(x, T.n_not_cell, bottom=T.n_cell + T.n_check, color="#e74c3c", label="not a cell")
    ax[6].set_xticks(x, lab, fontsize=7); ax[6].set_title("candidates per recording (rule verdicts)", fontsize=9)
    ax[6].legend(fontsize=7, frameon=False)
    ax[7].bar(x - 0.2, T.n_pass1, 0.4, label="EXTRACT pass 1"); ax[7].bar(x + 0.2, T.n_seeds, 0.4, label="added Cn x PNR seeds")
    ax[7].set_xticks(x, lab, fontsize=7); ax[7].set_title("how the candidates were found", fontsize=9)
    ax[7].legend(fontsize=7, frameon=False)
    fig.suptitle("Detection preview, first minutes of every recording (length under each bar), same recipe (grey baseline, red injection; "
                 "! = more than 1.5x away from the median of the recordings)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, "preview_comparison.png"), dpi=110); plt.close(fig)
    # sheet of the cell maps
    import cv2
    tiles = []
    for r in T.itertuples():
        im = cv2.imread(r.cells_map) if isinstance(r.cells_map, str) and os.path.isfile(r.cells_map) else None
        if im is None:
            continue
        h = 520
        im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        cv2.putText(im, f"{r.recording_id}: {r.n_candidates} candidates ({r.n_cell} cell, {r.n_check} check, {r.n_not_cell} not)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        tiles.append(im)
    if tiles:
        w = max(t.shape[1] for t in tiles)
        tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
        rows_ = [np.hstack(tiles[i:i + 2] + ([np.full_like(tiles[0], 255)] if len(tiles[i:i + 2]) == 1 else [])) for i in range(0, len(tiles), 2)]
        cv2.imwrite(os.path.join(a.dir, "preview_cell_maps.png"), np.vstack(rows_))
    print(T[[c for c in cols if c in T]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
