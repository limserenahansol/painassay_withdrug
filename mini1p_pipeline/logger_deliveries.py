"""logger_deliveries.py - stimulus deliveries from the pressure logger ("Rogger"), Day 2 onwards.

The load cell sits on the stimulus tool. Three kinds of signal (Day 2, 2026-10-21, all 8 sessions):
  flat              the tool lies still (the first 3 min, the gaps between blocks)
  handling          the tool is held: slow wander of |corrected - baseline| up to ~1e5 counts
  press = delivery  a fast excursion of 5e5 counts up to ADC saturation (8.4e6), usually upward;
                    downward when the load cell is flipped (you, 10-21: "high down and up" counts
                    too); sometimes a long saturated bout - only its onset matters
The peak histogram of all excursions is bimodal with its trough at 2e5-5e5 counts, so
  delivery = a run of |corrected - running median| > --thr (default 3e5), runs closer than
             --merge-s (0.5 s) are one delivery (a press that bounces at saturation); an
             opposite-sign excursion starting within --rebound-s (0.6 s) after a press ends is its
             release rebound, not a new delivery
  onset    = the last sample before the crossing with |d| below --onset-level (1e5 = the
             handling ceiling), i.e. where the fast rise leaves the handling level
The camera frame of an onset = ttl_count at that sample (the camera frames taken so far, the one
sync rule: camera frame n = isx_frame n, K = 0); session time = frame / 30 Hz.
Each delivery gets the block it falls in (blocks of the session table, default 180-360, 390-570,
600-780, 810-990 s; up to --block-early-s before a block and --block-tol-s after it) and the
stimulus of that block.

  python logger_deliveries.py --root <Rogger folder> --table <day2_session_table.csv> --out <dir>
Writes <out>/<rid>_logger_deliveries.csv, <out>/<rid>_logger_qc.png (every block, onsets marked),
<out>/logger_delivery_counts.csv (deliveries per session x block) and logger_sessions.csv.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = 30.0
SCOL = {"Light touch": "#5dade2", "Mild touch": "#f5b041", "Pin prick": "#e74c3c", "Heat": "#8e44ad"}


def detect(x, thr, onset_level, merge_n, rebound_n, base_n):
    base = pd.Series(x).rolling(base_n, center=True, min_periods=base_n // 10).median().to_numpy()
    d = x - base
    a = np.abs(d)
    m = a > thr
    e = np.diff(np.r_[0, m.astype(np.int8), 0])
    runs = list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1) - 1))
    ev = []
    for i, j in runs:
        sg = int(np.sign(d[i + int(np.argmax(a[i:j + 1]))]))
        if ev and i - ev[-1]["i_end"] <= merge_n and sg == ev[-1]["sign"]:
            ev[-1]["i_end"] = j                                   # the same press, bounced
            continue
        if ev and i - ev[-1]["i_end"] <= rebound_n and sg != ev[-1]["sign"]:
            ev[-1]["i_end"] = j; ev[-1]["rebound"] = True          # release rebound
            continue
        k = i
        floor = ev[-1]["i_end"] if ev else 0
        while k > floor and a[k] > onset_level:
            k -= 1
        ev.append(dict(i_onset=k, i_cross=i, i_end=j, sign=sg, rebound=False))
    for v in ev:
        seg = a[v["i_cross"]:v["i_end"] + 1]
        v["peak"] = float(seg.max())
        v["saturated"] = bool(v["peak"] > 8.0e6)
    return ev, d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--table", required=True, help="csv: logger_session, mouse, phase, treatment, block1..block4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=3e5)
    ap.add_argument("--onset-level", type=float, default=1e5)
    ap.add_argument("--merge-s", type=float, default=0.5)
    ap.add_argument("--rebound-s", type=float, default=0.6)
    ap.add_argument("--block-tol-s", type=float, default=10.0, help="a delivery up to this long after a block end belongs to it")
    ap.add_argument("--block-early-s", type=float, default=15.0,
                    help="... and up to this long before a block start (a block started a little early: 1191 d2 post)")
    ap.add_argument("--blocks", default="180-360,390-570,600-780,810-990")
    ap.add_argument("--source", default="logger", choices=["logger", "offline"],
                    help="logger (default, your rule): the logger's own indicator onsets + the downward presses; "
                         "offline: only the offline detection (both directions, --thr)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    blocks = [tuple(float(v) for v in b.split("-")) for b in a.blocks.split(",")]
    T = pd.read_csv(a.table)
    counts, sess = [], []
    for _, r in T.iterrows():
        cands = [p for p in glob.glob(os.path.join(a.root, "**", f"{r.logger_session}_*_pressure.csv"), recursive=True)
                 if "__MACOSX" not in p]
        if len(cands) != 1:
            print(f"*** {r.logger_session}: {len(cands)} logger files"); continue
        p = cands[0]
        info = json.load(open(p.replace("_pressure.csv", "_info.json")))
        D = pd.read_csv(p)
        t, c, x = D.t_s.to_numpy(), D.ttl_count.to_numpy(), D.corrected.to_numpy(float)
        fs = (len(t) - 1) / (t[-1] - t[0])
        ev, d = detect(x, a.thr, a.onset_level, int(round(a.merge_s * fs)), int(round(a.rebound_s * fs)), int(round(10 * fs)) | 1)
        if a.source == "logger":
            # your rule (2026-10-21): every onset of the logger's own Schmitt indicator (lower/upper of the
            # info file, e.g. 25 / 90000) is a delivery, PLUS every large DOWNWARD press (load cell pushed the
            # other way), which the indicator cannot see; a downward press within 0.5 s of an indicator onset is
            # the same delivery and is not added twice
            ind = D.indicator.to_numpy().astype(int)
            e = np.diff(np.r_[0, ind, 0]); on, off = np.flatnonzero(e == 1), np.flatnonzero(e == -1) - 1
            near = int(round(0.5 * fs))
            sel = [dict(i_onset=int(i), i_cross=int(i), i_end=int(j), sign=1, rebound=False, src="logger indicator",
                        peak=float(np.max(np.abs(d[i:j + 1]))), saturated=bool(np.max(np.abs(d[i:j + 1])) > 8.0e6))
                   for i, j in zip(on, off)]
            for v in ev:
                if v["sign"] < 0 and not np.any(np.abs(on - v["i_onset"]) <= near):
                    v["src"] = "downward press (added)"
                    sel.append(v)
            ev = sorted(sel, key=lambda v: v["i_onset"])
        else:
            for v in ev:
                v["src"] = "offline"
        rid = f"{r.mouse}_d{int(r.day)}_{r.phase}"
        stim = [r[f"block{k}"] for k in range(1, 5)]
        rows = []
        for v in ev:
            f = int(c[v["i_onset"]])
            ts = f / FPS
            blk = next((k + 1 for k, (b0, b1) in enumerate(blocks) if b0 - a.block_early_s <= ts <= b1 + a.block_tol_s), 0)
            rows.append(dict(recording_id=rid, onset_cam_frame=f, onset_isx_frame=f, onset_session_s=round(ts, 3),
                             onset_logger_s=round(float(t[v["i_onset"]]), 4), end_session_s=round(int(c[v["i_end"]]) / FPS, 3),
                             peak_counts=int(v["peak"]), sign=v["sign"], saturated=v["saturated"], with_rebound=v["rebound"],
                             source=v["src"], block=blk, stimulus=stim[blk - 1] if blk else "outside blocks"))
        E = pd.DataFrame(rows)
        E.to_csv(os.path.join(a.out, f"{rid}_logger_deliveries.csv"), index=False)
        cnt = {"recording_id": rid, "logger_session": r.logger_session, "treatment": r.treatment}
        for k in range(1, 5):
            cnt[f"block{k} {stim[k - 1]}"] = int((E.block == k).sum()) if len(E) else 0
        cnt["outside blocks"] = int((E.block == 0).sum()) if len(E) else 0
        cnt["downward"] = int((E.sign < 0).sum()) if len(E) else 0
        cnt["from logger indicator"] = int((E.source == "logger indicator").sum()) if len(E) else 0
        cnt["downward added"] = int((E.source == "downward press (added)").sum()) if len(E) else 0
        counts.append(cnt)
        first = np.flatnonzero(c >= 1)
        sess.append(dict(recording_id=rid, logger_session=r.logger_session, trigger=info.get("trigger"), ttl_pulses=info.get("ttl_pulses"),
                         max_ttl_count=int(c.max()), logger_s_before_first_pulse=round(float(t[first[0]]), 3) if len(first) else np.nan,
                         count_steps_over_1=int((np.diff(np.unique(c)) > 1).sum()), rate_hz=round(fs, 2)))
        # QC: the no-stimulus start + each block, symlog, onsets marked in the block's stimulus colour
        ts_all = c / FPS
        fig, ax = plt.subplots(5, 1, figsize=(22, 14))
        for k, (b0, b1) in enumerate([(0.0, blocks[0][0])] + blocks):
            g = ax[k]
            mm = (ts_all >= b0 - 8) & (ts_all <= b1 + 12)
            g.plot(ts_all[mm], d[mm], lw=0.5, color="k")
            g.set_yscale("symlog", linthresh=2000); g.set_ylim(-1e7, 1e7)
            for yv in (a.thr, -a.thr):
                g.axhline(yv, color="0.6", lw=0.6, ls="--")
            sel = E[(E.onset_session_s >= b0 - 8) & (E.onset_session_s <= b1 + 12)] if len(E) else E
            col = "k" if k == 0 else SCOL.get(stim[k - 1], "k")
            for _, q in sel.iterrows():
                g.axvline(q.onset_session_s, color=col if q.block == k else "#ff00ff", lw=1.0, alpha=0.8)
            nb = int((E.block == k).sum()) if (len(E) and k) else int((E.block == 0).sum()) if len(E) else 0
            g.set_title(f"{rid} ({r.treatment})   " + ("no stimulus (deliveries here: %d)" % len([1 for _, q in sel.iterrows() if q.block == 0])
                        if k == 0 else f"block {k}: {stim[k - 1]}  -  {nb} deliveries"), loc="left", fontsize=11)
            g.set_xlim(b0 - 8, b1 + 12)
        ax[-1].set_xlabel("session time (s) = camera frame / 30")
        fig.suptitle(f"{r.logger_session}: |corrected - 10 s running median|, symlog; dashed = threshold {a.thr:.0e}; "
                     "vertical lines = delivery onsets (magenta = outside every block)", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"{rid}_logger_qc.png"), dpi=70); plt.close(fig)
        print(f"{rid}: {len(E)} deliveries " + ", ".join(f"{k}={v}" for k, v in cnt.items() if k.startswith("block") or k in ("outside blocks", "downward")))
    pd.DataFrame(counts).to_csv(os.path.join(a.out, "logger_delivery_counts.csv"), index=False)
    pd.DataFrame(sess).to_csv(os.path.join(a.out, "logger_sessions.csv"), index=False)
    overview(a.out, T, blocks)


def overview(out, T, blocks):
    """One row per session: the blocks (shaded, labelled) and every delivery onset as a tick."""
    fig, ax = plt.subplots(len(T), 1, figsize=(16, 1.05 * len(T) + 1.2), sharex=True)
    for k, (_, r) in enumerate(T.iterrows()):
        rid = f"{r.mouse}_d{int(r.day)}_{r.phase}"
        E = pd.read_csv(os.path.join(out, f"{rid}_logger_deliveries.csv"))
        g = ax[k]
        for b, (b0, b1) in enumerate(blocks):
            s = r[f"block{b + 1}"]
            g.axvspan(b0, b1, color=SCOL.get(s, "0.8"), alpha=0.18, lw=0)
            g.text((b0 + b1) / 2, 1.02, f"{s} ({int((E.block == b + 1).sum())})", ha="center", va="bottom", fontsize=8)
        for _, q in E.iterrows():
            col = SCOL.get(q.stimulus, "#ff00ff")
            g.plot([q.onset_session_s] * 2, [0.1, 0.9] if q.sign > 0 else [0.1, 0.6], color=col, lw=1.2)
        g.set_ylim(0, 1.25); g.set_yticks([])
        g.set_ylabel(f"{r.mouse} {r.phase}\n{r.treatment.split(' (')[0]}", rotation=0, ha="right", va="center", fontsize=9)
        for sp in ("top", "right", "left"):
            g.spines[sp].set_visible(False)
    ax[-1].set_xlim(0, blocks[-1][1] + 20)
    ax[-1].set_xlabel("session time (s) = camera frame / 30")
    fig.suptitle("Day 2 deliveries from the pressure logger: one tick per delivery onset (short tick = downward press; "
                 "magenta = outside the blocks); (n) = deliveries per block", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "logger_deliveries_overview.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
