"""Analysis-ready Ca traces of one mouse-day, per recording (pre / post) and kept cell.

EXTRACT's least-squares traces (temporal_raw) are dF/F with ONE F0 per pixel = the mean of the
whole day movie (pre + post). Event-locked analyses with a local pre-event baseline are fine
with that, but whole-session comparisons are not: bleaching within a session and a dimmer
post session would look like a change in activity. So, per recording and cell:

  x          temporal_raw (dF/F relative to the day mean): F = F0_day (1 + x)
  b          slow baseline: running 8th percentile over 180 s (on 1-s bins, interpolated); slow
             enough not to ride on the activity of very active cells, fast enough for bleaching
  dff        local dF/F = (F - F_base) / F_base = (x - b) / (1 + b)   <- use this
  sn         noise of dff from its power spectrum (7.5-15 Hz, as CaImAn GetSn)
  z          dff / (1.4826 MAD of dff): robust z over the whole recording
  c, s       thresholded OASIS AR(1) deconvolution of dff (Friedrich et al. 2017, as CNMF-E /
             CaImAn): c = denoised calcium, s = event sizes. The decay g per cell is measured
             on its own ISOLATED transients (pass 1 with tau 1 s finds them; the mean decay
             from the peak, fitted where it is above 20 % of the peak, gives tau; fewer than 5
             isolated transients -> the median tau of the day's cells), limited to 0.15-3 s.
             (The autocovariance estimate is dominated by slow wander and hit 3 s.) An event must
             rise in ONE frame by >= s_min = 2.5 * sn * sqrt(1 + g^2) (about 3.5 noise SD of
             the frame-to-frame step), so slow wander (no Ca transient shape) gives no event.
  events     deconvolved events, those within 0.5 s merged into one (onset = first frame);
             amplitude = c at its peak within 1 s
The first 15 s of each recording are left out of events and statistics: the LED / room light
switch-on makes every trace decay there (bleaching onset), which is not neural activity.
Per cell: SNR (95th percentile of z), event rate per minute (whole session and every schedule
block), mean event amplitude, fraction of time active, mean local dF/F.

Outputs (out_dir):
  <recording_id>_traces.npz   frame (isx frame, 0-based), t_s, x, baseline, dff, z, sigma, cell_ids
  <recording_id>_events.csv   cell, onset_frame, peak_frame, onset_s, peak_s, amplitude_dff, peak_z
  <tag>_cell_summary.csv      one row per cell and recording
  <recording_id>_trace_qc.png examples: x with its baseline, dff with the detected events

  python ca_trace_outputs.py --extract <tag>_extract.mat --keep <tag>_cell_review.csv --out-dir DIR
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import percentile_filter
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)

FPS = 30.0
BLOCKS = [("baseline", 0, 180), ("stim1", 180, 360), ("stim2", 390, 570), ("stim3", 600, 780), ("stim4", 810, 990)]


def baseline(x, fps=FPS, win_s=180.0, q=8):
    """running q-th percentile over win_s, computed on 1-s bin means, linearly interpolated"""
    n = len(x)
    k = int(round(fps))
    nb = int(np.ceil(n / k))
    xp = np.r_[x, np.full(nb * k - n, np.nan)].reshape(nb, k)
    bm = np.nanmean(xp, 1)
    ok = np.isfinite(bm)
    bm = np.interp(np.arange(nb), np.where(ok)[0], bm[ok]) if ok.any() else np.zeros(nb)
    w = max(3, int(round(win_s)) | 1)
    bl = percentile_filter(bm, q, size=w, mode="nearest")
    return np.interp(np.arange(n) / k, np.arange(nb) + 0.5, bl)


SKIP_S = 15.0


def noise_psd(y):
    from scipy.signal import welch
    f, P = welch(y, fs=1.0, nperseg=min(256, len(y)))
    ind = (f > 0.25) & (f <= 0.5)
    return float(np.sqrt(np.exp(np.mean(np.log(P[ind] / 2 + 1e-30)))))


def estimate_g(y, sn, fps=FPS):
    y = y - y.mean()
    a1, a2 = np.dot(y[1:], y[:-1]) / len(y), np.dot(y[2:], y[:-2]) / len(y)
    g = a2 / a1 if a1 > 0 else 0.9
    lo, hi = np.exp(-1 / (0.15 * fps)), np.exp(-1 / (3.0 * fps))
    return float(np.clip(g, lo, hi))


def oasis_ar1(y, g, s_min):
    """thresholded OASIS for AR(1): c_t = g c_{t-1} + s_t, s_t = 0 or >= s_min, c >= 0"""
    T = len(y)
    v, w, t0, ln = [float(y[0])], [1.0], [0], [1]
    for t in range(1, T):
        v.append(float(y[t])); w.append(1.0); t0.append(t); ln.append(1)
        while len(v) > 1 and v[-2] * g ** ln[-2] + s_min > v[-1]:
            gl = g ** ln[-2]
            nw = w[-2] + gl * gl * w[-1]
            v[-2] = (w[-2] * v[-2] + gl * w[-1] * v[-1]) / nw
            w[-2] = nw
            ln[-2] += ln[-1]
            v.pop(); w.pop(); t0.pop(); ln.pop()
    c = np.empty(T)
    for vi, ti, li in zip(v, t0, ln):
        c[ti:ti + li] = max(vi, 0.0) * g ** np.arange(li)
    # c[0] is the calcium already present when the trace starts (the initial condition), not a
    # spike: counting it put one false event at the first frame (t = SKIP_S) of almost every cell
    s = np.r_[0.0, c[1:] - g * c[:-1]]
    s[s < 1e-12] = 0
    return c, s


TAU_G = {}                       # filled per cell by main (pass 1), used by pass 2


def fit_tau(segs, fps=FPS):
    from scipy.optimize import curve_fit
    m = np.mean(segs, 0)
    tt = np.arange(len(m)) / fps
    try:
        (A, tau, C), _ = curve_fit(lambda t, A, tau, C: A * np.exp(-t / tau) + C, tt, m, p0=(1.0, 0.8, 0.0),
                                   bounds=([0.2, 0.05, -0.5], [2.0, 6.0, 0.6]), maxfev=5000)
    except Exception:
        return None
    return float(np.clip(tau, 0.15, 3.0))


def tau_from_isolated(dff, sn, fps=FPS, g0=None, min_n=5, return_segs=False):
    """g from the mean decay of isolated transients (>= 3 s to the next one); None if < min_n"""
    g0 = np.exp(-1 / (1.0 * fps)) if g0 is None else g0
    c, sp = oasis_ar1(dff, g0, 2.5 * sn * np.sqrt(1 + g0 * g0))
    ev = events_from_s(sp, c)
    L = int(3 * fps)
    segs = []
    for i, (on, pk, amp, _) in enumerate(ev):
        nxt = ev[i + 1][0] if i + 1 < len(ev) else len(dff)
        if nxt - pk >= L and on >= 15:
            pre = np.median(dff[on - 15:on])          # the trace just before the onset
            h = dff[pk] - pre
            if h > 4 * sn:
                segs.append((dff[pk:pk + L] - pre) / h)
    if return_segs:
        return segs
    if len(segs) < min_n:
        return None
    tau = fit_tau(segs, fps)
    return None if tau is None else float(np.exp(-1 / (tau * fps)))


def events_from_s(s, c, fps=FPS, merge_s=0.5):
    idx = np.where(s > 0)[0]
    out = []
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] - idx[j] <= int(round(merge_s * fps)):
            j += 1
        on = idx[i]
        hi = min(len(c), idx[j] + int(fps))
        pk = on + int(np.argmax(c[on:hi]))
        out.append((on, pk, float(c[pk]), float(s[idx[i]:idx[j] + 1].sum())))
        i = j + 1
    return out


def events(z, dff, fps=FPS, thr=3.0, low=1.0, min_s=0.1, gap_s=1.0, peak_s=2.0):
    zs = pd.Series(z).rolling(max(1, int(round(0.2 * fps))), center=True, min_periods=1).mean().values
    above = zs > thr
    d = np.diff(np.r_[0, above.astype(int), 0])
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    out, last = [], -10 ** 9
    for a, b in zip(st, en):
        if b - a < int(round(min_s * fps)):
            continue
        on = a
        while on > 0 and zs[on - 1] >= low and a - on < int(2 * fps):
            on -= 1
        if on - last < int(round(gap_s * fps)):
            continue
        hi = min(len(z), a + int(round(peak_s * fps)))
        pk = a + int(np.nanargmax(dff[a:hi])) if hi > a else a
        out.append((on, pk, float(dff[pk] - dff[on]), float(z[pk])))
        last = on
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True, help="<tag>_extract.mat of extract_1p(_seeded)")
    ap.add_argument("--keep", default="", help="<tag>_cell_review.csv (column keep) - empty = every component")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-example", type=int, default=8)
    ap.add_argument("--bg-regress", default="",
                    help="folder with <rid>_background.npz (ca_background_signal.py): remove the background "
                         "signal from every cell by least squares before everything else (a second trace version)")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = sio.loadmat(a.extract, squeeze_me=True, struct_as_record=False)
    T = np.asarray(M["temporal_raw"], float)                    # rows = raw frames of the day table
    ft = M["frame_table"]
    FT = pd.DataFrame({k: np.atleast_1d(getattr(ft, k)) for k in ft._fieldnames})
    FT["recording_id"] = FT.recording_id.astype(str)
    tag = os.path.basename(a.extract).replace("_extract.mat", "")
    C = T.shape[1]
    ids = np.arange(1, C + 1)
    if a.keep and os.path.isfile(a.keep):
        R = pd.read_csv(a.keep)
        ids = R.id[R.keep == 1].values.astype(int)
    bg_info = []
    if a.bg_regress:
        # x_j(t) - beta_j * b(t): b = the background's fractional change (outside every footprint),
        # beta_j fitted per cell and recording on its valid frames after the switch-on (with intercept)
        for rid in FT.recording_id.unique():
            f = os.path.join(a.bg_regress, f"{rid}_background.npz")
            if not os.path.isfile(f):
                print(f"  no background signal for {rid} - not corrected"); continue
            Bz = np.load(f)
            sel = np.where((FT.recording_id == rid).values)[0]
            bmap = dict(zip(Bz["frame"].astype(int), Bz["b_frac"].astype(float)))
            bvec = np.array([bmap.get(int(fr_), np.nan) for fr_ in FT.isx_frame.values[sel]])
            t_ = FT.isx_frame.values[sel] / FPS
            for c in ids:
                x = T[sel, c - 1]
                ok = np.isfinite(x) & np.isfinite(bvec) & (t_ >= SKIP_S)
                if ok.sum() < 100:
                    continue
                A = np.c_[np.ones(ok.sum()), bvec[ok]]
                coef = np.linalg.lstsq(A, x[ok], rcond=None)[0]
                r_before = np.corrcoef(x[ok], bvec[ok])[0, 1]
                fin = np.isfinite(bvec)
                T[sel[fin], c - 1] = x[fin] - coef[1] * bvec[fin]
                bg_info.append(dict(recording_id=rid, cell=int(c), beta=coef[1], r_with_background=r_before))
        if bg_info:
            Bi = pd.DataFrame(bg_info)
            Bi.round(5).to_csv(os.path.join(a.out_dir, f"{tag}_background_regression.csv"), index=False)
            print(f"background regressed out of {Bi.cell.nunique()} cells: median r with background "
                  f"{Bi.r_with_background.median():.2f}, median beta {Bi.beta.median():.3f}")
    # pass 1: each cell's decay from its isolated transients over the whole day
    gs, pooled = {}, []
    for j, cid in enumerate(ids):
        x = T[:, cid - 1]
        ok = np.isfinite(x)
        if ok.sum() < 60 * FPS:
            continue
        xs = x[ok][int(SKIP_S * FPS):]
        b = baseline(xs)
        dff = (xs - b) / np.maximum(1 + b, 1e-3)
        segs = tau_from_isolated(dff, noise_psd(dff), return_segs=True)
        pooled += segs
        if len(segs) >= 5:
            t_ = fit_tau(segs)
            if t_ is not None:
                gs[int(cid)] = float(np.exp(-1 / (t_ * FPS)))
    tp = fit_tau(pooled) if len(pooled) >= 5 else None
    gmed = float(np.exp(-1 / ((tp if tp else 1.0) * FPS)))
    print(f"pooled decay of {len(pooled)} isolated transients of all cells: tau {tp if tp else float('nan'):.2f} s")
    for cid in ids:
        TAU_G[int(cid)] = gs.get(int(cid), gmed)
    taus = -1 / (FPS * np.log(np.array(list(gs.values())))) if gs else np.array([])
    print(f"decay from isolated transients: {len(gs)} of {len(ids)} cells, tau median "
          f"{np.median(taus) if len(taus) else np.nan:.2f} s (IQR {np.percentile(taus, 25) if len(taus) else np.nan:.2f}-"
          f"{np.percentile(taus, 75) if len(taus) else np.nan:.2f}); the others use the median")
    rows = []
    for rid in FT.recording_id.unique():
        sel = (FT.recording_id == rid).values
        fr = FT.isx_frame.values[sel].astype(int)
        valid = FT.valid.values[sel].astype(bool)
        X = T[sel][:, ids - 1]
        n, k = X.shape
        B, D, Z = np.full_like(X, np.nan), np.full_like(X, np.nan), np.full_like(X, np.nan)
        Cc, Ss = np.zeros_like(X), np.zeros_like(X)
        sig, gg = np.full(k, np.nan), np.full(k, np.nan)
        ev_rows = []
        t_s = fr / FPS
        for j in range(k):
            x = X[:, j].copy()
            okv = valid & np.isfinite(x)
            if okv.sum() < 10 * FPS:
                continue
            xi = np.interp(np.arange(n), np.where(okv)[0], x[okv])   # dropped frames bridged for the filters only
            st = int(SKIP_S * FPS)
            b = np.empty(n)
            b[st:] = baseline(xi[st:])          # the switch-on decay must not lift the baseline
            b[:st] = b[st]
            dff = (xi - b) / np.maximum(1 + b, 1e-3)
            sn = noise_psd(dff[st:])
            mad = 1.4826 * np.median(np.abs(dff[st:] - np.median(dff[st:])))
            g = TAU_G.get(int(ids[j]))
            if g is None:
                g = tau_from_isolated(dff[st:], sn)
            c, sp = oasis_ar1(dff[st:], g, 2.5 * sn * np.sqrt(1 + g * g))
            z = dff / max(mad, 1e-9)
            if not (mad > 1e-6 and sn > 1e-6):     # a flat trace (e.g. a footprint outside the lens): no z
                print(f"  {rid} cell {int(ids[j])}: flat trace (noise {sn:.2g}) - z left empty")
                z[:] = np.nan
            dff[~okv] = np.nan; z[~okv] = np.nan; dff[:st] = np.nan; z[:st] = np.nan
            B[:, j], D[:, j], Z[:, j], sig[j], gg[j] = b, dff, z, sn, g
            Cc[st:, j], Ss[st:, j] = c, sp
            for on, pk, amp, ssum in events_from_s(sp, c):
                on, pk = on + st, pk + st
                if not okv[on]:
                    continue
                ev_rows.append(dict(cell=int(ids[j]), onset_frame=int(fr[on]), peak_frame=int(fr[pk]), onset_s=t_s[on],
                                    peak_s=t_s[pk], amplitude_dff=round(amp, 5), peak_z=round(amp / max(mad, 1e-9), 2)))
        E = pd.DataFrame(ev_rows, columns=["cell", "onset_frame", "peak_frame", "onset_s", "peak_s", "amplitude_dff", "peak_z"])
        E.to_csv(os.path.join(a.out_dir, f"{rid}_events.csv"), index=False)
        np.savez_compressed(os.path.join(a.out_dir, f"{rid}_traces.npz"), frame=fr, t_s=t_s, valid=valid,
                            x=X.astype(np.float32), baseline=B.astype(np.float32), dff=D.astype(np.float32),
                            z=Z.astype(np.float32), c=Cc.astype(np.float32), s=Ss.astype(np.float32),
                            sn=sig, g=gg, tau_s=-1 / (FPS * np.log(gg)), cell_ids=ids, skip_s=SKIP_S)
        dur_min = (valid & (t_s >= SKIP_S)).sum() / FPS / 60
        for j, c in enumerate(ids):
            e = E[E.cell == c]
            r = dict(tag=tag, recording_id=rid, cell=int(c), sn_dff=sig[j], tau_s=-1 / (FPS * np.log(gg[j])),
                     snr_p95_z=np.nanpercentile(Z[:, j], 95),
                     events_per_min=len(e) / dur_min, mean_amp_dff=e.amplitude_dff.mean() if len(e) else np.nan,
                     deconv_per_min=np.nansum(Ss[:, j]) / dur_min,
                     frac_time_z_gt2=np.nanmean(Z[:, j] > 2), mean_dff=np.nanmean(D[:, j]),
                     baseline_drift=(np.nanmean(B[-int(60 * FPS):, j]) - np.nanmean(B[:int(60 * FPS), j])))
            for bn, s0, s1 in BLOCKS:
                s0 = max(s0, SKIP_S)
                m = (t_s >= s0) & (t_s < s1)
                if m.sum() > 10 * FPS:
                    r[f"rate_{bn}"] = ((e.onset_s >= s0) & (e.onset_s < s1)).sum() / (m.sum() / FPS / 60)
            rows.append(r)
        # QC figure: the cells with most events, the median one and the quietest
        S = pd.DataFrame([r for r in rows if r["recording_id"] == rid]).sort_values("events_per_min", ascending=False)
        pick = list(S.cell.values[:a.n_example - 2]) + list(S.cell.values[[len(S) // 2, -1]]) if len(S) > a.n_example else list(S.cell.values)
        fig, ax = plt.subplots(len(pick), 1, figsize=(18, 1.6 * len(pick) + 1), sharex=True)
        ax = np.atleast_1d(ax)
        for i, c in enumerate(pick):
            j = int(np.where(ids == c)[0][0])
            ax[i].plot(t_s, X[:, j], color="0.6", lw=0.5, label="x (dF/F, day F0)")
            ax[i].plot(t_s, B[:, j], color="tab:orange", lw=1, label="baseline (8th pct, 180 s)")
            ax2 = ax[i].twinx()
            ax2.plot(t_s, D[:, j], color="k", lw=0.5, label="local dF/F")
            ax2.plot(t_s, Cc[:, j], color="tab:cyan", lw=0.9, label="deconvolved calcium (OASIS)")
            e = E[E.cell == c]
            ax2.plot(e.onset_s, np.interp(e.peak_s, t_s, np.nan_to_num(D[:, j])) + 0.1 * np.nanmax(np.abs(D[:, j])), "v",
                     color="tab:red", ms=4, label="event onset")
            ax[i].axvspan(0, SKIP_S, color="0.9", lw=0)
            ax[i].set_ylabel(f"#{c}", rotation=0, labelpad=18, fontsize=9)
            ax[i].text(0.995, 0.85, f"{len(e)} events, {len(e) / dur_min:.1f}/min, noise {sig[j]:.4f}, tau {-1 / (FPS * np.log(gg[j])):.2f} s",
                       transform=ax[i].transAxes, ha="right", fontsize=7)
            if i == 0:
                ax[i].legend(loc="upper left", fontsize=7, frameon=False); ax2.legend(loc="upper center", fontsize=7, frameon=False)
        ax[-1].set_xlabel("time in recording (s)")
        fig.suptitle(f"{rid}: trace QC - grey = EXTRACT trace (day F0) + slow baseline (orange); black = local dF/F; cyan = "
                     f"OASIS calcium; red = event onsets; first {SKIP_S:.0f} s (grey band) excluded ({len(ids)} kept cells)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, f"{rid}_trace_qc.png"), dpi=100)
        plt.close(fig)
        print(f"{rid}: {len(ids)} cells, {n} frames, {len(E)} transients "
              f"({len(E) / max(len(ids), 1) / dur_min:.2f} per cell per min); median noise {np.nanmedian(sig):.4f}, "
              f"median tau {np.nanmedian(-1 / (FPS * np.log(gg))):.2f} s")
    pd.DataFrame(rows).round(5).to_csv(os.path.join(a.out_dir, f"{tag}_cell_summary.csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
