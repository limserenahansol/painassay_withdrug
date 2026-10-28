"""A second, demixing-free trace version: footprint pixels minus the LOCAL neuropil, straight from
the motion-corrected movie (an independent check of the EXTRACT traces; R9 showed a field-wide
brightness change at the deliveries that EXTRACT removes differently from cell to cell).

Per kept cell (same ids as the EXTRACT run):
  core     footprint > 30 % of its peak, minus pixels shared with another kept core (if >= 10 px remain)
  annulus  4-15 px around the core, every kept footprint (> 10 % of its peak, +2 px) removed (>= 50 px,
           else 4-22 px)
  F_core, F_ann = pixel means per frame (both sessions of the day, streamed in 100-frame blocks)
  beta     slope of F_core on F_ann on the high-passed signals (60-s running mean removed), refitted
           on the quieter half of the frames (cell's own transients left out), clipped to [0, 1.5]
  x        F_corr = F_core - beta (F_ann - median F_ann); x = F_corr / median(F_corr) - 1 (day F0)
then exactly the processing of ca_trace_outputs.py: local baseline (8th pct, 180 s), dF/F, robust z,
thresholded OASIS AR(1) with the cell's decay from the EXTRACT trace files, events.
Writes <tag>/5_analysis/traces_roi/<rid>_traces.npz + _events.csv (+ _roi_info.csv), the same format,
so every analysis runs on it with --traces traces_roi.

  python ca_roi_traces.py --day-root <mini1p_out/20261019/5941_d1>
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from scipy.ndimage import binary_dilation, uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
import ca_trace_outputs as cto  # noqa: E402

FPS = cto.FPS
SKIP_S = cto.SKIP_S


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def build_masks(A, ids):
    H, W = A.shape[1], A.shape[0]                       # movie frame = footprint transposed
    fp = {k: A[:, :, k - 1].T for k in ids}
    core = {k: f > 0.3 * f.max() for k, f in fp.items()}
    cnt = np.zeros((H, W), np.int16)
    for c in core.values():
        cnt += c
    allm = np.zeros((H, W), bool)
    for f in fp.values():
        allm |= f > 0.1 * f.max()
    allm = binary_dilation(allm, disk(2))
    out = {}
    for k in ids:
        c = core[k] & (cnt == 1)
        if c.sum() < 10:
            c = core[k]
        ann = binary_dilation(core[k], disk(15)) & ~binary_dilation(core[k], disk(4)) & ~allm
        if ann.sum() < 50:
            ann = binary_dilation(core[k], disk(22)) & ~binary_dilation(core[k], disk(4)) & ~allm
        out[k] = (c, ann)
    return out, H, W


def weights(masks, ids, H, W, which):
    rows, cols, vals = [], [], []
    for j, k in enumerate(ids):
        m = masks[k][which].ravel()
        idx = np.where(m)[0]
        if len(idx):
            rows.append(idx); cols.append(np.full(len(idx), j)); vals.append(np.full(len(idx), 1.0 / len(idx)))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(H * W, len(ids)))


def fill_nan(v):
    """frames without a value (the camera dropped them) filled by interpolation, so the running filter does not spread NaN"""
    v = np.asarray(v, float).copy()
    bad = ~np.isfinite(v)
    if bad.any() and (~bad).sum() > 1:
        idx = np.arange(len(v))
        v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    return v


def fit_beta(fc, fa, ok):
    hp = lambda v: fill_nan(v) - uniform_filter1d(fill_nan(v), int(60 * FPS), mode="nearest")
    y, x = hp(fc)[ok], hp(fa)[ok]
    if np.std(x) <= 0:
        return 0.0
    b = np.polyfit(x, y, 1)[0]
    res = y - b * x
    q = res <= np.median(res)
    if q.sum() > 100 and np.std(x[q]) > 0:
        b = np.polyfit(x[q], y[q], 1)[0]
    return float(np.clip(b, 0.0, 1.5))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-root", required=True)
    ap.add_argument("--block", type=int, default=100)
    a = ap.parse_args(argv)
    R = a.day_root
    tag = os.path.basename(os.path.normpath(R))
    out = os.path.join(R, "5_analysis", "traces_roi")
    os.makedirs(out, exist_ok=True)
    rv = pd.read_csv(os.path.join(R, "4_cell_review", f"{tag}_cell_review.csv"))
    ids = np.sort(rv.loc[rv.keep == 1, "id"].astype(int).values)
    M = sio.loadmat(os.path.join(R, "3_extract", f"{tag}_extract.mat"), squeeze_me=True, struct_as_record=False,
                    variable_names=["spatial_weights", "frame_table"])
    A = np.asarray(M["spatial_weights"], np.float32)
    ft = M["frame_table"]
    FT = pd.DataFrame({k: np.atleast_1d(getattr(ft, k)) for k in ft._fieldnames})
    FT["recording_id"] = FT.recording_id.astype(str)
    masks, H, W = build_masks(A, ids)
    del A
    Wc, Wa = weights(masks, ids, H, W, 0), weights(masks, ids, H, W, 1)
    MF = pd.read_csv(os.path.join(R, "1_movie", f"{tag}_mc_frames.csv"))
    raw = {}
    for rid in FT.recording_id.unique():
        ph = rid.split("_")[-1]
        sel = FT[FT.recording_id == rid]
        mf = MF[(MF.recording_id == rid) & MF.rec_h5_row.notna()]       # frames the camera dropped have no h5 row
        row_of = dict(zip(mf.isx_frame.astype(int), mf.rec_h5_row.astype(int) - 1))
        rows = np.array([row_of.get(int(f), -1) for f in sel.isx_frame], int)
        with h5py.File(os.path.join(R, "1_movie", f"{tag}_{ph}_mc.h5"), "r") as h:
            mov = h["mov"]
            n = mov.shape[0]
            Fc = np.full((n, len(ids)), np.nan, np.float32)
            Fa = np.full((n, len(ids)), np.nan, np.float32)
            for t0 in range(0, n, a.block):
                blk = mov[t0:t0 + a.block].reshape(-1, H * W).astype(np.float32)
                Fc[t0:t0 + len(blk)] = (Wc.T @ blk.T).T
                Fa[t0:t0 + len(blk)] = (Wa.T @ blk.T).T
                if t0 % (100 * a.block) == 0:
                    print(f"  {rid}: frame {t0}/{n}", flush=True)
        okr = rows >= 0
        fc = np.full((len(sel), len(ids)), np.nan, np.float32); fa = fc.copy()
        fc[okr], fa[okr] = Fc[rows[okr]], Fa[rows[okr]]
        raw[rid] = (sel, fc, fa)
    # beta and F0 per cell over the whole day
    rids = list(raw)
    t_all = np.concatenate([raw[r][0].isx_frame.values / FPS for r in rids])
    v_all = np.concatenate([raw[r][0].valid.values.astype(bool) for r in rids])
    FC = np.vstack([raw[r][1] for r in rids]); FA = np.vstack([raw[r][2] for r in rids])
    ok = v_all & (t_all >= SKIP_S) & np.isfinite(FC).all(1) & np.isfinite(FA).all(1)
    beta = np.array([fit_beta(FC[:, j].astype(float), FA[:, j].astype(float), ok) for j in range(len(ids))])
    medA = np.nanmedian(FA[ok], 0)
    FCOR = FC - beta * (FA - medA)
    F0 = np.nanmedian(FCOR[ok], 0)
    info = pd.DataFrame(dict(cell=ids, beta=beta, F0=F0, core_px=[masks[k][0].sum() for k in ids],
                             annulus_px=[masks[k][1].sum() for k in ids]))
    info.round(5).to_csv(os.path.join(out, f"{tag}_roi_info.csv"), index=False)
    print(f"{tag}: {len(ids)} cells, neuropil beta median {np.median(beta):.2f} (IQR {np.percentile(beta, 25):.2f}-"
          f"{np.percentile(beta, 75):.2f}), core px median {int(info.core_px.median())}")
    # decays from the EXTRACT-based trace files (same cells)
    gmap = {}
    for rid in rids:
        f = os.path.join(R, "5_analysis", "traces", f"{rid}_traces.npz")
        if os.path.isfile(f):
            Z = np.load(f)
            gmap.update(dict(zip(Z["cell_ids"].astype(int), np.asarray(Z["g"], float))))
    i0 = 0
    for rid in rids:
        sel, fc, fa = raw[rid]
        n = len(sel)
        X = (FCOR[i0:i0 + n] / F0 - 1).astype(float)
        i0 += n
        fr = sel.isx_frame.values.astype(int)
        valid = sel.valid.values.astype(bool) & np.isfinite(X).all(1)
        t_s = fr / FPS
        k = X.shape[1]
        B, D, Zs = np.full_like(X, np.nan), np.full_like(X, np.nan), np.full_like(X, np.nan)
        Cc, Ss = np.zeros_like(X), np.zeros_like(X)
        sig, gg = np.full(k, np.nan), np.full(k, np.nan)
        ev_rows = []
        st = int(SKIP_S * FPS)
        for j in range(k):
            x = X[:, j]
            okv = valid & np.isfinite(x)
            if okv.sum() < 10 * FPS:
                continue
            xi = np.interp(np.arange(n), np.where(okv)[0], x[okv])
            b = np.empty(n)
            b[st:] = cto.baseline(xi[st:])
            b[:st] = b[st]
            dff = (xi - b) / np.maximum(1 + b, 1e-3)
            sn = cto.noise_psd(dff[st:])
            mad = 1.4826 * np.median(np.abs(dff[st:] - np.median(dff[st:])))
            g = gmap.get(int(ids[j]), np.exp(-1 / (1.27 * FPS)))
            c, spk = cto.oasis_ar1(dff[st:], g, 2.5 * sn * np.sqrt(1 + g * g))
            z = dff / max(mad, 1e-9)
            dff[~okv] = np.nan; z[~okv] = np.nan; dff[:st] = np.nan; z[:st] = np.nan
            B[:, j], D[:, j], Zs[:, j], sig[j], gg[j] = b, dff, z, sn, g
            Cc[st:, j], Ss[st:, j] = c, spk
            for on, pk, amp, ssum in cto.events_from_s(spk, c):
                on, pk = on + st, pk + st
                if not okv[on]:
                    continue
                ev_rows.append(dict(cell=int(ids[j]), onset_frame=int(fr[on]), peak_frame=int(fr[pk]), onset_s=t_s[on],
                                    peak_s=t_s[pk], amplitude_dff=round(amp, 5), peak_z=round(amp / max(mad, 1e-9), 2)))
        E = pd.DataFrame(ev_rows, columns=["cell", "onset_frame", "peak_frame", "onset_s", "peak_s", "amplitude_dff", "peak_z"])
        E.to_csv(os.path.join(out, f"{rid}_events.csv"), index=False)
        np.savez_compressed(os.path.join(out, f"{rid}_traces.npz"), frame=fr, t_s=t_s, valid=valid,
                            x=X.astype(np.float32), baseline=B.astype(np.float32), dff=D.astype(np.float32),
                            z=Zs.astype(np.float32), c=Cc.astype(np.float32), s=Ss.astype(np.float32),
                            sn=sig, g=gg, tau_s=-1 / (FPS * np.log(gg)), cell_ids=ids, skip_s=SKIP_S)
        dur = (valid & (t_s >= SKIP_S)).sum() / FPS / 60
        print(f"{rid}: {k} cells, {len(E)} transients ({len(E) / max(k, 1) / max(dur, 1e-9):.2f} per cell per min), "
              f"median noise {np.nanmedian(sig):.4f}")
    print("->", out)


if __name__ == "__main__":
    main()
