"""Check that an isxd_to_h5 output is a lossless copy of the raw IDAS movie(s), without IDPS.

The raw frames are read straight from the .isxd bytes. Layout of a raw IDAS movie (nVista3,
IDAS 1.5.2; see isx_meta.py): frames are stored back to back from byte 0, each a block of
width x (height + 4) little-endian uint16 values (2 header and 2 footer rows when the footer
JSON says hasFrameHeaderFooter), and the JSON footer follows the last frame. Which rows of a
block are the image is NOT assumed: it is established by comparing a few frames with the h5
(every candidate offset 0..4 is tried; exactly one must match all probe frames), and that
rule is then applied to every compared frame. On the 2024 nVista3 files the image is rows
2..H+1 (2 header rows on top, 2 footer rows at the bottom).

Checks (each one can fail the run):
  footer   dataType uint16, numTimes * block bytes == JSON offset (file not truncated)
  table    one row per raw frame, isx_frame 0..n-1 per recording; valid == 0 exactly at the
           footer's dropped / cropped / blank frames, with that invalid_reason; h5_row empty
           for invalid frames and 1..N in table order for valid ones
  h5       /mov is uint16 with h5py shape (N, W, H), i.e. MATLAB [H W N]
  pixels   every h5 frame is byte-identical to its raw frame (or every frame of a sample:
           --sample N picks N random valid frames plus the first and last frame of each
           recording); max |diff| is reported
  checksum the frame table's checksum_sum / checksum_wsum equal the values recomputed here
  zeros    no compared h5 frame is all zero

h5py is used when it can be imported. Without it, pixel equality cannot be read from the
h5 here; the script then compares checksums computed from the raw bytes with the frame
table's checksums (which isxd_to_h5.m recomputed from the written h5 in its own verification
pass) and says so in the report (mode 'checksum_fallback'). Nothing is ever installed.

Usage
  python verify_lossless.py --h5 OUT/P1_d1_pre_raw.h5
        [--frames OUT/P1_d1_pre_raw_frames.csv]   default: next to the h5
        [--isxd RAW.isxd [RAW2.isxd ...]]          default: the sources named in <name>_raw.json
        [--sample N] [--seed 0] [--json-out report.json] [--no-h5py]
  python verify_lossless.py --selftest
Exit code 0 = PASS, 1 = FAIL.
"""

import argparse
import csv
import json
import os
import struct
import sys
import tempfile
import time

import numpy as np

from isx_meta import read_footer_json, write_fake_isxd, fake_movie_footer

try:
    import h5py
    HAVE_H5PY = True
except ImportError:          # reported, never installed
    h5py = None
    HAVE_H5PY = False

EXTRA_ROWS = 4
TABLE_COLUMNS = ["recording_id", "isx_frame", "valid", "invalid_reason", "h5_row"]


# ----------------------------------------------------------------------------- raw movie

class RawMovie:
    """Frames of a raw IDAS .isxd read from its bytes (no IDPS)."""

    def __init__(self, path):
        js, layout, json_offset, _ = read_footer_json(path)
        if layout != "isxd_footer":
            raise ValueError(f"{path}: not an isxd movie footer")
        self.path = path
        self.footer = js
        ti = js.get("timingInfo") or {}
        px = js.get("spacingInfo") or {}
        self.n = int(ti.get("numTimes"))
        self.width = int(px["numPixels"]["x"])
        self.height = int(px["numPixels"]["y"])
        self.has_hf = bool(js.get("hasFrameHeaderFooter"))
        self.extra_rows = EXTRA_ROWS if self.has_hf else 0
        self.rows = self.height + self.extra_rows
        self.data_type = js.get("dataType")
        self.block_bytes = self.width * self.rows * 2
        self.json_offset = json_offset
        self.size_ok = (self.data_type == 0 and json_offset == self.n * self.block_bytes)
        self.invalid = {}
        for k in ti.get("dropped") or []:
            self._mark(k, "dropped")
        for c in ti.get("cropped") or []:
            if isinstance(c, (list, tuple)) and len(c) == 2:
                for k in range(int(c[0]), int(c[1]) + 1):
                    self._mark(k, "cropped")
            else:
                self._mark(c, "cropped")
        for k in ti.get("blank") or []:
            self._mark(k, "blank")
        self._fh = open(path, "rb")

    def _mark(self, k, why):
        k = int(k)
        # dropped wins over cropped wins over blank (the order isxd_to_h5 uses)
        order = {"dropped": 0, "cropped": 1, "blank": 2}
        if k not in self.invalid or order[why] < order[self.invalid[k]]:
            self.invalid[k] = why

    def block(self, k):
        """The whole (height + extra) x width block of frame k, uint16."""
        self._fh.seek(k * self.block_bytes)
        buf = self._fh.read(self.block_bytes)
        if len(buf) != self.block_bytes:
            raise ValueError(f"{self.path}: frame {k} is truncated")
        return np.frombuffer(buf, dtype="<u2").reshape(self.rows, self.width)

    def close(self):
        self._fh.close()


# ----------------------------------------------------------------------------- frame table

def read_frame_table(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: empty frame table")
    missing = [c for c in TABLE_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: frame table lacks column(s) {', '.join(missing)}")
    out = []
    for r in rows:
        out.append({
            "recording_id": r["recording_id"],
            "isx_frame": int(r["isx_frame"]),
            "valid": int(r["valid"]),
            "invalid_reason": r.get("invalid_reason") or "",
            "h5_row": int(r["h5_row"]) if r["h5_row"] not in ("", None) else None,
            "checksum_sum": int(r["checksum_sum"]) if r.get("checksum_sum") else None,
            "checksum_wsum": int(r["checksum_wsum"]) if r.get("checksum_wsum") else None,
        })
    return out


def check_table(table, movies):
    """Contract checks of the frame table against the raw footers. Returns error strings."""
    errors = []
    ids = []
    for r in table:
        if r["recording_id"] not in ids:
            ids.append(r["recording_id"])
    if len(ids) != len(movies):
        return [f"frame table has {len(ids)} recordings ({', '.join(ids)}), {len(movies)} .isxd given"]
    expect_row = 1
    for rid, mv in zip(ids, movies):
        rows = [r for r in table if r["recording_id"] == rid]
        n = getattr(mv, "n_check", mv.n)       # a max_frames test copy holds the first n only
        if [r["isx_frame"] for r in rows] != list(range(n)):
            errors.append(f"{rid}: isx_frame is not 0..{n - 1} (footer numTimes {mv.n}, "
                          f"table has {len(rows)} rows)")
            continue
        for r in rows:
            k = r["isx_frame"]
            why = mv.invalid.get(k, "")
            if why:
                if r["valid"] != 0 or r["invalid_reason"] != why or r["h5_row"] is not None:
                    errors.append(f"{rid} frame {k}: footer says {why}, table says valid={r['valid']} "
                                  f"reason='{r['invalid_reason']}' h5_row={r['h5_row']}")
            else:
                if r["valid"] != 1 or r["invalid_reason"] != "" or r["h5_row"] != expect_row:
                    errors.append(f"{rid} frame {k}: expected valid with h5_row {expect_row}, table says "
                                  f"valid={r['valid']} reason='{r['invalid_reason']}' h5_row={r['h5_row']}")
                    if len(errors) > 20:
                        return errors + ["... (more)"]
                expect_row += 1
    return errors


# ----------------------------------------------------------------------------- checks

def weights(n_pix):
    return np.arange(1, n_pix + 1, dtype=np.uint64)


def checksums(frame_wh, w):
    """(sum, weighted sum) of a frame given as h5py reads it, (W, H): its C-order ravel is
    MATLAB's column-major v(:) of the H x W frame, so the weights match isxd_to_h5.m."""
    v = frame_wh.ravel().astype(np.uint64)
    return int(v.sum(dtype=np.uint64)), int(np.dot(v, w))


def find_layout(movies, items, ds):
    """Row offset of the image inside each raw block, from up to 5 probe frames.

    Every candidate offset is tried on every probe frame. The rule is the offset that the
    most probe frames match, if exactly one offset is best and at least one frame matches;
    a damaged probe frame then still gets counted by the full comparison instead of hiding
    the layout. Returns (offset or None, number of probe frames, matches per offset)."""
    probe = items[:2] + items[len(items) // 2: len(items) // 2 + 1] + items[-2:]
    seen = []
    for it in probe:
        if it not in seen:
            seen.append(it)
    votes = {o: 0 for o in range(movies[0].extra_rows + 1)}
    for mi, k, row in seen:
        mv = movies[mi]
        blk = mv.block(k)
        frame = np.asarray(ds[row - 1]).T          # (H, W)
        for o in votes:
            if np.array_equal(blk[o:o + mv.height], frame):
                votes[o] += 1
    best = max(votes.values())
    winners = [o for o, v in votes.items() if v == best]
    if best == 0 or len(winners) != 1:
        return None, len(seen), votes
    return winners[0], len(seen), votes


def select_items(table, movies, sample, seed):
    """[(movie index, isx_frame, h5_row)] of the frames to compare, in h5 order."""
    ids = []
    for r in table:
        if r["recording_id"] not in ids:
            ids.append(r["recording_id"])
    items = []
    ends = set()
    for mi, rid in enumerate(ids):
        rows = [(mi, r["isx_frame"], r["h5_row"]) for r in table
                if r["recording_id"] == rid and r["valid"] == 1 and r["h5_row"] is not None]
        items += rows
        if rows:
            ends.update({rows[0], rows[-1]})
    if sample and sample < len(items):
        rng = np.random.default_rng(seed)
        pick = set(rng.choice(len(items), size=sample, replace=False).tolist())
        items = [it for i, it in enumerate(items) if i in pick or it in ends]
    return sorted(items, key=lambda it: it[2])


def verify(h5_path, frames_csv, isxd_paths, sample=0, seed=0, use_h5py=True, progress=True):
    t0 = time.time()
    rep = {"h5": os.path.abspath(h5_path), "frames_csv": os.path.abspath(frames_csv),
           "isxd": [os.path.abspath(p) for p in isxd_paths], "errors": [],
           "mode": "pixels" if (use_h5py and HAVE_H5PY) else "checksum_fallback",
           "h5py": HAVE_H5PY, "sample": sample or "all"}
    movies = [RawMovie(p) for p in isxd_paths]
    # isxd_to_h5 max_frames (test copies of the first N frames): only when its own JSON says so
    js_path = h5_path[:-3] + ".json" if h5_path.endswith(".h5") else ""
    if js_path and os.path.isfile(js_path):
        try:
            with open(js_path, encoding="utf-8") as f:
                mf = json.load(f).get("params", {}).get("max_frames")
            if isinstance(mf, (int, float)) and mf > 0:
                for mv in movies:
                    mv.n_check = min(mv.n, int(mf))
                rep["max_frames"] = int(mf)
        except (ValueError, OSError):
            pass
    try:
        for mv in movies:
            if not mv.size_ok:
                rep["errors"].append(f"{mv.path}: footer size check failed (dataType {mv.data_type}, "
                                     f"JSON offset {mv.json_offset} != {mv.n} x {mv.block_bytes})")
        if len({(m.width, m.height) for m in movies}) != 1:
            rep["errors"].append("raw movies differ in frame size")
        table = read_frame_table(frames_csv)
        rep["table_errors"] = check_table(table, movies)
        rep["errors"] += rep["table_errors"]
        items = select_items(table, movies, sample, seed)
        n_valid = sum(1 for r in table if r["valid"] == 1)
        rep["n_valid"] = n_valid
        W, H = movies[0].width, movies[0].height
        w = weights(W * H)
        have_cs = all(r["checksum_sum"] is not None for r in table if r["valid"] == 1)
        rep["table_has_checksums"] = have_cs
        stats = {"n_frames_compared": 0, "n_frames_differ": 0, "max_abs_diff": 0,
                 "n_checksum_mismatch": 0, "n_zero_frames": 0}
        by_row = {r["h5_row"]: r for r in table if r["h5_row"] is not None}

        if rep["mode"] == "pixels":
            with h5py.File(h5_path, "r") as f:
                if "mov" not in f:
                    raise ValueError(f"{h5_path}: no /mov dataset")
                ds = f["mov"]
                rep["h5_shape"] = list(ds.shape)
                rep["h5_dtype"] = str(ds.dtype)
                rep["h5_chunks"] = list(ds.chunks) if ds.chunks else None
                if ds.dtype != np.dtype("<u2"):
                    rep["errors"].append(f"/mov dtype is {ds.dtype}, not uint16")
                if tuple(ds.shape) != (n_valid, W, H):
                    rep["errors"].append(f"/mov h5py shape {tuple(ds.shape)} != (n_valid, W, H) = "
                                         f"{(n_valid, W, H)} (MATLAB [H W N])")
                if rep["errors"]:
                    return finish(rep, stats, t0)
                off, n_probe, votes = find_layout(movies, items, ds)
                rep["layout_probe_frames"] = n_probe
                rep["layout_votes"] = {str(o): v for o, v in votes.items()}
                if off is None:
                    rep["errors"].append("no single row offset makes the raw blocks equal the h5 frames "
                                         "(layout not established: data differ or orientation is wrong)")
                    return finish(rep, stats, t0)
                rep["layout"] = {"image_rows": [off, off + H - 1], "header_rows": off,
                                 "footer_rows": movies[0].extra_rows - off, "block_rows": movies[0].rows,
                                 "rule": f"image = rows {off}..{off + H - 1} of each {movies[0].rows} x {W} block"}
                depth = ds.chunks[0] if ds.chunks else 64
                # Read the h5 one chunk slab at a time so that every chunk is read once.
                slabs = {}
                for it in items:
                    slabs.setdefault((it[2] - 1) // depth, []).append(it)
                done = 0
                for s in sorted(slabs):
                    lo = s * depth
                    block = ds[lo:min(lo + depth, ds.shape[0])]
                    for mi, k, row in slabs[s]:
                        fr_wh = block[row - 1 - lo]
                        raw = movies[mi].block(k)[off:off + H]
                        a = fr_wh.T
                        if not np.array_equal(a, raw):
                            stats["n_frames_differ"] += 1
                            d = int(np.max(np.abs(a.astype(np.int32) - raw.astype(np.int32))))
                            stats["max_abs_diff"] = max(stats["max_abs_diff"], d)
                        if not fr_wh.any():
                            stats["n_zero_frames"] += 1
                        if have_cs:
                            cs = checksums(fr_wh, w)
                            r = by_row[row]
                            if cs != (r["checksum_sum"], r["checksum_wsum"]):
                                stats["n_checksum_mismatch"] += 1
                        stats["n_frames_compared"] += 1
                    done += len(slabs[s])
                    if progress and (s % 40 == 0):
                        print(f"  {done} / {len(items)} frames ({time.time() - t0:.0f} s)", flush=True)
        else:
            rep["note"] = ("h5py not available: pixel equality NOT read from the h5 here; raw-byte "
                           "checksums compared with the frame table (verified against the h5 by "
                           "isxd_to_h5.m)")
            if not have_cs:
                rep["errors"].append("checksum fallback needs checksum_sum / checksum_wsum in the frame table")
                return finish(rep, stats, t0)
            off = 2 if movies[0].extra_rows == EXTRA_ROWS else 0
            rep["layout"] = {"image_rows": [off, off + H - 1], "rule": "assumed (cannot be established "
                             "without reading the h5)"}
            for mi, k, row in items:
                raw = movies[mi].block(k)[off:off + H]
                cs = checksums(np.ascontiguousarray(raw.T), w)
                r = by_row[row]
                if cs != (r["checksum_sum"], r["checksum_wsum"]):
                    stats["n_checksum_mismatch"] += 1
                if not raw.any():
                    stats["n_zero_frames"] += 1
                stats["n_frames_compared"] += 1
        return finish(rep, stats, t0)
    finally:
        for mv in movies:
            mv.close()


def finish(rep, stats, t0):
    rep.update(stats)
    for key, what in (("n_frames_differ", "frames differ from the raw bytes"),
                      ("n_checksum_mismatch", "checksum mismatches"),
                      ("n_zero_frames", "all-zero h5 frames")):
        if stats[key]:
            rep["errors"].append(f"{stats[key]} {what}")
    rep["all_frames"] = (rep.get("sample") == "all" and stats["n_frames_compared"] == rep.get("n_valid"))
    rep["passed"] = not rep["errors"] and stats["n_frames_compared"] > 0
    rep["runtime_s"] = round(time.time() - t0, 1)
    return rep


def sources_from_log(h5_path):
    """The .isxd files named in <name>_raw.json next to the h5, in recording order."""
    log = h5_path[:-len(".h5")] + ".json" if h5_path.endswith(".h5") else None
    if not log or not os.path.isfile(log):
        raise ValueError(f"give --isxd: no log {log}")
    with open(log, encoding="utf-8") as f:
        js = json.load(f)
    recs = js.get("recordings")
    if isinstance(recs, dict):
        recs = [recs]
    return [r["file"] for r in recs]


def summary(rep):
    lay = rep.get("layout", {}).get("rule", "layout not established")
    head = "PASS" if rep["passed"] else "FAIL"
    s = (f"{head}  {os.path.basename(rep['h5'])}: {rep.get('n_frames_compared', 0)} of {rep.get('n_valid')} "
         f"valid frames compared ({rep['mode']}), max |diff| = {rep.get('max_abs_diff')}, "
         f"{rep.get('n_frames_differ', 0)} differ, {rep.get('n_checksum_mismatch', 0)} checksum mismatches, "
         f"{rep.get('n_zero_frames', 0)} zero frames; {lay}; {rep.get('runtime_s')} s")
    for e in rep["errors"][:20]:
        s += f"\n  ERROR {e}"
    return s


# ----------------------------------------------------------------------------- self-test

def _make_case(tmp, name, n, W, H, dropped=(), cropped=(), seed=0):
    """A raw-IDAS-like .isxd (header rows with junk) and its true image frames."""
    rng = np.random.default_rng(seed)
    footer, _ = fake_movie_footer(n, width=W, height=H, dropped=dropped, cropped=cropped)
    blocks = []
    frames = []
    for k in range(n):
        img = rng.integers(0, 65536, size=(H, W), dtype=np.uint16)
        img[0, 0], img[-1, -1] = 0, 65535
        hdr = rng.integers(0, 65536, size=(2, W), dtype=np.uint16)
        ftr = rng.integers(0, 65536, size=(2, W), dtype=np.uint16)
        blocks.append(np.vstack([hdr, img, ftr]).astype("<u2").tobytes())
        frames.append(img)
    path = os.path.join(tmp, name + ".isxd")
    write_fake_isxd(path, footer, b"".join(blocks))
    return path, frames


def _write_outputs(tmp, cases, h5_name, table_name, layout="matlab"):
    """h5 + frame table the way isxd_to_h5.m writes them (h5py: (N, W, H) uint16)."""
    rows = []
    valid_frames = []
    W = H = None
    for rid, (path, frames, invalid) in cases.items():
        H, W = frames[0].shape
        for k, img in enumerate(frames):
            if k in invalid:
                rows.append([rid, "", k, 0, invalid[k], "", "", "", ""])
            else:
                valid_frames.append(img)
                v = np.ascontiguousarray(img.T).ravel().astype(np.uint64)
                rows.append([rid, "", k, 1, "", len(valid_frames), 1000 + k,
                             int(v.sum()), int(np.dot(v, weights(W * H)))])
    table = os.path.join(tmp, table_name)
    with open(table, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["recording_id", "assay", "isx_frame", "valid", "invalid_reason", "h5_row", "tsc_us",
                     "checksum_sum", "checksum_wsum"])
        wr.writerows(rows)
    h5 = os.path.join(tmp, h5_name)
    if HAVE_H5PY:
        with h5py.File(h5, "w") as f:
            if layout == "matlab":
                data = np.stack([img.T for img in valid_frames])          # (N, W, H)
            else:
                data = np.stack(valid_frames)                              # (N, H, W): wrong
            f.create_dataset("mov", data=data.astype("<u2"), chunks=(3,) + data.shape[1:])
    return h5, table


def selftest():
    failures = []

    def check(cond, what):
        print(("  ok    " if cond else "  FAIL  ") + what)
        if not cond:
            failures.append(what)

    print(f"verify_lossless selftest (h5py {'available' if HAVE_H5PY else 'NOT available'})")
    with tempfile.TemporaryDirectory() as tmp:
        W, H = 11, 7
        pa, fa = _make_case(tmp, "a", 9, W, H, dropped=[3], seed=1)
        pb, fb = _make_case(tmp, "b", 6, W, H, cropped=[[1, 2]], seed=2)
        cases = {"T_pre": (pa, fa, {3: "dropped"}), "T_post": (pb, fb, {1: "cropped", 2: "cropped"})}
        h5, table = _write_outputs(tmp, cases, "good.h5", "good.csv")
        mode_pixels = HAVE_H5PY

        rep = verify(h5, table, [pa, pb], progress=False, use_h5py=mode_pixels)
        check(rep["passed"] and rep["n_frames_compared"] == 12, "good copy PASSES, every valid frame compared")
        if mode_pixels:
            check(rep["layout"]["image_rows"] == [2, H + 1], "layout established: image rows 2..H+1")
            check(rep["all_frames"], "all_frames flag")
            rep = verify(h5, table, [pa, pb], sample=3, seed=5, progress=False)
            check(rep["passed"] and 3 <= rep["n_frames_compared"] <= 7, "--sample: random frames + ends PASS")

            # one changed pixel
            bad = os.path.join(tmp, "pix.h5")
            with h5py.File(h5, "r") as f, h5py.File(bad, "w") as g:
                d = f["mov"][...]
                d[4, 2, 3] ^= 1
                g.create_dataset("mov", data=d, chunks=f["mov"].chunks)
            rep = verify(bad, table, [pa, pb], progress=False)
            check(not rep["passed"] and rep["n_frames_differ"] == 1 and rep["max_abs_diff"] == 1
                  and rep["n_checksum_mismatch"] == 1, "one changed pixel FAILS (max |diff| 1)")

            # two swapped frames
            with h5py.File(h5, "r") as f, h5py.File(bad, "w") as g:
                d = f["mov"][...]
                d[[5, 6]] = d[[6, 5]]
                g.create_dataset("mov", data=d, chunks=f["mov"].chunks)
            rep = verify(bad, table, [pa, pb], progress=False)
            check(not rep["passed"] and rep["n_frames_differ"] == 2, "two swapped frames FAIL")

            # an all-zero frame
            with h5py.File(h5, "r") as f, h5py.File(bad, "w") as g:
                d = f["mov"][...]
                d[7] = 0
                g.create_dataset("mov", data=d, chunks=f["mov"].chunks)
            rep = verify(bad, table, [pa, pb], progress=False)
            check(not rep["passed"] and rep["n_zero_frames"] == 1, "all-zero h5 frame FAILS")

            # wrong orientation (N, H, W)
            hw, _ = _write_outputs(tmp, cases, "hw.h5", "hw.csv", layout="hw")
            rep = verify(hw, table, [pa, pb], progress=False)
            check(not rep["passed"] and any("shape" in e for e in rep["errors"]), "transposed /mov FAILS")

            # float copy of the right values
            with h5py.File(h5, "r") as f, h5py.File(bad, "w") as g:
                g.create_dataset("mov", data=f["mov"][...].astype(np.float32))
            rep = verify(bad, table, [pa, pb], progress=False)
            check(not rep["passed"] and any("dtype" in e for e in rep["errors"]), "single /mov FAILS (not uint16)")

        # frame table damage (both modes)
        def damaged_table(edit):
            with open(table, newline="") as f:
                rows = list(csv.reader(f))
            edit(rows)
            p = os.path.join(tmp, "bad.csv")
            with open(p, "w", newline="") as f:
                csv.writer(f).writerows(rows)
            return p

        def shift_row(rows):
            rows[6][5] = str(int(rows[6][5]) + 1)
        rep = verify(h5, damaged_table(shift_row), [pa, pb], progress=False, use_h5py=mode_pixels)
        check(not rep["passed"] and rep["table_errors"], "h5_row off by one in the table FAILS")

        def undrop(rows):
            rows[4][3], rows[4][4], rows[4][5] = "1", "", "4"
        rep = verify(h5, damaged_table(undrop), [pa, pb], progress=False, use_h5py=mode_pixels)
        check(not rep["passed"] and rep["table_errors"], "a dropped frame marked valid FAILS")

        def bad_checksum(rows):
            rows[2][8] = str(int(rows[2][8]) + 1)
        rep = verify(h5, damaged_table(bad_checksum), [pa, pb], progress=False, use_h5py=mode_pixels)
        check(not rep["passed"] and rep["n_checksum_mismatch"] == 1, "a wrong checksum FAILS")

        # checksum fallback (forced): passes on the good copy, catches a wrong checksum
        rep = verify(h5, table, [pa, pb], progress=False, use_h5py=False)
        check(rep["passed"] and rep["mode"] == "checksum_fallback", "checksum fallback PASSES the good copy")
        rep = verify(h5, damaged_table(bad_checksum), [pa, pb], progress=False, use_h5py=False)
        check(not rep["passed"], "checksum fallback catches a wrong checksum")

        # truncated raw file
        short = os.path.join(tmp, "short.isxd")
        footer, _ = fake_movie_footer(9, width=W, height=H, dropped=[3])
        with open(pa, "rb") as f:
            data = f.read(8 * W * (H + 4) * 2)
        write_fake_isxd(short, footer, data)
        rep = verify(h5, table, [short, pb], progress=False, use_h5py=mode_pixels)
        check(not rep["passed"] and any("footer size" in e for e in rep["errors"]), "truncated .isxd FAILS")

    print("verify_lossless selftest:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h5", help="<name>_raw.h5 written by isxd_to_h5.m")
    ap.add_argument("--frames", help="frame table (default <name>_raw_frames.csv next to the h5)")
    ap.add_argument("--isxd", nargs="+", help="raw movies in recording order (default: from <name>_raw.json)")
    ap.add_argument("--sample", type=int, default=0, help="compare N random valid frames (+ first/last); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", help="write the report as JSON")
    ap.add_argument("--no-h5py", action="store_true", help="force the checksum fallback")
    ap.add_argument("--selftest", action="store_true", help="run the built-in tests and exit")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.h5:
        ap.error("give --h5, or --selftest")
    frames = args.frames or (args.h5[:-len(".h5")] + "_frames.csv")
    isxd = args.isxd or sources_from_log(args.h5)
    if not HAVE_H5PY:
        print("h5py is not importable: falling back to checksums (nothing is installed)")
    rep = verify(args.h5, frames, isxd, sample=args.sample, seed=args.seed, use_h5py=not args.no_h5py)
    print(summary(rep))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
