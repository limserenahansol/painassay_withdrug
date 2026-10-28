"""Read the metadata JSON that Inscopix IDAS / IDPS store inside .isxd, .gpio and .imu files.

No Inscopix software is needed. Only the last bytes of a file are read (plus the first
64 bytes of a raw .gpio), so multi-GB movies cost nothing.

Two layouts exist on disk:

  isxd footer    [data ...][JSON][NUL][uint64 LE = len(JSON)]
                 raw IDAS movies (*_video.isxd) and IDPS files (*_gpio.isxd, *_imu.isxd, -PP.isxd ...)
  trailing JSON  [binary ...][JSON]  (no length field)
                 raw IDAS *_video.gpio and *_video.imu; the JSON is the IDAS session JSON

Both were verified on the 2024-12 nVista3 recordings (IDAS 1.5.2 / IDPS 1.9.6): for every raw
movie numTimes * width * (height + 4) * 2 + len(JSON) + 9 == file size exactly (uint16 pixels,
2 header + 2 footer rows per frame), and for every raw .gpio header word 5 == offset of the JSON.

The per-frame TSC timestamps live inside the frame header rows and are NOT read here. In a
raw IDAS movie the footer `period` is the nominal one (50 ms on the 2024 rig, while the true
mean SYNC period was 49.96 ms); IDPS-processed movies store a measured mean period instead.
Never use `period_s` from a raw footer for sync (CONTRACTS.md §1).

Use:
  python isx_meta.py FILE [FILE ...]        one summary line per file
  python isx_meta.py FILE --json            normalised metadata as JSON
  python isx_meta.py FILE --raw             the whole stored JSON
  python isx_meta.py --selftest             synthetic files only, exits 1 on failure

From Python:  from isx_meta import read_meta;  meta = read_meta(path)
"""

import argparse
import json
import os
import struct
import sys
import tempfile
from datetime import datetime, timezone

# Footer 'type' values seen in real files. Others are reported as 'other'.
ISX_TYPE_KIND = {0: "movie", 3: "gpio", 7: "imu"}
# Footer 'dataType' -> bytes per pixel. Only uint16 (0) is verified by the exact size check.
BYTES_PER_PIXEL = {0: 2}
FRAME_EXTRA_ROWS = 4          # 2 header + 2 footer rows when hasFrameHeaderFooter is true
RAW_GPIO_HEADER_BYTES = 64
MAX_FOOTER_BYTES = 64 * 1024 * 1024   # sanity bound on the stored length (real footers: ~20 kB)
MAX_TAIL_BYTES = 8 * 1024 * 1024      # how far back to look for a trailing JSON (real: ~7 kB)
SYNC_CHANNEL = "BNC Sync Output"


# ----------------------------------------------------------------------------- reading

def _read_isxd_footer(f, size):
    """JSON from the isxd footer layout, or None if the file does not follow it."""
    if size < 9:
        return None
    f.seek(size - 8)
    n = struct.unpack("<Q", f.read(8))[0]
    if not 2 <= n <= min(size - 9, MAX_FOOTER_BYTES):
        return None
    f.seek(size - 9 - n)
    blob = f.read(n + 1)
    if blob[-1:] != b"\x00":
        return None
    try:
        return json.loads(blob[:-1].decode("utf-8")), size - 9 - n, n
    except (UnicodeDecodeError, ValueError):
        return None


def _read_trailing_json(f, size):
    """The JSON object that runs to the end of the file (raw .gpio / .imu), or None.

    There is no length field, so we try every '{"' in the tail, earliest first: the first one
    that decodes all the way to EOF is the outermost object. Binary bytes before it only cost a
    failed attempt each.
    """
    decoder = json.JSONDecoder()
    tail_len = 64 * 1024
    while True:
        start = max(0, size - tail_len)
        f.seek(start)
        tail = f.read(size - start)
        end = len(tail.rstrip(b"\x00 \r\n\t"))
        if not tail[:end].endswith(b"}"):
            return None                       # not a JSON-terminated file: skip the costly scan
        i = tail.find(b'{"')
        while 0 <= i < end:
            try:
                text = tail[i:end].decode("utf-8")
                obj, used = decoder.raw_decode(text)
                if used == len(text) and isinstance(obj, dict):
                    return obj, start + i, end - i
            except (UnicodeDecodeError, ValueError):
                pass
            i = tail.find(b'{"', i + 1)
        if start == 0 or tail_len >= MAX_TAIL_BYTES:
            return None
        tail_len = min(tail_len * 8, MAX_TAIL_BYTES)


def read_footer_json(path):
    """(json_dict, layout, json_offset, json_length) for an Inscopix file.

    Raises ValueError if neither layout is found.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        found = _read_isxd_footer(f, size)
        if found:
            return found[0], "isxd_footer", found[1], found[2]
        found = _read_trailing_json(f, size)
        if found:
            return found[0], "trailing_json", found[1], found[2]
    raise ValueError(f"{path}: no Inscopix metadata JSON found (neither isxd footer nor trailing JSON)")


# ----------------------------------------------------------------------------- helpers

def _get(d, dotted, default=None):
    """d['a']['b']['c'] for 'a.b.c', or default if any level is missing."""
    for key in dotted.split("."):
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def _frac(x):
    """IDPS rationals are {'num': n, 'den': d}; plain numbers pass through."""
    if isinstance(x, dict) and "num" in x and "den" in x:
        return x["num"] / x["den"] if x["den"] else None
    return x


def _epoch(t):
    """Seconds since the Unix epoch from an IDPS time {'secsSinceEpoch': {num, den}, 'utcOffset'}."""
    return _frac(t.get("secsSinceEpoch")) if isinstance(t, dict) else None


def _utc_iso(epoch_s):
    if epoch_s is None:
        return None
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat(timespec="milliseconds")


def _n_in_ranges(items):
    """Number of samples in an IDPS list: plain indices count 1, [first, last] ranges count all."""
    n = 0
    for it in items or []:
        if isinstance(it, (list, tuple)) and len(it) == 2:
            n += int(it[1]) - int(it[0]) + 1
        else:
            n += 1
    return n


def _session_fields(session):
    """Acquisition settings from an IDAS session JSON (the movie's extraProperties, or the
    whole JSON of a raw .gpio/.imu)."""
    if not isinstance(session, dict):
        return {}
    mode = session.get("adMode") or "auto"
    return {
        "microscope_type": _get(session, "microscope.type"),
        "microscope_serial": _get(session, "microscope.serial"),
        "system_serial": _get(session, "microscope.systemSerial"),
        "acquisition_sw": " ".join(str(x) for x in (_get(session, "producer.name"),
                                                    _get(session, "producer.version")) if x) or None,
        "session_name": session.get("name"),
        "fps_setting": _get(session, "microscope.fps.fps"),
        "exposure_ms": _get(session, "microscope.exp"),
        "int_time_ms": _get(session, f"{mode}.mScope.intTime"),
        "gain": _get(session, "microscope.gain"),
        "focus": _get(session, "microscope.focus"),
        "ex_led_mw": _get(session, "microscope.led.exPower"),
        "sensor_mode": _get(session, "microscope.sensorMode"),
        "downsample_space": _get(session, "microscope.downSamplingFactor"),
        "multiplane": _get(session, "microscope.multiplane.enabled"),
        "gpio_clock_hz": _get(session, f"{mode}.clock"),
        "ad_mode": session.get("adMode"),
        "start_tsc": session.get("startTsc"),
        "first_tsc": session.get("firstTsc"),
    }


# ----------------------------------------------------------------------------- normalise

def read_meta(path):
    """Normalised metadata for one Inscopix file (dict, JSON-serialisable).

    Keys that do not apply to a file kind are None. Frame-related keys (n_frames, dropped,
    cropped, blank, period_*, start_*) come from the movie footer `timingInfo`.
    """
    size = os.path.getsize(path)
    js, layout, json_offset, json_length = read_footer_json(path)
    ext = os.path.splitext(path)[1].lower()
    meta = {"path": os.path.abspath(path), "file_size": size, "layout": layout,
            "json_offset": json_offset, "json_length": json_length}

    if layout == "isxd_footer":
        isx_type = js.get("type")
        meta["isx_type"] = isx_type
        meta["kind"] = ISX_TYPE_KIND.get(isx_type, "other")
        session = js.get("extraProperties")
        meta["isx_producer"] = " ".join(str(x) for x in (_get(js, "producer.name"),
                                                         ".".join(map(str, _get(js, "producer.version") or [])))
                                        if x) or None
    else:
        meta["isx_type"] = None
        meta["kind"] = {".gpio": "gpio_raw", ".imu": "imu_raw"}.get(ext, "session_json")
        session = js
        meta["isx_producer"] = None
    meta.update(_session_fields(session))

    # Movie timing (raw and processed .isxd movies).
    ti = js.get("timingInfo") if meta["kind"] == "movie" else None
    if isinstance(ti, dict):
        period_s = _frac(ti.get("period"))
        start = ti.get("start") or {}
        meta.update({
            "n_frames": ti.get("numTimes"),
            "dropped": list(ti.get("dropped") or []),
            "n_dropped": _n_in_ranges(ti.get("dropped")),
            "cropped": list(ti.get("cropped") or []),
            "n_cropped": _n_in_ranges(ti.get("cropped")),
            "blank": list(ti.get("blank") or []),
            "n_blank": _n_in_ranges(ti.get("blank")),
            "period_us": None if period_s is None else round(period_s * 1e6, 3),
            "period_s": period_s,
            "fps_footer": None if not period_s else 1.0 / period_s,
            "start_epoch_s": _epoch(start),
            "start_utc": _utc_iso(_epoch(start)),
            "utc_offset": start.get("utcOffset"),
        })
        px = js.get("spacingInfo") or {}
        meta.update({
            "width": _get(px, "numPixels.x"),
            "height": _get(px, "numPixels.y"),
            "pixel_size": _frac(_get(px, "pixelSize.x")),
            "has_frame_header_footer": js.get("hasFrameHeaderFooter"),
            "data_type": js.get("dataType"),
            "file_version": js.get("fileVersion"),
        })
        # How many frames the bytes actually hold: catches a truncated copy of a raw movie.
        bpp = BYTES_PER_PIXEL.get(js.get("dataType"))
        meta["frames_in_file"] = None
        meta["size_consistent"] = None
        if bpp and meta["width"] and meta["height"] and json_offset is not None:
            rows = meta["height"] + (FRAME_EXTRA_ROWS if js.get("hasFrameHeaderFooter") else 0)
            frame_bytes = meta["width"] * rows * bpp
            if json_offset % frame_bytes == 0:
                meta["frames_in_file"] = json_offset // frame_bytes
            meta["size_consistent"] = (meta["n_frames"] is not None
                                       and json_offset == meta["n_frames"] * frame_bytes)

    # IDPS-imported GPIO (*_gpio.isxd): per-channel sample counts and start/end times.
    if meta["kind"] == "gpio" and layout == "isxd_footer":
        chans = js.get("channel list") or []
        counts = js.get("numSamples") or []
        times = js.get("global times") or []
        g0 = _epoch(times[0]) if len(times) > 0 else None
        g1 = _epoch(times[1]) if len(times) > 1 else None
        meta.update({
            "channels": chans,
            "n_samples": dict(zip(chans, counts)),
            "gpio_start_epoch_s": g0,
            "gpio_end_epoch_s": g1,
            "gpio_duration_s": None if g0 is None or g1 is None else g1 - g0,
        })
        # Samples are change events: one low sample at t=0, a high+low pair per pulse, one
        # low sample at the end. So edges = (n - 2) / 2. An estimate only; count_check counts
        # the exported CSV instead.
        n_sync = meta["n_samples"].get(SYNC_CHANNEL)
        meta["sync_samples"] = n_sync
        meta["sync_edges_est"] = (n_sync - 2) // 2 if n_sync and n_sync >= 2 and n_sync % 2 == 0 else None

    # Raw IDAS .gpio: start time and packet count are in the 64-byte binary header.
    if meta["kind"] == "gpio_raw":
        with open(path, "rb") as f:
            head = f.read(RAW_GPIO_HEADER_BYTES)
        if len(head) == RAW_GPIO_HEADER_BYTES:
            words = struct.unpack("<8Q", head)
            # Word 5 is the JSON offset on every 2024 file; only trust the header if it agrees.
            if words[5] == json_offset:
                clock = meta.get("gpio_clock_hz")
                meta.update({
                    "gpio_start_epoch_s": words[0] / 1000.0,
                    "gpio_n_packets": words[3],
                    "gpio_duration_s": words[3] / clock if clock else None,
                })

    # Accelerometer timing (raw .imu: top-level timingInfo; *_imu.isxd: extraProperties.timingInfo).
    if meta["kind"] in ("imu", "imu_raw"):
        ti = js.get("timingInfo") or _get(js, "extraProperties.timingInfo") or {}
        acc = ti.get("accelerometer") or {}
        meta.update({
            "acc_n_samples": acc.get("numTimes"),
            "acc_n_dropped": _n_in_ranges(acc.get("dropped")),
            "acc_period_ms": _frac(acc.get("periodMs")),
            "mag_n_samples": _get(ti, "magnetometer.numTimes"),
            "imu_start_epoch_s": _epoch(ti.get("start")),
            "imu_end_epoch_s": _epoch(ti.get("end")),
        })
    return meta


def summary_line(meta):
    """One human-readable line per file."""
    name = os.path.basename(meta["path"])
    if meta.get("n_frames") is not None:
        size_note = "" if meta.get("size_consistent") in (True, None) else \
            f"  SIZE MISMATCH (bytes hold {meta.get('frames_in_file')} frames)"
        return (f"{name}: {meta['kind']}  frames={meta['n_frames']}  dropped={meta['n_dropped']}"
                f"  cropped={meta['n_cropped']}  period={meta['period_us']} us (footer)"
                f"  start={meta['start_utc']}  {meta.get('width')}x{meta.get('height')}"
                f"  {meta.get('microscope_type')}{size_note}")
    if meta["kind"] == "gpio":
        return (f"{name}: gpio  start={_utc_iso(meta.get('gpio_start_epoch_s'))}"
                f"  duration={meta.get('gpio_duration_s')} s  sync_edges_est={meta.get('sync_edges_est')}")
    if meta["kind"] == "gpio_raw":
        return (f"{name}: raw gpio  start={_utc_iso(meta.get('gpio_start_epoch_s'))}"
                f"  packets={meta.get('gpio_n_packets')}  duration={meta.get('gpio_duration_s')} s")
    if meta["kind"] in ("imu", "imu_raw"):
        return (f"{name}: {meta['kind']}  acc_samples={meta.get('acc_n_samples')}"
                f"  acc_dropped={meta.get('acc_n_dropped')}  acc_period={meta.get('acc_period_ms')} ms"
                f"  start={_utc_iso(meta.get('imu_start_epoch_s'))}")
    return f"{name}: {meta['kind']} (type {meta.get('isx_type')}), layout {meta['layout']}"


# ----------------------------------------------------------------------------- test files

def write_fake_isxd(path, footer, data=b""):
    """Write `data` followed by an isxd footer holding `footer` (a dict). Used by self-tests."""
    blob = json.dumps(footer, indent=4).encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)
        f.write(blob)
        f.write(b"\x00")
        f.write(struct.pack("<Q", len(blob)))


def fake_movie_footer(n_frames, width=8, height=4, period_us=50000, start_ms=1733930697233,
                      dropped=(), cropped=()):
    """A raw-IDAS-like movie footer for tests, plus the matching frame bytes length."""
    footer = {
        "dataType": 0, "type": 0, "fileVersion": 1, "hasFrameHeaderFooter": True,
        "producer": {"name": "isxcore", "version": [1, 2, 1]},
        "extraProperties": {"adMode": "auto", "auto": {"clock": 5000, "mScope": {"intTime": 49.9}},
                            "microscope": {"type": "NVista3", "serial": 1, "fps": {"fps": 20},
                                           "exp": 50, "gain": 6.3},
                            "producer": {"name": "AcquisitionSW", "version": "1.5.2"},
                            "startTsc": 123},
        "spacingInfo": {"numPixels": {"x": width, "y": height},
                        "pixelSize": {"x": {"num": 3, "den": 1}, "y": {"num": 3, "den": 1}}},
        "timingInfo": {"numTimes": n_frames, "dropped": list(dropped), "cropped": list(cropped),
                       "period": {"num": period_us, "den": 1000000},
                       "start": {"secsSinceEpoch": {"num": start_ms, "den": 1000}, "utcOffset": 0}},
    }
    return footer, n_frames * width * (height + FRAME_EXTRA_ROWS) * 2


def write_fake_raw_gpio(path, start_ms, n_packets, clock=5000):
    """A raw-IDAS-like .gpio: 64-byte header, 52-byte packets, trailing session JSON."""
    packets = bytes(52 * n_packets)
    json_offset = RAW_GPIO_HEADER_BYTES + len(packets)
    header = struct.pack("<8Q", start_ms, 1000, 0, n_packets, 210, json_offset, 0, 0)
    session = {"adMode": "auto", "auto": {"clock": clock}, "microscope": {"type": "NVista3"},
               "type": "inscopix-session"}
    with open(path, "wb") as f:
        f.write(header + packets + json.dumps(session).encode("utf-8"))


# ----------------------------------------------------------------------------- self-test

def selftest():
    failures = []

    def check(cond, what):
        print(("  ok    " if cond else "  FAIL  ") + what)
        if not cond:
            failures.append(what)

    with tempfile.TemporaryDirectory() as tmp:
        # 1. raw movie footer: all fields, exact size check
        footer, nbytes = fake_movie_footer(37, dropped=[5, 6], cropped=[[10, 12]])
        p = os.path.join(tmp, "a_video.isxd")
        write_fake_isxd(p, footer, bytes(nbytes))
        m = read_meta(p)
        check(m["layout"] == "isxd_footer" and m["kind"] == "movie", "movie footer layout found")
        check(m["n_frames"] == 37, "numTimes -> n_frames")
        check(m["dropped"] == [5, 6] and m["n_dropped"] == 2, "dropped list and count")
        check(m["n_cropped"] == 3, "cropped [first,last] range counted as 3 frames")
        check(abs(m["period_s"] - 0.05) < 1e-12 and m["period_us"] == 50000, "period in s and us")
        check(abs(m["start_epoch_s"] - 1733930697.233) < 1e-6
              and m["start_utc"] == "2024-12-11T15:24:57.233+00:00", "start epoch and UTC string")
        check((m["width"], m["height"]) == (8, 4) and m["microscope_type"] == "NVista3", "size and scope")
        check(m["size_consistent"] is True and m["frames_in_file"] == 37, "file size matches numTimes")

        # 2. truncated copy: the bytes hold fewer frames than numTimes says
        p2 = os.path.join(tmp, "short.isxd")
        write_fake_isxd(p2, footer, bytes(nbytes - 2 * (8 * 8 * 2)))
        m2 = read_meta(p2)
        check(m2["size_consistent"] is False and m2["frames_in_file"] == 35, "truncated file detected")

        # 3. raw .imu: trailing JSON, binary prefix that contains a decoy '{"'
        p3 = os.path.join(tmp, "a_video.imu")
        imu = {"type": "inscopix-session", "adMode": "auto", "auto": {"clock": 5000},
               "timingInfo": {"accelerometer": {"dropped": [3, 4, 5], "numTimes": 480,
                                                "periodMs": {"num": 20830904, "den": 1000000}},
                              "magnetometer": {"numTimes": 0},
                              "start": {"secsSinceEpoch": {"num": 1733407034412, "den": 1000}},
                              "end": {"secsSinceEpoch": {"num": 1733407044412, "den": 1000}}}}
        with open(p3, "wb") as f:
            f.write(b"\x01\x02{\"bad\x00\xff" + os.urandom(3000) + json.dumps(imu).encode())
        m3 = read_meta(p3)
        check(m3["layout"] == "trailing_json" and m3["kind"] == "imu_raw", "raw .imu trailing JSON found")
        check(m3["acc_n_samples"] == 480 and m3["acc_n_dropped"] == 3, "imu accelerometer counts")

        # 4. raw .gpio: header start time trusted only because word 5 == JSON offset
        p4 = os.path.join(tmp, "a_video.gpio")
        write_fake_raw_gpio(p4, 1733930697154, 1000)
        m4 = read_meta(p4)
        check(m4["kind"] == "gpio_raw" and abs(m4["gpio_start_epoch_s"] - 1733930697.154) < 1e-6,
              "raw .gpio start time from header")
        check(m4["gpio_n_packets"] == 1000 and abs(m4["gpio_duration_s"] - 0.2) < 1e-9,
              "raw .gpio packets / clock -> duration")

        # 5. IDPS *_gpio.isxd: sync edge estimate from change-event sample counts
        p5 = os.path.join(tmp, "a_video_gpio.isxd")
        write_fake_isxd(p5, {"type": 3, "channel list": ["GPIO-1", SYNC_CHANNEL],
                             "numSamples": [7, 2 * 1301 + 2],
                             "global times": [{"secsSinceEpoch": {"num": 1733407034104, "den": 1000}},
                                              {"secsSinceEpoch": {"num": 8667035500000, "den": 5000}}],
                             "extraProperties": {"firstTsc": 1}}, bytes(100))
        m5 = read_meta(p5)
        check(m5["kind"] == "gpio" and m5["sync_edges_est"] == 1301, "gpio isxd sync edges estimate")
        check(abs(m5["gpio_duration_s"] - (1733407100.0 - 1733407034.104)) < 1e-6, "gpio isxd duration")

        # 6. garbage file: must raise, not return nonsense
        p6 = os.path.join(tmp, "junk.isxd")
        with open(p6, "wb") as f:
            f.write(os.urandom(5000).replace(b'{"', b"xx"))
        try:
            read_meta(p6)
            check(False, "garbage file raises ValueError")
        except ValueError:
            check(True, "garbage file raises ValueError")

    print("isx_meta selftest:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help=".isxd / .gpio / .imu files")
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--json", action="store_true", help="print normalised metadata as JSON")
    out.add_argument("--raw", action="store_true", help="print the whole stored JSON")
    ap.add_argument("--selftest", action="store_true", help="run the self-test on synthetic files")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.files:
        ap.error("give at least one file, or --selftest")
    status = 0
    results = []
    for path in args.files:
        try:
            if args.raw:
                results.append(read_footer_json(path)[0])
            else:
                meta = read_meta(path)
                results.append(meta)
                if not args.json:
                    print(summary_line(meta))
        except (OSError, ValueError) as e:
            print(f"ERROR {path}: {e}", file=sys.stderr)
            status = 1
    if args.json or args.raw:
        print(json.dumps(results[0] if len(results) == 1 else results, indent=1))
    return status


if __name__ == "__main__":
    sys.exit(main())
