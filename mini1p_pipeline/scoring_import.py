"""Scoring import: ScoringAB_*.mat (dual-view manual scorer) -> behaviour_events.csv.

The scorer (scoring/score_AB_dual_view.m) stamps every mark with the BOTTOM camera
(cam1) frame index, 1-based. It saves, among others:
  score         per-frame affective state 0-4 (0 none, 1 paw attending, 2 licking or
                biting, 3 sustained lifting / guarding, 4 escape / rearing)
  uncert        per-frame 'uncertain' flag (held u)
  reflexEvents  [frame code] taps, code 1 = paw withdrawal, 2 = flinch / flick
  dFrames, dTypes  delivery taps [frame], type key 1-4 (0 = unknown), named by stimNames
  nUsed         number of frames scored; offFrames / syncOff the typed side-view offset

Output (CONTRACTS.md, behaviour_events.csv, plus a 'flags' column):
  affective states -> on/off intervals (off_cam_frame is the LAST frame of the bout,
  inclusive); reflex taps -> point events (on == off); uncertain -> 'uncertain' intervals.
  The master-clock columns (on/off isx_frame, t_master_s), event_id and latency_s are left
  empty here: master_timeline fills them with count_to_isx_frame (CONTRACTS.md 1a: camera
  frame n falls in isx_frame n + K) through camera_frames.csv, which knows about lost frames.
Also writes video_deliveries.csv (recording_id, camera, cam_frame, key, stim_name_typed,
stim_type, flags) so master_timeline can time trials the logger cannot see (heat).

The .mat file is only read (scipy.io.loadmat); it is never modified. It is tied to its
recording before anything is written: a file with nothing scored (nUsed = 0) or no delivery
taps is refused, and so is one whose filled-in mouseID / dayNo / phase (pre/post) name another
recording (--ignore-metadata overrides; blank fields and the default phase 'Baseline' are not
compared). A botFile other than the manifest's cam1 video only warns.

Usage
  python scoring_import.py --mat ScoringAB_x.mat --recording-id M01_d1_pre \
      --out-dir out/M01_d1_pre [--scorer HL] [--cam1-video M01_d1_pre_cam1.avi] [--ignore-metadata]
  python scoring_import.py --selftest
"""

import argparse
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd
import scipy.io as sio

import master_timeline as mt

AFFECTIVE_CODES = {1: "attending", 2: "licking_biting", 3: "guarding", 4: "escape"}
REFLEX_CODES = {1: "withdrawal", 2: "flinch"}
# Words the scorer's own label lists must contain, so a changed key map fails loudly
# instead of silently renaming behaviours.
AFF_NAME_WORDS = {1: ("attend",), 2: ("lick", "bit"), 3: ("guard", "lift"), 4: ("escape", "rear")}
REF_NAME_WORDS = {1: ("withdraw",), 2: ("flinch", "flick")}
STIM_WORDS = {"light": ("light",), "mild": ("mild",), "heat": ("heat", "hot", "water"),
              "pin": ("pin",)}
CAMERA = "cam1"          # the scorer stamps marks with the bottom-camera frame


class ScoringError(ValueError):
    """The .mat file does not look like a dual-view ScoringAB file."""


def _vec(x):
    return np.asarray(x).ravel() if x is not None else np.array([])


def _str(x):
    if isinstance(x, str):
        return x
    a = np.asarray(x).ravel()
    return str(a[0]) if a.size else ""


def _strlist(x):
    if x is None:
        return []
    return [str(v) for v in np.asarray(x, dtype=object).ravel()]


def load_scoring(path):
    """Read a ScoringAB .mat file into plain Python/numpy values (read-only)."""
    m = sio.loadmat(path, squeeze_me=True)
    for need in ("score", "reflexEvents", "affNames", "refNames"):
        if need not in m:
            raise ScoringError(f"{path}: no '{need}' field; not a ScoringAB file")
    ref = np.asarray(m["reflexEvents"], dtype=float)
    # squeeze_me turns a single [frame code] row into shape (2,) and none into (0,)
    if ref.size == 0:
        ref = np.zeros((0, 2))
    elif ref.ndim == 1:
        ref = ref.reshape(1, 2)
    if ref.shape[1] != 2:
        raise ScoringError(f"{path}: reflexEvents must be [frame code], got shape {ref.shape}")
    s = dict(
        path=os.path.abspath(path),
        score=_vec(m["score"]).astype(int),
        uncert=_vec(m.get("uncert", np.zeros(0))).astype(int),
        reflex=ref.astype(int),
        dFrames=_vec(m.get("dFrames")).astype(int),
        dTypes=_vec(m.get("dTypes")).astype(int),
        stimNames=_strlist(m.get("stimNames")),
        affNames=_strlist(m["affNames"]),
        refNames=_strlist(m["refNames"]),
        frameRate=float(_vec(m.get("frameRate", np.nan))[0]) if m.get("frameRate") is not None else np.nan,
        nUsed=int(_vec(m.get("nUsed", len(_vec(m["score"]))))[0]),
        offFrames=int(_vec(m.get("offFrames", 0))[0]) if np.size(m.get("offFrames", 0)) else 0,
        syncOff=float(_vec(m.get("syncOff", 0))[0]) if np.size(m.get("syncOff", 0)) else 0.0,
        mode=int(_vec(m.get("mode", 3))[0]) if np.size(m.get("mode", 3)) else 3,
        mouseID=_str(m.get("mouseID", "")), dayNo=_str(m.get("dayNo", "")),
        phase=_str(m.get("phase", "")), scorer=_str(m.get("scorer", "")),
        botFile=_str(m.get("botFile", "")), sidFile=_str(m.get("sidFile", "")),
    )
    _check_names(s)
    return s


def _check_names(s):
    aff, ref = s["affNames"], s["refNames"]
    if len(aff) < 5 or len(ref) < 2:
        raise ScoringError(f"{s['path']}: expected 5 affNames and 2 refNames, got {aff} / {ref}")
    for code, words in AFF_NAME_WORDS.items():
        if not any(w in aff[code].lower() for w in words):
            raise ScoringError(f"{s['path']}: affective code {code} is '{aff[code]}', expected "
                               f"{AFFECTIVE_CODES[code]}; the scorer's key map changed")
    for code, words in REF_NAME_WORDS.items():
        if not any(w in ref[code - 1].lower() for w in words):
            raise ScoringError(f"{s['path']}: reflex code {code} is '{ref[code - 1]}', expected "
                               f"{REFLEX_CODES[code]}")
    bad = set(np.unique(s["score"])) - {0, *AFFECTIVE_CODES}
    if bad:
        raise ScoringError(f"{s['path']}: score has unknown codes {sorted(bad)}")
    bad = set(np.unique(s["reflex"][:, 1])) - set(REFLEX_CODES)
    if bad:
        raise ScoringError(f"{s['path']}: reflexEvents has unknown codes {sorted(bad)}")


def canonical_stim(name):
    """Map a free-text stimulus name typed in the scorer to light/mild/heat/pin ('' if none)."""
    low = str(name).lower()
    hits = [k for k, words in STIM_WORDS.items() if any(w in low for w in words)]
    return hits[0] if len(hits) == 1 else ""


def _runs(mask):
    """(start, end) index pairs (inclusive, 0-based) of the True runs of a boolean array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def behaviour_events(s, recording_id, scorer=""):
    """behaviour_events rows (cam1 frames) from a loaded ScoringAB dict."""
    flags = ""
    if s["offFrames"] != 0 or s["syncOff"] != 0:
        # With hardware-triggered cameras the side view needs no offset; a non-zero typed
        # offset means marks made while watching the side view are shifted by it.
        flags = mt.add_flag(flags, f"side_offset_{s['offFrames']}")
        print(f"WARNING: {s['path']}: side-view offset offFrames={s['offFrames']} "
              f"(syncOff={s['syncOff']} s). Hardware-triggered cameras need 0.", file=sys.stderr)
    scorer = scorer or s["scorer"]
    rows = []
    n = min(s["nUsed"], s["score"].size)
    for code, name in AFFECTIVE_CODES.items():
        for a, b in _runs(s["score"][:n] == code):
            rows.append(dict(type=name, on_cam_frame=a + 1, off_cam_frame=b + 1))
    if s["uncert"].size:
        for a, b in _runs(s["uncert"][:n] != 0):
            rows.append(dict(type="uncertain", on_cam_frame=a + 1, off_cam_frame=b + 1))
    for frame, code in s["reflex"]:
        rows.append(dict(type=REFLEX_CODES[int(code)], on_cam_frame=int(frame),
                         off_cam_frame=int(frame)))
    df = pd.DataFrame(rows, columns=["type", "on_cam_frame", "off_cam_frame"])
    df.insert(0, "camera", CAMERA)
    df.insert(0, "recording_id", recording_id)
    df["scorer"] = scorer
    df["flags"] = flags
    out_of_range = (df["on_cam_frame"] < 1) | (df["off_cam_frame"] > n)
    df.loc[out_of_range, "flags"] = df.loc[out_of_range, "flags"].map(
        lambda f: mt.add_flag(f, "outside_scored_range"))
    df = df.sort_values(["on_cam_frame", "type"]).reset_index(drop=True)
    return mt.ensure_columns(df, mt.BEHAVIOUR_EVENTS_COLUMNS + ["flags"])


def video_deliveries(s, recording_id):
    """Delivery taps (cam1 frames) with the typed name mapped to a stimulus key."""
    names = s["stimNames"]
    canon = [canonical_stim(x) for x in names]
    dup = {c for c in canon if c and canon.count(c) > 1}
    rows = []
    for f, k in zip(s["dFrames"], s["dTypes"]):
        typed = names[k - 1] if 1 <= k <= len(names) else ""
        stim = canon[k - 1] if 1 <= k <= len(canon) else ""
        flags = ""
        if k == 0:
            flags = mt.add_flag(flags, "unknown_type")
        elif not stim:
            flags = mt.add_flag(flags, "unrecognised_name")
        if stim in dup:
            # two keys carry the same name (seen before: 'Mild touch' typed twice), so the
            # key cannot tell which stimulus this was
            flags = mt.add_flag(flags, "ambiguous_name")
            stim = ""
        if s["mode"] == 2 and s["syncOff"] != 0:
            flags = mt.add_flag(flags, "mode2_syncoff_shift")   # score_AB_dual_view.m line 391
        rows.append(dict(recording_id=recording_id, camera=CAMERA, cam_frame=int(f), key=int(k),
                         stim_name_typed=typed, stim_type=stim, flags=flags))
    return pd.DataFrame(rows, columns=["recording_id", "camera", "cam_frame", "key",
                                       "stim_name_typed", "stim_type", "flags"])


def _norm(text):
    """Lower-case letters and digits only, leading zeros dropped ('M01' == 'm1')."""
    s = "".join(ch for ch in str(text).lower() if ch.isalnum())
    return re.sub(r"\d+", lambda g: str(int(g.group())), s)


def check_identity(s, recording_id, cam1_video=None, ignore_metadata=False):
    """Tie the .mat to its recording. The scorer dialog is often left blank (and Phase defaults
    to 'Baseline'), so only filled fields are compared: a mouseID or dayNo that names another
    mouse or day, or a phase that says pre/post against the recording, raises ScoringError
    (unless ignore_metadata); a botFile that is not the manifest's cam1 video only warns, as
    videos are often renamed after scoring. A file with nothing scored raises too."""
    if s["nUsed"] <= 0:
        raise ScoringError(f"{s['path']}: nUsed = {s['nUsed']}: nothing was scored")
    if s["dFrames"].size == 0:
        raise ScoringError(f"{s['path']}: no delivery taps: the file cannot be placed on the "
                           "trials (and heat cannot be timed)")
    m = re.match(r"^(.+)_d(\d+)_(pre|post)$", str(recording_id))
    problems = []
    if m:
        mouse, day, assay = m.group(1), int(m.group(2)), m.group(3)
        mid = _norm(s["mouseID"])
        if mid and _norm(mouse) not in mid and mid not in _norm(mouse):
            problems.append(f"mouseID '{s['mouseID']}' is not mouse {mouse}")
        if str(s["dayNo"]).strip().isdigit() and int(s["dayNo"]) != day:
            problems.append(f"dayNo {s['dayNo']} is not day {day}")
        phase = str(s["phase"]).lower()
        said = {a for a in ("pre", "post") if a in phase}
        if len(said) == 1 and said != {assay}:
            problems.append(f"phase '{s['phase']}' is not assay {assay}")
    if problems:
        msg = f"{s['path']} does not look like {recording_id}: " + "; ".join(problems)
        if not ignore_metadata:
            raise ScoringError(msg + " (pass --ignore-metadata if the file is right and the "
                               "dialog fields are wrong)")
        print(f"WARNING: {msg} (--ignore-metadata)", file=sys.stderr)
    if s["botFile"] and cam1_video:
        stem = lambda p: _norm(os.path.splitext(os.path.basename(str(p).replace("\\", "/")))[0])
        if stem(s["botFile"]) != stem(cam1_video):
            print(f"WARNING: {s['path']}: scored on '{s['botFile']}', but the manifest's cam1 "
                  f"video is '{os.path.basename(cam1_video)}' (renamed? or another recording)",
                  file=sys.stderr)


def run(mat_path, recording_id, out_dir, scorer="", cam1_video=None, ignore_metadata=False):
    """Import one ScoringAB file; write behaviour_events.csv and video_deliveries.csv."""
    s = load_scoring(mat_path)
    check_identity(s, recording_id, cam1_video, ignore_metadata)
    be = behaviour_events(s, recording_id, scorer)
    vd = video_deliveries(s, recording_id)
    os.makedirs(out_dir, exist_ok=True)
    mt.replace_rows(os.path.join(out_dir, "behaviour_events.csv"), be, recording_id,
                    mt.BEHAVIOUR_EVENTS_COLUMNS + ["flags"])
    mt.replace_rows(os.path.join(out_dir, "video_deliveries.csv"), vd, recording_id,
                    list(vd.columns))
    return s, be, vd


def _fake_mat(path, n=600, reflex=None, score_runs=(), uncert_runs=(), names=None,
              aff=None, d_frames=(100, 400), d_types=(1, 3), n_used=None):
    score = np.zeros(n, np.uint8)
    for a, b, c in score_runs:
        score[a - 1:b] = c
    unc = np.zeros(n, np.uint8)
    for a, b in uncert_runs:
        unc[a - 1:b] = 1
    obj = lambda xs: np.array(list(xs), dtype=object)
    sio.savemat(path, dict(
        score=score.reshape(-1, 1), uncert=unc.reshape(-1, 1),
        reflexEvents=np.array(reflex if reflex is not None else np.zeros((0, 2)), dtype=np.uint16),
        dFrames=np.array(d_frames, np.uint16).reshape(-1, 1), dTypes=np.array(d_types, np.uint8).reshape(-1, 1),
        stimNames=obj(names or ["Light touch", "Mild touch", "Heat", "Pin prick"]),
        affNames=obj(aff or ["None", "Paw attending", "Licking or biting",
                             "Sustained lifting / guarding", "Escape / rearing"]),
        affKeys=obj("asdf"), refNames=obj(["Paw withdrawal", "Flinch / flick"]),
        frameRate=20.0, nUsed=n if n_used is None else n_used, offFrames=0, syncOff=0, mode=3,
        mouseID="T01", dayNo="1",
        phase="pre", botFile="b.avi", sidFile="s.avi"))
    return path


def selftest():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    with tempfile.TemporaryDirectory() as d:
        p = _fake_mat(os.path.join(d, "ScoringAB_a.mat"),
                      reflex=[[105, 1], [108, 2], [403, 1]],
                      score_runs=[(120, 150, 2), (151, 160, 3), (500, 500, 1)],
                      uncert_runs=[(300, 310)])
        s, be, vd = run(p, "T01_d1_pre", d, scorer="HL")
        got = [(r.type, r.on_cam_frame, r.off_cam_frame) for r in be.itertuples()]
        want = [("withdrawal", 105, 105), ("flinch", 108, 108), ("licking_biting", 120, 150),
                ("guarding", 151, 160), ("uncertain", 300, 310), ("withdrawal", 403, 403),
                ("attending", 500, 500)]
        check(got == want, f"events {got}")
        check((be.camera == "cam1").all() and (be.scorer == "HL").all(), "camera and scorer")
        check(be.on_isx_frame.isna().all(), "isx columns are left for master_timeline")
        check(list(vd.stim_type) == ["light", "heat"] and list(vd.cam_frame) == [100, 400],
              "deliveries mapped by typed name")
        on_disk = pd.read_csv(os.path.join(d, "behaviour_events.csv"))
        check(list(on_disk.columns[:12]) == mt.BEHAVIOUR_EVENTS_COLUMNS, "contract column order")

        # a single reflex row is squeezed to shape (2,) by loadmat; it must still load
        p1 = _fake_mat(os.path.join(d, "ScoringAB_b.mat"), reflex=[[42, 1]])
        s1 = load_scoring(p1)
        check(s1["reflex"].shape == (1, 2) and s1["reflex"][0, 0] == 42, "single reflex row")
        p0 = _fake_mat(os.path.join(d, "ScoringAB_c.mat"), reflex=None)
        check(load_scoring(p0)["reflex"].shape == (0, 2), "no reflex rows")

        # names typed twice -> ambiguous, never guessed
        p2 = _fake_mat(os.path.join(d, "ScoringAB_d.mat"),
                       names=["heat", "Mild touch", "pin prick", "mild "], d_types=(2, 4))
        vd2 = video_deliveries(load_scoring(p2), "T01_d1_pre")
        check((vd2.stim_type == "").all() and vd2["flags"].str.contains("ambiguous_name").all(),
              "duplicate names are flagged ambiguous")

        # a changed key map must fail loudly
        p3 = _fake_mat(os.path.join(d, "ScoringAB_e.mat"),
                       aff=["None", "Licking", "Paw attending", "Guarding", "Escape"])
        try:
            load_scoring(p3)
            check(False, "a changed affective key map must raise")
        except ScoringError:
            pass
        mtime = os.path.getmtime(p)
        load_scoring(p)
        check(os.path.getmtime(p) == mtime, "the .mat file is not modified")

        # the .mat must belong to the recording, and must hold something scored
        for what, mat, rid in (
                ("a file whose mouseID/dayNo/phase name another recording", p, "T02_d2_post"),
                ("nUsed = 0 (nothing scored)",
                 _fake_mat(os.path.join(d, "ScoringAB_f.mat"), n_used=0), "T01_d1_pre"),
                ("no delivery taps", _fake_mat(os.path.join(d, "ScoringAB_g.mat"), d_frames=(),
                                               d_types=()), "T01_d1_pre")):
            try:
                run(mat, rid, os.path.join(d, "o"))
                check(False, f"{what} must raise")
            except ScoringError:
                pass
        s5, _, _ = run(p, "T02_d2_post", os.path.join(d, "o2"), ignore_metadata=True)
        check(s5["mouseID"] == "T01", "--ignore-metadata imports a file whose dialog fields are wrong")
        run(p, "T01_d1_pre", os.path.join(d, "o3"), cam1_video="other.avi")   # warns only
        check(os.path.exists(os.path.join(d, "o3", "behaviour_events.csv")),
              "a botFile that differs from cam1_video only warns")

    if failures:
        print("scoring_import selftest FAILED:")
        for f_ in failures:
            print("  -", f_)
        return 1
    print("scoring_import selftest passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mat", help="ScoringAB_*.mat (read-only)")
    ap.add_argument("--recording-id")
    ap.add_argument("--out-dir")
    ap.add_argument("--scorer", default="", help="scorer ID (the scorer GUI leaves it blank)")
    ap.add_argument("--cam1-video", help="the manifest's cam1 video, compared with the file's botFile")
    ap.add_argument("--ignore-metadata", action="store_true",
                    help="import although the file's mouseID/dayNo/phase name another recording")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not (args.mat and args.recording_id and args.out_dir):
        ap.error("--mat, --recording-id and --out-dir are required")
    folder = os.path.basename(os.path.dirname(os.path.abspath(args.mat)))
    if "SYNTHETIC" in folder.upper():
        # videos\output_day2_SYNTHETIC reuses the real file names; never mix it with real data
        print(f"WARNING: {args.mat} is in the synthetic folder '{folder}'", file=sys.stderr)
    try:
        s, be, vd = run(args.mat, args.recording_id, args.out_dir, args.scorer, args.cam1_video,
                        args.ignore_metadata)
    except ScoringError as e:
        print(f"SCORING IMPORT FAILED: {e}", file=sys.stderr)
        return 1
    print(f"{os.path.basename(args.mat)}: {s['nUsed']} frames scored, "
          f"{len(be)} behaviour events ({be.type.value_counts().to_dict()}), "
          f"{len(vd)} deliveries -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
