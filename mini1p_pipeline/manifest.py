"""Session manifest: write a template, load and validate a manifest.

The manifest (CONTRACTS.md section 3, ``manifest.csv``) is written by the user and has one
row per recording (one Inscopix recording = one logger session = one video per camera).
Every other module takes its file paths and assay metadata from here, so this loader is
strict: wrong types, unknown values or a recording_id that does not follow
``<mouse>_d<day>_<assay>`` are errors, not warnings.

Checks
  - required columns present; unknown extra columns are allowed (modules may add columns)
  - recording_id == <mouse_id>_d<day>_<assay>, unique across the file
  - day in {1, 2}; assay in {pre, post}; sex in {F, M}
  - arm == 'none' for every pre; a blinded code (not 'none') for every post
  - stim_order lists light, mild, heat and pin exactly once each
  - mild_g, assay_t0_s numeric; injection_time a clock time or ISO date-time
  - optional pulse_to_frame_offset is an integer
  - every path column is resolved relative to the manifest's folder; missing files are
    reported (as errors only with --require-files, because files may not exist yet)

Optional columns understood by this pipeline (added after the contract columns):
  pulse_to_frame_offset  K of CONTRACTS.md section 1 for the rig (fixed by a test recording)
  isx_timestamps         IDPS frame-timestamp CSV of the raw .isxd (inscopix_day.m writes
                         <recording_id>_raw_tsc.csv); if empty, the frame table's tsc_us is used
  extract_frame_table    <mouse>_d<day>_frame_table.csv from extract_1p.m
  gpio_csv               IDPS GPIO CSV export, when the gpio column holds the raw .gpio

Usage
  python manifest.py template [--out manifest_template.csv]
  python manifest.py check MANIFEST.csv [--require-files]
  python manifest.py --selftest
"""

import argparse
import csv
import os
import re
import sys
import tempfile

MANIFEST_COLUMNS = [
    "recording_id", "mouse_id", "sex", "day", "assay", "arm", "stim_order", "mild_g",
    "injection_time", "assay_t0_s", "isxd", "gpio", "imu", "pressure_csv", "pressure_json",
    "cam1_video", "cam2_video", "scoring_mat", "notes",
]
OPTIONAL_COLUMNS = ["pulse_to_frame_offset", "isx_timestamps", "extract_frame_table", "gpio_csv"]
PATH_COLUMNS = [
    "isxd", "gpio", "imu", "pressure_csv", "pressure_json", "cam1_video", "cam2_video",
    "scoring_mat", "isx_timestamps", "extract_frame_table", "gpio_csv",
]
STIM_KEYS = ("light", "mild", "heat", "pin")
ASSAYS = ("pre", "post")
DAYS = (1, 2)
SEXES = ("F", "M")

# Words that would reveal the drug arm. The post arm must stay a blinded code until
# unblinding, so a manifest that names the treatment is flagged.
UNBLINDED_WORDS = ("vehicle", "veh", "saline", "sbi", "drug", "morphine", "none")

_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?$")
_MOUSE_RE = re.compile(r"^[A-Za-z0-9]+$")


class ManifestError(ValueError):
    """Raised when a manifest violates the contract."""


def recording_id_for(mouse_id, day, assay):
    """The contract's recording_id: <mouse>_d<day>_<assay>."""
    return f"{mouse_id}_d{int(day)}_{assay}"


def parse_stim_order(text):
    """Return the four stimulus keys in delivery order, or raise ValueError."""
    keys = [k.strip().lower() for k in str(text).split(",") if k.strip()]
    if sorted(keys) != sorted(STIM_KEYS) or len(keys) != len(STIM_KEYS):
        raise ValueError(f"stim_order must list {', '.join(STIM_KEYS)} once each, got '{text}'")
    return keys


def _resolve(base_dir, path):
    if not path:
        return ""
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.normpath(path)


def _float_or_none(text):
    if text is None or str(text).strip() == "":
        return None
    return float(text)


class Manifest:
    """Validated manifest: .rows (list of dicts with typed values), .errors, .warnings,
    .missing_files (list of (recording_id, column, path))."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.base_dir = os.path.dirname(self.path)
        self.rows = []
        self.errors = []
        self.warnings = []
        self.missing_files = []
        self.columns = []

    def row(self, recording_id):
        for r in self.rows:
            if r["recording_id"] == recording_id:
                return r
        raise ManifestError(f"recording_id '{recording_id}' is not in {self.path}")

    def ok(self, require_files=False):
        return not self.errors and not (require_files and self.missing_files)

    def report(self):
        lines = [f"manifest {self.path}: {len(self.rows)} row(s)"]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  warning {w}" for w in self.warnings]
        lines += [f"  missing {rid}: {col} -> {p}" for rid, col, p in self.missing_files]
        if not (self.errors or self.warnings or self.missing_files):
            lines.append("  all checks passed")
        return "\n".join(lines)


def _validate_row(raw, n, man):
    """Type-check one CSV row. Appends problems to man; returns the typed row or None."""
    errs = []
    r = {k: (v or "").strip() for k, v in raw.items() if k is not None}
    tag = f"row {n} ({r.get('recording_id', '?')})"

    mouse = r.get("mouse_id", "")
    if not _MOUSE_RE.match(mouse):
        # '_' separates the parts of recording_id, so it cannot appear inside mouse_id
        errs.append(f"mouse_id '{mouse}' must be letters/digits only")

    sex = r.get("sex", "").upper()
    if sex not in SEXES:
        errs.append(f"sex '{r.get('sex', '')}' must be one of {SEXES}")

    day = None
    try:
        day = int(float(r.get("day", "")))
        if float(r["day"]) != day or day not in DAYS:
            raise ValueError
    except ValueError:
        errs.append(f"day '{r.get('day', '')}' must be one of {DAYS}")

    assay = r.get("assay", "").lower()
    if assay not in ASSAYS:
        errs.append(f"assay '{r.get('assay', '')}' must be one of {ASSAYS}")

    arm = r.get("arm", "")
    if assay == "pre" and arm.lower() != "none":
        errs.append(f"arm must be 'none' for a pre assay, got '{arm}'")
    if assay == "post":
        if arm == "" or arm.lower() == "none":
            errs.append("arm must be a blinded code (e.g. A / B) for a post assay")
        elif not re.match(r"^[A-Za-z0-9-]+$", arm):
            errs.append(f"arm '{arm}' must be a short code (letters, digits, '-')")
        elif any(w in arm.lower() for w in UNBLINDED_WORDS):
            man.warnings.append(f"{tag}: arm '{arm}' looks unblinded; use a code until unblinding")

    stim_order = None
    try:
        stim_order = parse_stim_order(r.get("stim_order", ""))
    except ValueError as e:
        errs.append(str(e))

    mild_g = None
    try:
        mild_g = _float_or_none(r.get("mild_g"))
        if mild_g is None:
            man.warnings.append(f"{tag}: mild_g is empty (mild-touch filament grams)")
        elif mild_g <= 0:
            errs.append(f"mild_g must be > 0, got {mild_g}")
    except ValueError:
        errs.append(f"mild_g '{r.get('mild_g')}' is not a number")

    inj = r.get("injection_time", "")
    if inj and not (_TIME_RE.match(inj) or _DATETIME_RE.match(inj)):
        errs.append(f"injection_time '{inj}' must be HH:MM[:SS] or YYYY-MM-DD HH:MM[:SS]")
    if assay == "pre" and inj:
        man.warnings.append(f"{tag}: injection_time is set on a pre (no-injection) assay")
    if assay == "post" and not inj:
        man.warnings.append(f"{tag}: injection_time is empty on a post assay")

    t0 = 0.0
    try:
        t0 = _float_or_none(r.get("assay_t0_s"))
        t0 = 0.0 if t0 is None else t0
        if t0 < 0:
            errs.append(f"assay_t0_s must be >= 0 (seconds after isx_frame 0), got {t0}")
    except ValueError:
        errs.append(f"assay_t0_s '{r.get('assay_t0_s')}' is not a number")

    k = None
    if r.get("pulse_to_frame_offset", "") != "":
        try:
            k = int(r["pulse_to_frame_offset"])
        except ValueError:
            errs.append(f"pulse_to_frame_offset '{r['pulse_to_frame_offset']}' must be an integer")

    rid = r.get("recording_id", "")
    if day is not None and assay in ASSAYS and mouse:
        want = recording_id_for(mouse, day, assay)
        if rid != want:
            errs.append(f"recording_id '{rid}' must be '{want}'")

    man.errors += [f"{tag}: {e}" for e in errs]
    if errs:
        return None

    out = dict(r)
    out.update(recording_id=rid, mouse_id=mouse, sex=sex, day=day, assay=assay, arm=arm,
               stim_order=stim_order, mild_g=mild_g, injection_time=inj, assay_t0_s=t0,
               pulse_to_frame_offset=k, notes=r.get("notes", ""))
    for col in PATH_COLUMNS:
        out[col] = _resolve(man.base_dir, r.get(col, ""))
        if out[col] and not os.path.exists(out[col]):
            man.missing_files.append((rid, col, out[col]))
    return out


def load_manifest(path):
    """Read and validate a manifest. Never raises for content problems: inspect
    .errors / .warnings / .missing_files, or call load_manifest_strict()."""
    man = Manifest(path)
    with open(path, newline="", encoding="utf-8-sig") as f:   # Excel writes a BOM
        reader = csv.DictReader(f)
        man.columns = [c.strip() for c in (reader.fieldnames or [])]
        reader.fieldnames = man.columns
        missing = [c for c in MANIFEST_COLUMNS if c not in man.columns]
        if missing:
            man.errors.append(f"missing column(s): {', '.join(missing)}")
            return man
        for n, raw in enumerate(reader, start=2):              # row 1 is the header
            if not any((v or "").strip() for v in raw.values() if isinstance(v, str)):
                continue                                         # blank line
            row = _validate_row(raw, n, man)
            if row is not None:
                man.rows.append(row)

    seen = {}
    for r in man.rows:
        if r["recording_id"] in seen:
            man.errors.append(f"recording_id '{r['recording_id']}' appears twice")
        seen[r["recording_id"]] = r
    # Crossover: each mouse should get a different post arm on day 1 and day 2.
    post_arms = {}
    for r in man.rows:
        if r["assay"] == "post":
            post_arms.setdefault(r["mouse_id"], {})[r["day"]] = r["arm"]
    for mouse, arms in post_arms.items():
        if len(arms) == 2 and len(set(arms.values())) == 1:
            man.warnings.append(f"{mouse}: same post arm '{arms[1]}' on day 1 and day 2 "
                                "(crossover expects different arms)")
    return man


def load_manifest_strict(path, require_files=False):
    """Load a manifest and raise ManifestError if it has errors."""
    man = load_manifest(path)
    if not man.ok(require_files):
        raise ManifestError(man.report())
    return man


TEMPLATE_ROWS = [
    dict(recording_id="M01_d1_pre", mouse_id="M01", sex="F", day="1", assay="pre", arm="none",
         stim_order="light,mild,heat,pin", mild_g="1.4", injection_time="", assay_t0_s="0",
         isxd="inscopix/M01_d1/2026-10-20-10-00-00_video.isxd",
         gpio="inscopix/M01_d1/2026-10-20-10-00-00_gpio.csv",
         imu="inscopix/M01_d1/2026-10-20-10-00-00.imu",
         pressure_csv="pressure/pain_cohort1/M01/2026-10-20/M01_2026-10-20_S01_pressure.csv",
         pressure_json="pressure/pain_cohort1/M01/2026-10-20/M01_2026-10-20_S01_info.json",
         cam1_video="cameras/M01_d1_pre_cam1.avi", cam2_video="cameras/M01_d1_pre_cam2.avi",
         scoring_mat="scoring/ScoringAB_M01_d1_pre_cam1.mat",
         notes="example row: replace every value; arm is none for pre",
         pulse_to_frame_offset="0",
         isx_timestamps="inscopix/M01_d1/M01_d1_pre_raw_tsc.csv",
         extract_frame_table="extract/M01_d1_frame_table.csv"),
    dict(recording_id="M01_d1_post", mouse_id="M01", sex="F", day="1", assay="post", arm="A",
         stim_order="light,mild,heat,pin", mild_g="1.4", injection_time="10:41", assay_t0_s="0",
         isxd="inscopix/M01_d1/2026-10-20-10-52-00_video.isxd",
         gpio="inscopix/M01_d1/2026-10-20-10-52-00_gpio.csv",
         imu="inscopix/M01_d1/2026-10-20-10-52-00.imu",
         pressure_csv="pressure/pain_cohort1/M01/2026-10-20/M01_2026-10-20_S02_pressure.csv",
         pressure_json="pressure/pain_cohort1/M01/2026-10-20/M01_2026-10-20_S02_info.json",
         cam1_video="cameras/M01_d1_post_cam1.avi", cam2_video="cameras/M01_d1_post_cam2.avi",
         scoring_mat="scoring/ScoringAB_M01_d1_post_cam1.mat",
         notes="example row: arm is the blinded code (A/B), unblinded later",
         pulse_to_frame_offset="0",
         isx_timestamps="inscopix/M01_d1/M01_d1_post_raw_tsc.csv",
         extract_frame_table="extract/M01_d1_frame_table.csv"),
]


def write_manifest(path, rows):
    """Write rows (dicts) with the contract columns first, then any extra columns."""
    extra = []
    for r in rows:
        extra += [k for k in r if k not in MANIFEST_COLUMNS and k not in extra]
    cols = MANIFEST_COLUMNS + extra
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else
                            ",".join(r[c]) if isinstance(r.get(c), list) else r.get(c))
                        for c in cols})
    return path


def write_template(path):
    """Write manifest_template.csv: one example row per assay for one mouse-day."""
    return write_manifest(path, TEMPLATE_ROWS)


def selftest():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    with tempfile.TemporaryDirectory() as d:
        p = write_template(os.path.join(d, "manifest_template.csv"))
        man = load_manifest(p)
        check(not man.errors, f"template should validate: {man.errors}")
        check(len(man.rows) == 2, "template should have 2 rows")
        check(len(man.missing_files) > 0, "template paths do not exist; they must be reported")
        r = man.row("M01_d1_post")
        check(r["stim_order"] == ["light", "mild", "heat", "pin"], "stim_order parsed")
        check(r["pulse_to_frame_offset"] == 0, "pulse_to_frame_offset parsed as int")
        check(os.path.isabs(r["isxd"]) and r["isxd"].startswith(os.path.normpath(d)),
              "paths resolved relative to the manifest folder")

        # make every path of row 1 exist -> no missing files for it
        for col in PATH_COLUMNS:
            fp = man.rows[0][col]
            if not fp:
                continue
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            open(fp, "w").close()
        man = load_manifest(p)
        check(all(rid != "M01_d1_pre" for rid, _, _ in man.missing_files),
              "existing files must not be reported missing")

        bad_cases = {
            "arm": dict(arm="A"),                                   # pre with an arm
            "stim_order": dict(stim_order="light,mild,mild,pin"),
            "recording_id": dict(recording_id="M01_d2_pre"),
            "day": dict(day="3"),
            "assay": dict(assay="baseline"),
            "sex": dict(sex="X"),
            "mild_g": dict(mild_g="heavy"),
            "injection_time": dict(injection_time="soon"),
            "assay_t0_s": dict(assay_t0_s="-5"),
            "mouse_id": dict(mouse_id="M_01", recording_id="M_01_d1_pre"),
            "pulse_to_frame_offset": dict(pulse_to_frame_offset="0.5"),
        }
        for name, change in bad_cases.items():
            row = dict(TEMPLATE_ROWS[0], **change)
            q = write_manifest(os.path.join(d, f"bad_{name}.csv"), [row])
            check(load_manifest(q).errors, f"bad {name} must be an error")

        post_none = dict(TEMPLATE_ROWS[1], arm="none")
        check(load_manifest(write_manifest(os.path.join(d, "b1.csv"), [post_none])).errors,
              "post with arm none must be an error")
        unblind = dict(TEMPLATE_ROWS[1], arm="vehicle")
        m = load_manifest(write_manifest(os.path.join(d, "b2.csv"), [unblind]))
        check(not m.errors and any("unblinded" in w for w in m.warnings),
              "unblinded arm must be a warning")
        dup = load_manifest(write_manifest(os.path.join(d, "b3.csv"),
                                           [TEMPLATE_ROWS[0], TEMPLATE_ROWS[0]]))
        check(any("twice" in e for e in dup.errors), "duplicate recording_id is an error")
        q = os.path.join(d, "b4.csv")
        with open(q, "w") as f:
            f.write("recording_id,mouse_id\nM01_d1_pre,M01\n")
        check(any("missing column" in e for e in load_manifest(q).errors),
              "missing columns are an error")
        try:
            load_manifest_strict(os.path.join(d, "bad_day.csv"))
            check(False, "load_manifest_strict must raise on errors")
        except ManifestError:
            pass

    if failures:
        print("manifest selftest FAILED:")
        for f_ in failures:
            print("  -", f_)
        return 1
    print("manifest selftest passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="run the built-in tests and exit")
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("template", help="write manifest_template.csv")
    t.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "manifest_template.csv"))
    c = sub.add_parser("check", help="validate a manifest and list missing files")
    c.add_argument("manifest")
    c.add_argument("--require-files", action="store_true",
                   help="treat missing files as errors (use right before processing)")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.cmd == "template":
        print("wrote", write_template(args.out))
        return 0
    if args.cmd == "check":
        man = load_manifest(args.manifest)
        print(man.report())
        return 0 if man.ok(args.require_files) else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
