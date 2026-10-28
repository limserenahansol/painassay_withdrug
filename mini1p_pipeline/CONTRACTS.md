# mini1p pain pipeline — data contracts

Every module in this folder reads and writes the tables defined here. A module may add
columns; it may never rename or reinterpret one. All tables are CSV (UTF-8, comma), one row
per item, with a header row.

## 1. Clocks and indices — the one rule

The Inscopix DAQ **SYNC** BNC output sends one 5 V pulse (10 ms) at the end of exposure of
every recorded miniscope frame (IDAS 2.5.0 manual p.23). The same SYNC line, through a BNC
splitter, **triggers both Imaging Source cameras** and is **counted by the pressure logger**
(Arduino interrupt on rising edges; `ttl_count`). Therefore:

```
SYNC pulse n (1-based) ends isx_frame n-1+K (0-based)
                       triggers cam1 frame n and cam2 frame n
                       makes the logger's ttl_count become n
-> cam frame n and logger rows with ttl_count == n fall during isx_frame n+K   (see 1a)
```

`K` is a per-rig constant, expected 0, fixed once with a test recording (count_check).
It is stored per recording as `pulse_to_frame_offset` and never assumed silently.

**Only K = 0 is analysable (decided 2026-10-19).** A non-zero K would mean one count-stamped
stream started late (for example the logger missed the first K pulses). Then
`n_ttl = n_frames_isx - K` (not `+ K`, as an earlier version of this file said), and, because
the cameras are triggered independently of the logger, camera frame n and logger count n
would no longer be the same pulse, which one constant cannot repair. `run_pipeline` refuses
K != 0. All four 2024 recordings with a GPIO export measured K = 0 (`k_from_gpio`). Where this
file writes `n_frames_isx + K`, read it with K = 0.

- `isx_frame`: 0-based index of the raw IDAS movie, **including dropped frames** (IDPS
  convention). This is the master index of a recording.
- `t_master_s`: seconds since isx_frame 0 of that recording, from the per-frame TSC timestamps
  when available, else `isx_frame * period_s` (flag `t_source`).
- Camera frame numbers are 1-based (frame 1 = first trigger), matching pulse numbers.
- Arduino time (`t_s` in the logger CSV) is mapped to `t_master_s` by a robust linear fit of
  pulse-bracket midpoints against frame times (`a + b * t_s`; store ppm and residual).
- Never use `pc_time`, video file-name times or nominal frame rates for sync.

### 1a. Count-stamped streams — one rule for cameras AND the logger (revised 2026-10-19)

Pulse n is emitted at the END of exposure of isx_frame n-1+K, and the IDAS per-frame TSC
timestamp marks that same moment (2024 nVista3 data: the first BNC Sync edge falls 0.0-0.6 ms
after the movie start, count_check). So `t_master_s(isx_frame k)` = time of pulse k+1-K, and
the exposure of frame k spans `(t_master_s(k) - exposure_s, t_master_s(k)]`.

Anything stamped with a pulse COUNT happened after pulse n and before pulse n+1, i.e. during
the exposure of the NEXT frame:

```
logger rows with ttl_count == n   -> during isx_frame  n + K
camera frame n (triggered by n)   -> during isx_frame  n + K
```

One function does this everywhere: `count_to_isx_frame(n) = n + K` (in master_timeline).
Consequences, by construction: isx_frame 0 has no camera frame and no logger sample in a
TTL-started session; the last camera frame N (pulse N ends the last miniscope frame N-1)
falls after the movie ends. `cam_lead_frames` is superseded by this rule and kept only as a
documentation column equal to 1.

Sub-frame times (seconds) of logger events come from the Arduino->master fit of pulse
times: pulse n Arduino time <-> `t_master_s(isx_frame n-1+K)`.

If the DAQ sends no pulse for a dropped frame, the count mapping is wrong after the first
drop; count_check must FAIL (not repair) whenever `n_ttl != n_frames_isx + K`, until a
test recording with a dropped frame shows what the hardware does.

Filled gaps: `trial` is 1-3 within a block; `event_id` 001-060 follow schedule order (a later
video row for the same trial keeps its id) and extras are numbered from 061; logger file names
are not recording ids, so `recording_id` always comes from the manifest.

## 2. Keys

| key | meaning |
|---|---|
| `mouse_id` | e.g. M01 |
| `day` | 1 or 2 (crossover day) |
| `assay` | `pre` (no injection) or `post` (after injection + wait) |
| `arm` | `none` for every `pre`; blinded code for `post` (e.g. A / B), unblinded later |
| `recording_id` | `<mouse>_d<day>_<assay>`, e.g. `M01_d1_pre`; one Inscopix recording = one logger session = one video per camera |
| `event_id` | `<recording_id>_e<NNN>` |

## 3. Tables

Files and recordings (added 2026-10-19): `recordings.csv`, `camera_frames.csv`,
`behaviour_events.csv` and `video_deliveries.csv` may hold several recordings; a module
replaces only its own recording's rows. `pressure_frames.csv`, `stim_events.csv` and
`frames.csv` hold ONE recording: a module refuses to write one into a folder whose copy
belongs to another recording. `run_pipeline` uses one folder per recording and refuses a folder
that holds an earlier run's outputs unless `--overwrite` deletes them first.

### manifest.csv (written by the user, one row per recording)

`recording_id, mouse_id, sex, day, assay, arm, stim_order, mild_g, injection_time, assay_t0_s,
isxd, gpio, imu, pressure_csv, pressure_json, cam1_video, cam2_video, scoring_mat, notes`

- `stim_order`: comma list of the four stimulus keys in delivery order, e.g.
  `light,mild,heat,pin`. Keys are fixed: `light`, `mild`, `heat`, `pin`.
- `assay_t0_s`: `t_master_s` of the start of the 5-min baseline (default 0).
- Paths may be relative to the manifest's folder. Empty = not available.

### schedule (derived, not a file)

From `assay_t0_s` and `stim_order`: baseline 0-300 s; stimulus slot 1 300-600 s; rest
600-660; slot 2 660-960; rest 960-1020; slot 3 1020-1320; rest 1320-1380; slot 4 1380-1680
(28 min). Within a slot, block b = 1..5 starts at slot_start + 60(b-1); trials at block
start + 0, 15, 30 s. 15 trials per stimulus, 60 per assay.

### recordings.csv (count_check + master_timeline)

`recording_id, n_frames_isx, n_dropped, n_trimmed_early, period_s, fps, n_ttl,
n_gpio_sync_edges, n_cam1_frames, n_cam2_frames, pulse_to_frame_offset, cam_lead_frames,
arduino_fit_a, arduino_fit_b, arduino_fit_ppm, arduino_fit_resid_ms, count_check, notes`

`count_check` = PASS only if `n_ttl == n_frames_isx + K == n_cam1_frames == n_cam2_frames`
(and `== n_gpio_sync_edges` when the GPIO export exists); otherwise FAIL with the reason.

### pressure_frames.csv (pressure_sync)

`recording_id, pulse, isx_frame, t_arduino_s, force_mean, force_max, n_samples`
One row per SYNC pulse; force = `corrected` with a slowly varying baseline removed (counts).

### stim_events.csv (pressure_sync, then master_timeline)

`event_id, recording_id, stim_type, slot, block, trial, source, t_expected_master_s,
onset_t_arduino_s, onset_t_master_s, onset_isx_frame, onset_cam_frame, threshold_t_master_s,
peak_t_master_s, peak_force, offset_t_master_s, duration_s, match_residual_s, flags`

- `source`: `pressure` (detected from the logger), `video` (scored on the cameras),
  `expected` (schedule only — no detection yet).
- `flags`: semicolon list, e.g. `missing;extra;outside_slot;weak;pulse_ambiguous;duplicate`.

### camera_frames.csv (camera_sync)

`recording_id, camera, cam_frame, pulse, isx_frame, t_master_s, method, gap_before`
`method`: `hw_trigger` (frame n = pulse n, counts verified) or `fit` (fallback).

### behaviour_events.csv (scoring import)

`recording_id, camera, type, on_cam_frame, off_cam_frame, on_t_master_s, off_t_master_s,
on_isx_frame, off_isx_frame, event_id, latency_s, scorer`
`type` in `withdrawal, flinch, attending, licking_biting, guarding, escape, uncertain`.

### frames.csv (master_timeline; the single input to analysis)

`recording_id, isx_frame, t_master_s, t_assay_s, is_dropped, extract_row, epoch, slot,
stim_type, block, trial, stim_on, event_id, force_max, cam1_frame, cam2_frame,
affective_state, reflex, motion_cam1, motion_cam2`

- `epoch` in `pre_baseline, baseline, stim, rest, post_end`; `stim_on` = 1 from contact
  onset to offset.
- `extract_row` = row of the EXTRACT trace matrix for this frame, empty if dropped/trimmed.

### movie files before EXTRACT (inscopix_day.m, default mode `lossless_mc`, decided 2026-10-19)

Before EXTRACT only two things happen: a lossless copy and motion correction. EXTRACT does
dF/F, background removal, spatial high-pass, masking and detection binning itself.

Per recording (`<recording_id>`):
- `_raw.h5:/mov` — uint16, MATLAB `[H W N]` (h5py `(N, W, H)`), the full sensor frame, every
  valid frame in raw order, pixel values unchanged. Invalid frames (dropped, cropped, blank)
  are left out, never zero-filled.
- `_raw_frames.csv` — one row per raw frame: `recording_id, assay, isx_frame, valid,
  invalid_reason, h5_row, tsc_us, checksum_sum, checksum_wsum`. `h5_row` is 1-based along
  /mov's frame dimension, empty if invalid. `checksum_sum` = sum of the frame's pixel values;
  `checksum_wsum` = sum of i * v(i), i = 1..H*W in MATLAB column-major order (= numpy
  `ds[k].ravel()` order). Both are exact uint64 integers.
- `_mc.h5:/mov` — the raw values translated by the motion correction (single, or uint16 if
  asked), same rows: `h5_row` of `_raw_frames.csv` is valid for both files.
- `_mc_shifts.csv` — one row per h5 row: `recording_id, isx_frame, h5_row, x_shift_px,
  y_shift_px, corr_before, corr_after`. Shifts are what was applied to the frame (full-resolution
  px, x = columns, y = rows; the tissue moved by minus that).

Per day (`<mouse>_d<day>`):
- `_mc.h5:/mov` — the movie EXTRACT reads: an HDF5 virtual dataset concatenating the
  recordings' `_mc.h5:/mov` in manifest order (pre then post). It stores no pixels; the
  `_mc.h5` files must stay next to it. `extract_1p` refuses it if a source is missing.
- `_mc_frames.csv` — one row per raw frame of every recording: `recording_id, assay,
  isx_frame, valid, invalid_reason, h5_row, rec_h5_row, tsc_us`. `h5_row` indexes the day
  movie; `rec_h5_row` indexes that recording's `_raw.h5` / `_mc.h5`.

Mode `idps_preprocess` (the earlier default) writes the day `_mc.h5` as real data (IDPS 2x
binned, trimmed, cropped) and its `_mc_frames.csv` has `pp_frame` instead of `rec_h5_row`.

### extract outputs (inscopix_day.m / extract_1p.m)

Per day: `<mouse>_d<day>_extract.mat` with `spatial_weights` (H x W x C, on the input grid:
full resolution in mode `lossless_mc`), `temporal_raw`
(T x C, unconstrained least squares on fixed footprints — keeps decreases), `temporal_extract`
(EXTRACT output), `frame_table` (T rows: `recording_id, isx_frame`), `config`, `fps`.
Plus `<mouse>_d<day>_frame_table.csv` with the same rows.

## 4. Conventions

- Python 3.13 (system) with numpy, pandas, scipy, cv2. MATLAB R2024b with the IDPS 1.9.6
  `+isx` API (`C:\Program Files\Inscopix\Data Processing`) and EXTRACT
  (`C:\Users\hsollim\OneDrive\Documents\MATLAB\MATLAB\EXTRACT-public`). Call
  `mini2p_toolbox_paths()` first — it strips ActSort, which shadows ~20 EXTRACT helpers.
- No hard-coded drive letters, frame rates or mouse lists: everything comes from the manifest
  and the files.
- Every module has `--selftest` (runs on bundled or synthetic data) and fails loudly on a
  contract violation.
- Hand-scored data are read-only. Corrections go to new files.
