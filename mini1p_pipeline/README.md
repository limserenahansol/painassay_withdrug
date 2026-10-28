# mini1p pain pipeline

> **Start here: `PIPELINE_mini1p.md`** (2026-10-19). The pressure logger is no longer used.
> The MATLAB path `RUN_mini1p_20261019.m` -> `run_mini1p_day` does movie, EXTRACT
> parameter sweep, EXTRACT, cell review, merge with the video scoring and the mini2p
> classification. The Python modules below (count_check, master_timeline, ...) remain for
> the logger / GPIO count checks; they are not needed for that path.

Analysis pipeline for Inscopix nVista (mini1p) calcium imaging during the acute pain assay
in Ntsr1-Cre mice. Four streams are recorded together: the miniscope movie, two Imaging
Source behaviour cameras, and the Arduino pressure logger. The Inscopix DAQ SYNC pulse
triggers both cameras and is counted by the logger, so every stream can be placed on one
index, the miniscope frame (`isx_frame`). The result is one `frames.csv` per recording:
the single input to analysis.

`CONTRACTS.md` is binding. It defines every table, column and key, and the index rule.
Read it before changing a module.

## The sync rule (CONTRACTS.md 1a)

```
SYNC pulse n (1-based) ends isx_frame n-1+K (0-based); that frame's TSC timestamp is the pulse time
logger rows with ttl_count == n  -> fall during isx_frame n+K
camera frame n (triggered by n)  -> falls during isx_frame n+K
count check: n_ttl == n_frames_isx + K == n_cam1_frames == n_cam2_frames (== GPIO edges)
```

- `master_timeline.count_to_isx_frame(n, K) = n + K` places logger samples and camera
  frames. `pressure_sync` and `camera_sync` call the same function.
- `pulse_end_frame(n, K) = n - 1 + K` is used only for clock times. One example is the
  Arduino fit, which matches pulse n with `t_master_s(isx_frame n-1+K)`.
- `expected_n_ttl` holds the count rule.
- K is fixed once per rig with the test recording in `RIG_TEST_PROTOCOL.md`. It is expected
  to be 0, and the 2024 data agree: the first SYNC edge comes 0.0-0.6 ms after the movie
  start. K is never assumed silently.
- `cam_lead_frames` is kept in `recordings.csv` as a documentation column, always 1.

## Modules

| file | step | reads | writes |
|---|---|---|---|
| `manifest.py` | write a template; validate a manifest | `manifest.csv` | `manifest_template.csv` |
| `isx_meta.py` | read Inscopix metadata without Inscopix software | `.isxd`, `.gpio`, `.imu` (end of file only) | prints / JSON |
| `count_check.py` | check that the counts agree (PASS/FAIL, never repairs) | manifest, `.isxd` footer, logger CSV + `info.json`, both videos, GPIO CSV export | `recordings.csv` |
| `inscopix_day.m` | one mouse-day. Default mode `lossless_mc`: lossless copy, then motion correction with one template for pre + post. Mode `idps_preprocess` is the earlier IDPS path | raw `.isxd` (IDPS 1.9.6 `+isx`) | `<mouse>_d<day>_mc.h5` (day movie), `_mc_frames.csv`, `_inscopix.json`, `<recording_id>_raw_tsc.csv`, plus the per-recording files of the two steps below |
| `isxd_to_h5.m` | lossless copy: uint16, full frame, valid frames only, verified pixel-exact | raw `.isxd` (`+isx`) | `<recording_id>_raw.h5`, `_raw_frames.csv` (with checksums), `_raw.json` |
| `verify_lossless.py` | independent losslessness check from the `.isxd` bytes (no IDPS) | `.isxd`, `_raw.h5`, `_raw_frames.csv` | PASS/FAIL, optional JSON report |
| `motion_correct_h5.m` | NoRMCorre: shifts estimated on a filtered copy, applied to the raw values; one template per day | `_raw.h5`, `_raw_frames.csv` | `<recording_id>_mc.h5`, `_mc_shifts.csv`, `_mc_qc.png`; `<mouse>_d<day>_mc_template.mat`, `_mc_qc.png`, `_mc.json` |
| `extract_1p.m` | EXTRACT (1p settings, its own preprocessing and detection binning) plus least-squares traces that keep decreases | `_inscopix.json`, day `_mc.h5` | `<mouse>_d<day>_extract.mat`, `_frame_table.csv`, `_extract.json`, QC PNGs |
| `pressure_sync.py` | logger session: pulse QC, Arduino clock map, contact detection, schedule matching | logger CSV + `info.json`, frame times | `pressure_frames.csv`, `stim_events.csv`, `pressure_summary.json`, report, QC PNG |
| `camera_sync.py` | camera frame -> pulse -> isx_frame (`hw_trigger`; `fit` only with anchors) | video (frame count), `recordings.csv` | `camera_frames.csv` |
| `scoring_import.py` | read `ScoringAB_*.mat` (read-only) | dual-view scorer `.mat` | `behaviour_events.csv`, `video_deliveries.csv` |
| `master_timeline.py` | the join: one time base, schedule, epochs, links and latencies | all of the above | `frames.csv`; completes `stim_events.csv`, `behaviour_events.csv`, `recordings.csv` |
| `run_pipeline.py` | runs the Python steps for one manifest row and prints a one-page summary | manifest row | everything in `--out-dir` |
| `synthetic_session.py` | synthetic 4-stream recording with ground truth, in the real file formats | — | `out/synthetic/<variant>/...`, `truth.json` |

Notes on the modules:

- `RIG_TEST_PROTOCOL.md` is the one-time rig test that fixes K, plus the checks to run
  every recording day.
- `vendor/pressure_logger` is the logger app and Arduino sketch. Its file format is what
  `pressure_sync` reads.
- Bulky outputs go to `out/`, which git ignores.

## Order of work

1. **Once per rig set-up.** Do the test recording in `RIG_TEST_PROTOCOL.md` section A.
   Then run `count_check.py --exact`. A PASS with `k_from_gpio = 0` means K = 0. Write K
   into the manifest column `pulse_to_frame_offset`.
2. **Each recording day.** Fill in `manifest.csv`, one row per recording. Paths are
   relative to the manifest. Then run `manifest.py check` and `count_check.py` for every
   row. Every row must PASS.
3. **Each mouse-day (MATLAB).** Run `inscopix_day` on pre + post together, then
   `extract_1p`, then `verify_lossless.py` on each `_raw.h5` (see "The movie path" below).
   Put the outputs into the manifest:
   - `isx_timestamps` = `<recording_id>_raw_tsc.csv`
   - `extract_frame_table` = `<mouse>_d<day>_frame_table.csv`

   Without them, `t_master_s` falls back to the nominal footer period. On the 2024 rig that
   is 50 ms against a true 49.96 ms, about 1.3 s of drift over 28 min.
4. **Each recording (Python).** Run `run_pipeline.py`. It runs these steps in order:
   1. count_check
   2. pressure_sync
   3. camera_sync
   4. scoring_import
   5. master_timeline

   It stops with a non-zero exit and names the failing step when:
   - `--out-dir` already holds an earlier run's outputs (pass `--overwrite` to delete them
     first). One folder holds one recording, so no table of an earlier run is ever joined;
   - the logger pulse count is not `n_frames_isx + K` (e.g. the logger was started by hand
     after Inscopix). No option overrides this;
   - count_check FAILs for any other reason, unless an option names that reason:
     - `--allow-mismatch --anchors-camX` for a camera that lost frames;
     - `--no-camera camX` to run without a camera;
     - `--allow-gpio-tail` for a GPIO export one edge short at the end.

     A GPIO gap or a truncated movie always stops the run;
   - pressure_sync reports a loud sync problem. One of these is a stim_order the logger
     contradicts: the heat slot has touch-like contacts;
   - master_timeline refuses the join. Examples: most scored delivery keys of a slot disagree
     with stim_order (`--trust-stim-order` keeps the manifest order), or the timestamps do not
     cover the movie;
   - scoring_import finds that the `.mat` names another mouse, day or assay, or holds nothing
     scored.

   Each step can also be run on its own (see its `--help`).

## Commands for one recording

Run from this folder. The paths below are examples.

```
python manifest.py template --out manifest.csv            # then edit; one row per recording
python manifest.py check manifest.csv --require-files
python count_check.py --manifest manifest.csv --recording-id M01_d1_pre --pulse-offset 0 --exact
#   writes out/recordings.csv in this folder (never next to the manifest); --recordings to change
```

The MATLAB steps run once per mouse-day, headless. Write a small runner `.m` file that
starts with `restoredefaultpath`. Inline `-batch` strings with braces fail.

```
%   restoredefaultpath; addpath('<this folder>');
%   cfg = struct('mouse_id','M01','day',1,'out_dir','<out>/M01_d1');
%   cfg.recordings = struct('assay',{'pre','post'},'isxd',{'<pre>.isxd','<post>.isxd'});
%   info = inscopix_day(cfg);                 % mode 'lossless_mc' (default)
%   r = extract_1p(struct('day_json', fullfile(cfg.out_dir,'M01_d1_inscopix.json'), 'movie_mask','auto'));
matlab -batch "run('<out>/run_M01_d1.m')"
python verify_lossless.py --h5 <out>/M01_d1/M01_d1_pre_raw.h5      # every frame; --sample N for a sample
python verify_lossless.py --h5 <out>/M01_d1/M01_d1_post_raw.h5
```

The earlier IDPS path is still available: `cfg.mode = 'idps_preprocess'` (with its own
options, e.g. `spatial_downsample_factor`). `out/pilot_lossless/run_pilot_lossless.m` is a
complete runner.

## The movie path (decided 2026-10-19)

Before EXTRACT only two things happen. EXTRACT does every other step itself: dF/F
(`preprocess = 1`, `skip_dff = 0`), background removal (`remove_background = 1`), the spatial
high-pass (`spatial_highpass_cutoff`), the mask, and binning for detection
(`downsample_space_by`, `downsample_time_by`). EXTRACT does not motion-correct (its README).

1. **Lossless copy** (`isxd_to_h5.m`). Every recorded pixel value is copied unchanged (uint16,
   1280 x 800, every valid frame, raw order). There is no binning, crop, defective-pixel fix,
   early-frame trimming or filtering. Invalid frames (dropped, cropped, blank) have no data
   and are left out, never zero-filled; the frame table names them. The module re-reads every
   frame from the closed h5 and from the `.isxd` and requires max |diff| = 0 and equal
   checksums. `verify_lossless.py` repeats this without IDPS, straight from the `.isxd` bytes
   (the image is rows 2..801 of each 804-row block, established on every run by comparing
   frames, not assumed).
2. **Motion correction** (`motion_correct_h5.m`, NoRMCorre), written to separate
   `<recording_id>_mc.h5` files so the lossless copy stays untouched.
   - Shifts are estimated on a filtered copy: 3x3 median, the NoRMCorre/CaImAn 1p high-pass
     kernel (sigma 7 px), and 2x2 binning.
   - The shifts are applied to the raw values with NoRMCorre `apply_shifts` (FFT sub-pixel
     translation, phase term 0, border copied). The output is the raw movie translated,
     nothing else; the selftest checks it against an independent Fourier shift.
   - There is one template per mouse-day, built from the `pre` recording. Pre and post are
     both registered to it, so cells keep their positions.
   - Rigid by default; `mc_mode = 'piecewise'` for NoRMCorre patches.

The day movie `<mouse>_d<day>_mc.h5:/mov` is an HDF5 virtual dataset over the recordings'
`_mc.h5` files: EXTRACT reads one movie, nothing is copied, and the `_mc.h5` files must stay
next to it. `extract_1p` refuses a day movie whose sources are missing.

`extract_1p` defaults for this full-resolution movie:
- `avg_cell_radius` 13 px (the 2024 PCA-ICA used a 13 px diameter after 2x binning);
- EXTRACT's own `downsample_space_by = 2` for detection; the footprints come back on the full
  grid, and the least-squares traces are computed on them at full resolution;
- partitions planned from the free RAM and GPU memory, never fewer than EXTRACT's own split.

**Mask.** EXTRACT 1.x cannot combine `movie_mask` with `downsample_space_by > 1`: it applies
the full-size mask to the binned movie and stops with a size error (checked 2026-10-19). With
binning on, `extract_1p` therefore applies the mask to the result: a component with less than
half of its footprint weight inside the mask is dropped (`mask_min_inside`). The JSON logs the
count.

**Disk and time per mouse-day** (2 x 28 min at 20 Hz = 2 x 33,600 frames):

| file | per frame | per recording | per day |
|---|---|---|---|
| `_raw.h5` (uint16) | 2.05 MB | 69 GB | 138 GB |
| `_mc.h5` (single) | 4.1 MB | 138 GB | 276 GB |
| total | | | ~414 GB (`mc_output_class = 'uint16'` makes it ~276 GB) |

Disk C: has ~1.16 TB free, so about 2.8 days fit at once. Delete a day's `_mc.h5` files
(they can be remade from the raw copy) once its EXTRACT output is checked. The time and RAM
measured on the pilot are below (Known limitations, "Movie path").

The Python join runs per recording:

```
python run_pipeline.py --manifest manifest.csv --recording-id M01_d1_pre --scorer HL
#   output: out/M01_d1_pre/frames.csv, stim_events.csv, behaviour_events.csv, recordings.csv, ...
#   a rerun into the same folder needs --overwrite
```

If count_check says a camera lost frames, the camera table can only be rebuilt from
frame-verified contact frames:

```
python run_pipeline.py --manifest manifest.csv --recording-id M01_d1_pre \
    --allow-mismatch --anchors-cam1 anchors_cam1.csv          # columns cam_frame,pulse
```

## Self-tests

| command | what it shows |
|---|---|
| `python <module>.py --selftest` | every Python module tests itself on synthetic data and exits non-zero on failure |
| `python run_pipeline.py --selftest` | the end-to-end test (~10 s); details below |
| `matlab -batch "run('<this folder>/run_matlab_selftests.m')"` | `isxd_to_h5`, `motion_correct_h5`, `inscopix_day` (both modes) and `extract_1p` selftests (~2 min); the runner calls `restoredefaultpath` first |
| `python verify_lossless.py --selftest` | synthetic raw-IDAS-like `.isxd` + h5: passes a good copy and fails a changed pixel, swapped frames, a zero frame, a transposed or float `/mov`, table errors, a wrong checksum, a truncated `.isxd` |

The end-to-end test builds three synthetic recordings and checks them against
`truth.json`. Contacts are spread evenly over the pulse interval, so some fall right next
to a pulse:

- **`good`** must PASS:
  - every camera-based frame is exact: the 15 heat (video) stimuli and every behaviour event;
  - every logger onset is in the true contact's frame. The only exception is a contact flagged
    `pulse_ambiguous` and within two logger samples of a pulse, which may be one frame off
    (0-2 of 45 per run);
  - `stim_on` ends in the release's frame;
  - logger onset times are within 20 ms of the true contact, with the median within 3 ms;
  - camera event times and latencies are within 2 and 30 ms.
- **`cam_drop`** (cam1 loses 3 frames) must fail at count_check. With `--allow-mismatch` and
  anchors it must instead give a `fit` table and flag every event after the loss.
- **`manual_start`** (logger started 37 s late) must fail at count_check. No `frames.csv`
  may be written.
- Copies of the `good` row must also be refused:
  - a folder that already holds a run (refused without `--overwrite`);
  - a GPIO export missing 1000 edges (refused even with `--allow-gpio-tail`);
  - a missing cam2 video (runs with `--no-camera cam2`, and then frames.csv has no cam2 frames);
  - heat and mild swapped in stim_order (refused at pressure_sync).

The same check passed on 8 further runs, seeds 11 and 21-23 at 20.015 and 30 Hz. In each,
0-2 of 45 onsets were one frame off, early and late, and every one was flagged.

A mutation check (2026-10-19) confirms the test is sensitive. Each of these makes it fail:
- switching back to the superseded `n-1+K` rule;
- timing onsets by the last baseline sample;
- ending `stim_on` at the first baseline sample;
- moving camera times one frame late.

The ground truth is computed from the simulated pulse times, never with the pipeline's own
`count_to_isx_frame`. Before that change the `n-1+K` mutation went unnoticed, because the
truth moved with it.

## Outputs per recording (`--out-dir`)

| file | contents |
|---|---|
| `frames.csv` | one row per isx_frame (see the list below) |
| `stim_events.csv` | 60 scheduled trials plus extras (see the list below) |
| `behaviour_events.csv` | scored events (see the list below) |
| `recordings.csv` | counts, K, period, Arduino fit (ppm, residual), `count_check`, and `notes` with one segment per module, separated by ` \| ` |
| `pressure_frames.csv`, `camera_frames.csv`, `video_deliveries.csv`, `isx_frame_times.csv`, `pressure_report.txt`, `pressure_qc.png` | intermediate tables and QC |

**`frames.csv`** columns:
- `t_master_s` is the end of exposure.
- `t_assay_s` is time since the baseline start.
- `epoch`, `slot`, `stim_type`, `block`, `trial`
- `stim_on` and `event_id`
- `force_max`
- `cam1_frame`, `cam2_frame`
- `affective_state`, `reflex`
- `extract_row` is 1-based (MATLAB). Subtract 1 for numpy.

**`stim_events.csv`** has one row per scheduled trial plus extras:
- `source` is `pressure`, `video` (heat) or `expected` (missing).
- Times are on the master clock. A logger onset is the knee of the force rise (sub-sample).
  Its offset is the release knee.
- `onset_isx_frame = c + K` for a logger onset, where c is the number of pulses before the
  onset. c comes from `pressure_frames.t_pulse_arduino_s`: the fitted pulse grid, clipped to
  each pulse's sample bracket.
- `onset_cam_frame = c`, the camera frame of the same count. The first image that shows the
  contact is that frame + 1.
- `pulse_ambiguous` means a pulse lies inside the onset's uncertainty interval, so the frame
  may be one off either way. The interval comes from the fit statistics and is never narrower
  than half a sample on each side. On synthetic data 22-31% of onsets are flagged at 20 Hz.
- Ids `e001`-`e060` are the schedule positions, whatever the inputs. Extras are numbered from
  `e061`.
- `extra` rows have a slot and block but no trial. A second scored tap for the same trial is
  also flagged `duplicate`. Duplicates get no behaviour links and no `stim_on`.

**`behaviour_events.csv`**:
- `on_isx_frame = cam frame + K`.
- `on_t_master_s` is the camera snapshot (trigger) time.
- Each event is linked to the latest stimulus within 30 s, with `latency_s`. A `duplicate`
  tap is never the linked stimulus.

## Known limitations

**Not yet tested on real data**
- No real recording with all four streams exists yet. The full join is tested only on
  synthetic data.
- The real-data checks so far:
  - the logger bench sessions in `testdata/pressure`: pulse QC and clock fit only, because
    they contain no stimulus presses;
  - the 2024 `.isxd` footers and GPIO exports: frame counts and K = 0, but no logger or
    videos;
  - the 24 real ScoringAB files in `videos/output*`: import only; the files are unchanged
    (same SHA-256 before and after).

**The contract's sign of K**
- The binding count rule `n_ttl == n_frames_isx + K` and the index rule (pulse n ends
  isx_frame n-1+K, which implies `n_frames_isx - K` pulses) disagree whenever K != 0.
- With K = 0 they agree.
- A rig with K != 0 must not be analysed until CONTRACTS.md settles the sign. count_check
  says so in its K hint.

**Dropped frames**
- It is unknown whether the DAQ sends a SYNC pulse for a dropped frame. No recording so far
  has a dropped frame.
- If no pulse is sent, the counts differ and count_check FAILs by design. It never repairs
  the count.

**GPIO export one edge short**
- An export that ends a few ms before the last pulse is reported as FAIL with a "benign"
  explanation. This happened in 2 of the 4 2024 recordings.
- It stops `run_pipeline` like every other FAIL, unless `--allow-gpio-tail` is given. A real
  GPIO gap (more edges missing) is never accepted.

**Camera-scored onsets are one frame late on the index**
- Camera frame n is a snapshot at pulse n, which is the start of isx_frame n+K.
- So a change first seen at cam frame n happened during isx_frame n-1+K. Its time (the
  snapshot) is 0-1 frame period late.
- Heat onsets come from the scorer's live key taps. These are about 0.3 s late and are not
  frame-verified.

**Logger onsets: frame next to a pulse**
- A logger onset is timed directly, by the knee of the force rise. On synthetic data the
  error is: median within 0-2 ms, about 5 ms for firm presses, and up to ~16 ms for a faint
  light touch (~15 noise SD, two samples on the rise).
- An onset whose uncertainty interval holds a pulse is flagged `pulse_ambiguous`. Its frame
  may be one off, early or late. On synthetic data 22-31% of onsets are flagged at 20 Hz and
  29-51% at 30 Hz.
- On synthetic recordings 0-2 of 45 onsets per run are actually one frame off, and all of them
  are flagged.
- The pulse times come from one straight line fitted over the whole session. That assumes
  the logger's sample clock is not locked to the DAQ, so the pulse phases spread evenly.
  - The bench sessions show this: residual SD 3.6 ms, which is pure 12.5-ms quantisation.
  - An exactly 12.5-ms sample clock against 30 Hz pulses would repeat 3 phases and bias every
    pulse time by up to ~2 ms.
- The 2026-10-19 change replaced the old rule ("the last baseline sample before the contact"),
  which put about 1 onset in 4 a frame early and never late. The offset (end of `stim_on`) is
  now the release knee, no longer the first sample back at baseline (1-2 frames late).

**Logger detection**
- The detector settings (4 sigma, weak = 2x threshold, the onset interval of 9 sigma^2) are
  validated only on synthetic data.
- There is no counts-per-gram calibration.
- Force polarity is unconfirmed (`--polarity -1` exists).
- The NAU7802's own response lag (6-12 ms) is not corrected. It delays every logger onset by
  about the same amount.
- A wrong stim_order is caught only if heat lands in a slot with touch-like contacts (at least
  5, and at least half a typical touch slot), or if the typed scorer keys disagree. A swap
  between two touch stimuli is caught only by the typed keys.
- The QC PNG needs matplotlib, which is not in the contract's package list. Without it the
  PNG is skipped.

**Cameras**
- A `fit` camera table needs frame-verified contact frames as anchors, and no tool makes
  them yet.
- Anchors must give one distinct pulse of 1..n_ttl per frame: offsets that rise from 0 and
  never exceed the number of lost frames. Anchors that cover only part of the loss leave the
  rest at the end (a camera stopped early).
- A camera with more frames than pulses cannot be fitted at all.
- camera_sync on its own takes its times from the timestamps or the frame table's TSC, like
  master_timeline. With only the nominal footer period it leaves `t_master_s` empty.
- `motion_cam1` / `motion_cam2` stay empty, because no module computes motion energy yet.

**Schedule and epochs**
- Epoch, slot, block and trial come from the clock schedule only.
- So a delivery made early is labelled with the period before it (e.g. `baseline`).

**EXTRACT**
- The 2024 pilot had no clean somata. Radius, eccentricity and SNR thresholds must be tuned
  on the first Ntsr1 recording.
- Pre and post share one F0, so analysis should set its own baseline per assay.
- `temporal_raw` keeps slow drift.
- A full 2 x 28-min day is about a 66 GB h5 at the 2024 crop. Cropping to the lens about
  halves it.

**Timestamps and scoring files**
- A timestamp table must leave exactly the footer's dropped frames without a time, and never
  the last frame. So a recording whose very last raw frame was dropped is refused.
- A scoring `.mat` is compared with its recording only where its dialog fields are filled in:
  mouseID, dayNo, and a phase that says pre or post. The default phase 'Baseline' and blank
  fields are not checked.
- A `botFile` that differs from the cam1 video only warns, because videos are often renamed.

**Other**
- The MATLAB modules take a `'selftest'` argument instead of argparse.
- `run_pipeline` needs a pressure-logger file. A recording without one stops at count_check.
- master_timeline also refuses a recordings.csv row without `n_ttl`.
- pressure_sync exits 1 when any session has a LOUD issue. That includes a manual start, so
  the bench sessions in `testdata/pressure` exit 1. The tables are still written for QC.
