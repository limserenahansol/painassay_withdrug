# mini1p pain pipeline — the whole picture (2026-10-19)

Inscopix nVista3 (30 Hz) + two Imaging Source cameras triggered by the Inscopix DAQ SYNC
output (one camera frame per miniscope frame). Downstream statistics are the mini2p ones,
unchanged. One mouse-day = a baseline session (`pre`) and a post-injection session (`post`,
SBI-553 or vehicle), 16.5 min each.

```
.isxd ──► 1 movie ──► 2 sweep ──► 3 extract ──► 4 cell review ──► keep ──┐
                       (YOU pick)               (YOU edit keep)          │
AVI x2 ──► scoring GUI (YOU score) ──► ScoringAB_*.mat ─────────────────► merge ──► classify ──► cohort
```

| stage | code | what it does | QC you look at |
|---|---|---|---|
| 1 movie | `inscopix_day` (`isxd_to_h5`, `motion_correct_h5`) | lossless .isxd → h5 (every pixel unchanged, verified twice: IDPS re-read + independent byte check `verify_lossless.py`); NoRMCorre rigid motion correction, shifts estimated on a filtered copy and applied to the raw values, ONE template for pre and post (same pixel grid, same cells); stored uint16 + gzip (lossless, 1.77x smaller); the verified raw copy is then deleted | `<tag>_mc_qc.png`, `<rec>_mc_qc.png`; the run prints per session: largest shift, frames at the shift limit, frames that match the template badly. **Gate:** extract refuses to run while this says CHECK (2024 pilot: the post file was another field of view, r = 0.24 to the template) |
| 2 sweep | `extract_param_sweep` | 12 EXTRACT settings (radius 10/13 x T_min_snr 5/7 x spatial high-pass 5/8/12) on a 400-px crop of an active 5-min piece, binned 4x like the real detection; every component scored by `cell_qc_metrics` and by the movie at its own events (`event_image`); bright spots no component covers are counted as missed | `<tag>_sweep_overview.png` (where the cells are), `<tag>_sweep_sheet.png` (green cell-like / yellow check / red not a cell / cyan + missed, per setting) |
| 3 extract | `extract_1p` | whole day, EXTRACT's own dF/F + high-pass + background removal, detection on the 4x-binned movie, traces at 30 Hz; least-squares traces kept for inhibited cells | `<tag>_extract_footprints.png`, `<tag>_extract_traces.png` |
| 4 review | `extract_cell_review` | numbered map + one row per component (zoom, footprint, whole-day trace, average transient), keep table pre-filled | `<tag>_cells_map.png`, `<tag>_cells_page_NN.png`; edit `keep` in `<tag>_cell_review.csv` |
| keep | `apply_cell_keep` | `<tag>_extract_kept.mat` (never overwrites EXTRACT's file); cell ids stay the review numbers | |
| scoring | `scoring_gui\RUN_scoring_mini1p_20261019.m` | the mini2p dual-view scorer, with the 30 Hz triggered rate (AVI header says 60), sessions paired by equal frame count | hold-check printout after saving |
| merge | `merge_mini1p_scoring` | camera frame n = Inscopix frame n (refused if the frame counts differ); writes mini2p's `final_neuron_behavior.mat` (traces as F/F0 = 1 + dF/F) | |
| classify | mini2p `classify_pain_neurons` | peri-event traces, per cell and event: activated (up) / inhibited (down) / not (signrank [1,3] s vs [-3,-1] s, p < 0.05, ≥ 3 trials); events: pin, heat, light, mild, withdrawal, flinch, attending, lick/bite, guarding, escape | `figures\peri_*.png`, `figures\peri_per_cell\*.png`, `neuron_pain_classification.csv` |
| cohort | `summarize_mini1p_cohort` | counts per session; same cell before vs after the injection | `cohort_session_counts.csv`, `cohort_cells_pre_post.csv` |

Run everything from `RUN_mini1p_20261019.m` (stages are skipped when their files exist).

Extra QC tools: `qc_stage_media.m` (raw vs MC video, the filtered movie EXTRACT detects on,
stage images, soma-size zoom), `behavior_auto.py` (camera QC, scoring videos, rearing, motion
energy), `quick_behavior_downstream.m` (calcium vs automatic behaviour, before any scoring).
A worked example of every stage on real data: `mini1p_out\test_5941_5min\step1..step4*.m`.

## Why these settings (literature, checked 2026-10-19)

- Motion correction on a filtered copy, applied to the raw movie, rigid: Corder et al. 2019
  (BLA pain ensemble; TurboReg on a 6–10 µm band-pass copy), EXTRACT's own
  `run_normcorre_pipeline`, Inscopix IDPS.
- Detection on a 4x time-binned movie, traces on every frame: EXTRACT template
  (`downsampletime_pipeline`, then the full movie with `S_init`, `max_iter = 0`),
  Corder 2019 / Parker 2018 (20 → 5 Hz), the mini2p pipeline.
- No spatial binning of the data: EXTRACT bins 2x internally for detection only.
- 1p: EXTRACT `preprocess = 1`, `remove_background = 1` (EXTRACT README).
- Manual curation of every component: Corder 2019 kept 39 % of the ICA outputs.

## Disk (1280 x 800, 30 Hz, 16.5 min ≈ 30,000 frames per session)

raw copy 61 GB while processing (deleted after verification + MC), mc movie ≈ 35 GB kept,
so ≈ 70 GB per mouse-day kept and ≈ 130 GB free needed while it runs.

## Real-data test, mouse 5941 SBI session, first 5 min (2026-10-19)

`mini1p_out\test_5941_5min\` (E: and K: only read)

- Sync: 30,313 Inscopix frames = 30,313 GPIO SYNC pulses = 30,313 bottom-camera frames =
  30,313 side-camera frames (0 dropped, 29.9994 Hz).
- Lossless: 9,000 / 9,000 frames identical in both checks (IDPS re-read and the independent
  byte check `verify_lossless.py`).
- Motion: median shift 0.37 px, max 2.95 px; mc movie uint16 + gzip = 10.9 GB vs 18.4 GB raw.
- QC media (`2_qc\`): raw vs MC video, the filtered movie EXTRACT sees, stage images; the
  local-correlation image shows compact somata of radius ~9-13 px. Hot pixels and the
  vignetted edge are noise (edge excluded from the missed-soma count).
- Parameter search (`3_sweep_r*\`, 400-px crop, whole 5 min binned 4x): radius 9-15 and
  T_min_snr 4-10 change little; the spatial high-pass cutoff matters most (3 -> 5 -> 8: 5 -> 8
  -> 9 cell-like components) - milder is better here.
- Missed somata (bright Cn x PNR spots no component covers, frame edge excluded): high-pass
  8-12 covers 8-11 of 10-11; 20 or none is worse (none: almost no cell-like components).
  **Chosen: avg_cell_radius 10, T_min_snr 5, cellfind_min_snr 1, spatial_highpass_cutoff 8**
  (hot-pixel median filter: no effect).
- Whole field, 5 min: 18 components; the movie-at-its-events proof (`event_image`) is r 0.8-0.94
  for clear cells and ~0.5 for bright static spots whose traces are only the opening bleaching.
  A movement-suppressed cell has a mostly-decreasing trace: such traces are 'check', never
  rejected for that alone. Transients are measured on a 0.2-s running mean (at 30 Hz the
  largest raw peaks are shot-noise spikes).
- Quick downstream (`quick_behavior_downstream.m`, automatic behaviour only): 23 movement
  onsets, rearing 2 bouts (too few to test); 9 of 18 cells negatively and 3 positively
  correlated with movement - this 5-min piece has one still -> moving switch at ~185 s, so
  part of that is one slow state change.
- Behaviour (`behavior\`, `behavior_auto.py`): both cameras progressive, no duplicated frames;
  side camera 25 % of pixels saturated (the mouse is not); the bottom-view mouse is a backlit
  silhouette (grey 6-11) - no paw detail is recorded, so light the floor from the camera side
  for paw scoring. Rearing = high AND upright (height/width >= 1.3) side-view blob; the
  height alone was fooled by a mouse sitting with its back to the camera.

## Open

- µm per pixel of this rig is not measured (≈ 0.8 µm/px from the 1.2 mm FOV diagonal);
  image a stage micrometer once.
- Pressure logger not used; deliveries come from the video scoring.
- Missed somata (mini2p `append_uncovered_somata`) not ported yet.

## Whole-day run and downstream (2026-10-20 night)

Overnight driver: `RUN_ca_overnight_20261019.m` with `WORKER` = `copy` | `mc` | `detect` (three
MATLAB processes handing mice on through flag files in `mini1p_out\20261019`), then
`RUN_ca_overnight_post_20261019.sh` -> `ca_post_mouse.sh <mouse>` and, when all four are done,
`ca_post_cohort.sh`.

- Detection: the 5941 recipe (`extract_1p_seeded`), `detect_prebin` (EXTRACT on a movie binned
  4x in time and 2x2 in space, ONE partition: the 2x2-partition seams put false candidates on a
  line), EXTRACT on the CPU; least squares traces with `ls_use_gpu` 'auto' (spatial filter on the
  GPU in 250-frame blocks, CPU fallback; the filter is the slow step).
- `ca_transfer_decisions.py`: the user's keep / delete decisions of a reference run (5941 5-min
  test, a reviewed preview, or Day 1 for Day 2) -> the day candidates (NCC registration of the
  mean images, cosine-overlap Hungarian match >= 0.5; weak matches 0.3-0.5 keep only if the rules
  keep too; the rest by the rules, keep = verdict != not_cell). Writes keep / note /
  decision_source into the day review table (backup `_rules.csv`).
- `rescue_missing_cells.m`: reference cells the user kept that the day run did not find again:
  footprint shifted in, joint least squares with every candidate. Not added if it overlaps a day
  component at cos >= 0.5 (two user cells merged into one component - adding it would split the
  signal: anti-correlated traces) or lies < 50 % inside the lens mask.
- `ca_duplicate_check.py`: kept pairs with footprint cos >= 0.5, or >= 0.2 with |r| >= 0.7.
- `ca_trace_outputs.py`, `ca_session_figures.py`, `ca_qc_sheet.py`, `ca_behaviour_video.py`.
- `ca_day1_analysis.py` (A1-A5 whole session, B1-B10 event-locked, C1 arms) and
  `ca_day1_extended.py` (E1-E14), validated on `ca_synthetic_day1.py` data with known responses
  (analysis\ca_validation_synthetic): the trial-wise Wilcoxon misses inhibited cells, the
  time-shift null is the sensitive rule; % responsive depends on the trial count (B1b matches
  it); block design lets drift 'decode' stimuli (sham control); one ridge penalty per mouse.
- `ca_toy_day2.py` + `ca_longitudinal.py`: crossover Day 2 (toy until recorded), cross-day
  matching, baseline (mean of 2) vs vehicle vs SBI-553.
- `ca_morning_summary.py` -> `SUMMARY_20261021_KR.md`.
