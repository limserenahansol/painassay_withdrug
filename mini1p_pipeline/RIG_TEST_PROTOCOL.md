# Rig test protocol — Inscopix + 2 cameras + pressure logger

All three devices count the same thing: the Inscopix DAQ **SYNC** pulse (one 5 V, 10 ms pulse
at the end of every miniscope frame). Cam1, cam2 and the Arduino hang on one BNC splitter on
that line. The analysis assumes **pulse n ends miniscope frame n-1+K, triggers cam frame n and
makes the logger's ttl_count n**; logger rows with ttl_count n and cam frame n then fall during
miniscope frame n+K (CONTRACTS.md 1a). So every device must see every pulse of a recording and
nothing else, and every count must equal the miniscope frame count + K.
"Why" lines are in *italics*.

---

## A. One test recording before the cohort (3–5 min, once per rig set-up)

This run fixes **K** (expected 0) and confirms the one-frame camera/logger lead. Repeat it after any change to
cables, splitter, camera trigger settings, IDAS version or frame rate.

**Before recording**

1. Check the wiring: DAQ SYNC → T adapter → splitter → cam1 + cam2; T adapter → Arduino pin 18 + GND.
   *All three must hang on the same SYNC line, or their counts mean different things.*
2. Do all focusing and live preview in IDAS **now**, then stop the preview and wait until the
   logger shows no `● REC` (at least 3 s).
   *The DAQ can send SYNC pulses during live preview. The logger then saves extra short sessions,
   and a preview less than 2 s before Record gets merged into the real session.*
3. Pressure logger app: connected, **"Start automatically at the first TTL pulse" ticked**, sensor
   untouched for the first second after connecting. Fill in Folder / Mouse ID (e.g. `RIGTEST`).
   *In auto mode ttl_count 1 is the first SYNC pulse. The baseline is taken in the first second,
   and pulses arriving during it are not counted.*
4. **Do not press Start** in the logger.
   *A manual start counts from the button press: ttl_count then no longer equals the frame
   number, and it also counts any preview pulses.*
5. IC Capture, both cameras: external-trigger mode on (rising edge), exposure shorter than one
   miniscope frame, camera frame-rate setting above the miniscope rate. Start capture to file on
   **both cameras before** pressing Record in IDAS.
   *Camera frame n must be pulse n. A camera armed late misses the first pulses, and a camera that
   cannot keep up skips pulses.*

**Recording**

6. Press Record in IDAS. Check that the logger shows `● REC RIGTEST S..` with the pulse count
   rising, and that both camera frame counters rise.
   *Shows straight away that every branch of the splitter gets pulses.*
7. Make 2–3 firm taps on the pressure sensor, about 1 min apart, where **both cameras** can see
   the contact.
   *Each tap is a moment seen by the logger (pulse number) and by both videos: it checks cam
   frame n = pulse n.*
8. Optional, recommended once: with the miniscope off the mouse, aim it at a surface that both
   cameras see, then switch a light (phone torch) on and off 3 times, holding each state ≥ 2 s.
   *This is the only event the miniscope and the cameras both see. It confirms K and the one-frame
   lead from images, without relying on counts.*
9. Stop IDAS. Wait until the logger shows the session saved (about 2 s), **then** stop both cameras.
   *The logger closes the session 2 s after the last pulse. A camera stopped before IDAS loses the
   last frames.*

**After recording**

10. Copy `*_video.isxd`, `*_video.gpio` and `*_video.imu` from the DAQ, the logger
    `*_pressure.csv` and `*_info.json`, and both videos. In IDPS import the `.gpio` and export it to
    CSV (`Time (s), Channel Name, Value`).
    *The GPIO CSV is the DAQ's own record of the SYNC pulses it sent. It is a fourth, independent count.*
11. Add a row for the test to `manifest.csv` and run:
    `python count_check.py --manifest manifest.csv --recording-id RIGTEST --exact`
    *Compares the miniscope frames, logger pulses, both video frame counts and the GPIO SYNC edges.
    `--exact` decodes every video frame, because container counts can be wrong.*
12. Read the result:
    - **PASS** with `k_from_gpio = 0` → K = 0. Use `--pulse-offset 0` for every recording.
    - FAIL with "logger and cameras agree on P pulses ... would need K = ..." → read the rest of the
      message. If it says `k_from_gpio = 0`, K is 0 and pulses are missing at the end of the
      recording: find out why before the cohort. If `k_from_gpio` is not 0, the rig has a real
      K offset; CONTRACTS.md's count rule (n_frames + K) and index rule (pulse n ends frame n-1+K)
      disagree in sign for K ≠ 0, so stop and settle the contract before any analysis.
    - "GPIO export has 1 fewer SYNC edge ... stopped before the final frame's pulse" → harmless
      (the GPIO log ends a few ms before the last pulse). Go by the logger and camera counts.
      `run_pipeline` still stops on it unless you pass `--allow-gpio-tail`.
    - "camX has N fewer frames" or "logger has N more pulses" → fix the cause given in the
      message, then repeat the test.
13. Check the taps: run `pressure_sync.py` on the test session and read `onset_cam_frame` (= c)
    of each tap in `stim_events.csv`. In both videos, the first frame that shows contact should be
    **c+1**, or c or c+2 for a tap flagged `pulse_ambiguous`.
    *The pipeline times the contact by the knee of the force rise and counts the pulses before
    it: c. Camera frame c is the snapshot at pulse c, taken before the contact, so the next
    snapshot (c+1) is the first that can show it. A tap whose timing uncertainty holds a pulse
    is flagged pulse_ambiguous and may be one frame off either way. Do not use the first row
    where `corrected` rises: the contact happened before that row, and a pulse between the two
    gives that row a count one higher.*
14. If step 8 was done: the first lit camera frame (1-based) should be the first lit miniscope frame
    (0-based) + 1 when K = 0. Keep the files; the light step is analysed together with the counts.
    *Cam frame n is a snapshot taken at pulse n, the start of miniscope frame n+K. A change that
    happens during miniscope frame k is first seen at the next pulse, i.e. by cam frame k+1-K, which
    the pipeline places in miniscope frame k+1: a camera-scored onset lands one frame after the
    frame it happened in, and its time (the snapshot) is 0-1 frame period late. Logger onsets are
    timed directly (synthetic data: median error ~0-2 ms, up to ~16 ms for a faint light touch,
    plus the sensor's own response lag, which is not corrected); about a quarter of them at 20 Hz
    lie close enough to a pulse to be flagged `pulse_ambiguous` and may be one frame off.*

Write K, the date, the IDAS version and the frame rate in the rig log.

---

## B. Every recording day

**Start of day**

- Logger app open and connected, auto-TTL ticked, Start button never used. Do not touch the sensor
  for the first second after connecting.
  *Same reasons as A3–A4: ttl_count must start at the first frame.*
- Both cameras in external-trigger mode, with the same settings as in the rig test.
  *K and the one-frame lead were measured with these settings.*

**Each recording (one IDAS recording = one logger session = one video per camera)**

1. Focus and preview first, stop the preview, and wait until the logger shows no `● REC` (≥ 3 s).
   *Avoids preview pulses in the session and extra short sessions.*
2. Set the logger Mouse ID and Details. Name both camera files with the recording_id
   (e.g. `M01_d1_pre`).
   *These names are the keys that join the files in the manifest.*
3. Arm both cameras (capture to file), then press Record in IDAS.
   *The cameras must be armed for pulse 1.*
4. During the recording, check that the logger shows `● REC` with a rising pulse count, and that
   IDAS shows 0 dropped frames.
   *A stopped count or dropped frames are cheaper to catch now than after 28 min.*
5. Stop IDAS, wait until the logger has saved the session, then stop both cameras.
   *Same as A9.*
6. Write down the logger session number (S01, S02, ...) next to the recording_id. Do not delete
   short extra sessions; note them instead.
   *The manifest must point to the right session. The extra ones show what went wrong.*

**End of day**

- Copy `.isxd`, `.gpio`, `.imu`, logger CSV + info.json and both videos. Export each `.gpio` to CSV
  in IDPS.
  *Every recording needs all four counts.*
- Fill in `manifest.csv` and run
  `python count_check.py --manifest manifest.csv --pulse-offset K --exact`.
  Every row must be **PASS**. For a FAIL, the `notes` column of `recordings.csv` (by default
  `out/recordings.csv` in the pipeline folder) says which device
  and the likely cause.
  *A mismatch found on the same day can still be explained from memory and the lab notes.*
