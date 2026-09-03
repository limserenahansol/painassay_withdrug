# painassay_withdrug

Acute pain assay under mini1p imaging, with a β-arrestin-biased NTSR1 agonist
(SBI-553). Manual behaviour scoring, QC, and the whole analysis pipeline.

Day 1 (no drug) is complete for six mice. Day 2 (drug) is being scored. The
Day-2 figures currently in `figures/day2_mockup/` are generated from
**synthetic** data — they exist to settle the layout and the statistics before
the real day, and every one of them is stamped `MOCKUP`.

---

## What the assay measures

A 28-minute session per mouse: 5 min baseline, then four 5-min stimulus blocks
separated by 1-min rests. The four stimuli — light touch, mild touch, heat, pin
prick — are delivered by hand, and their block order is randomised per animal.

Six behaviours are scored, in **two classes that are never summed together**:

| Class | Behaviours | Why separate |
|---|---|---|
| **Reflexive** | paw withdrawal, flinch | spinally mediated, locked to contact (3 s window) |
| **Affective-motivational** | paw attending, licking/biting, guarding | supraspinal, builds over 1–8 s (10 s window) |
| *(scored, not a pain measure)* | escape / rearing | exploration — baseline 3.8/min and it **decreases** under stimulation |

A biased agonist is expected to move the affective class without a matching
reflexive change. Summing them into one "pain score" would hide exactly that.

---

## The one methodological point everything rests on

**Every measure is `total event number / total stimulus delivery`.** Never a
raw count.

The stimulus is delivered by hand, so the number of taps is not fixed. On Day 1
alone, across six mice:

| Stimulus | fewest | most | spread |
|---|---|---|---|
| Light touch | 17 | 30 | 1.76× |
| Mild touch | 19 | 26 | 1.37× |
| Heat | 15 | 23 | 1.53× |
| Pin prick | 16 | 31 | 1.94× |

Ten flinches from ten taps and ten flinches from twenty taps are the same raw
count and a different animal: 10/10 = 1.00 versus 10/20 = 0.50.

`events_per_min` does **not** fix this — it divides by time, and the session
length is fixed by the protocol while the tap count is not.

Counting bouts also beats a yes/no response. Injecting a known 40 % reduction
into synthetic data:

| Readout | True effect | Recovered |
|---|---|---|
| total events / total deliveries | 0.60× | **0.64–0.68×** |
| any response, yes/no | 0.60× | 0.82× — badly underestimated |

A binary hit saturates: if a window usually holds two or three bouts, removing
half of them barely moves the probability.

See `figures/block_window/` and `docs/PIPELINE.md` for the full argument.

---

## Statistics at two levels

With six animals the exact paired Wilcoxon has a hard arithmetic floor of
`2/2^n`:

| Usable pairs | Smallest possible p |
|---|---|
| 6 | 0.031 |
| 5 | 0.062 |
| 4 | 0.125 |

Reaching 0.031 needs **all six** animals to move the same way. One
non-responder is dropped, n falls to 5, and `p < 0.05` becomes *arithmetically
impossible* regardless of effect size.

**The per-delivery data rescues this.** Each delivery is a Bernoulli trial and
each mouse receives roughly 90 per day, so a within-mouse test is possible and
properly powered:

| Level | Unit | n | Test |
|---|---|---|---|
| Population | the mouse | 6 | paired Wilcoxon (**primary**) |
| Population | the mouse | 6 vs 6 | Mann-Whitney (secondary) |
| **Individual** | one delivery | ~90 per mouse per day | Fisher exact, rate ratio |
| Direction | the mouse | 6 | how many moved the same way |

Power, simulated from the measured between-mouse variability (median CV 0.49):

| True effect | n = 6, all respond | n = 6, one non-responder |
|---|---|---|
| 20 % | 0.37 | 0.21 |
| 30 % | 0.78 | 0.49 |
| 40 % | 0.98 | **0.49** |

**n = 8–10 is the robust range.** At n = 6 the design only works if every
animal responds.

---

## Repository layout

```
scoring/                 MATLAB manual scorer (both cameras, live label track)
analysis/                Python pipeline, step1 -> step4 plus figure scripts
docs/                    PIPELINE.md is the full workflow; qc/ holds the
                         validation reports
figures/
  lab_meeting/           Day 1 descriptive figures (F1-F6, E1-E5)
  block_window/          5 min vs 6 min block comparison (W1, W2)
  day2_mockup/           Day 1 vs Day 2 layouts, SYNTHETIC Day 2, stamped
  qc/ qc_face/ qc_locomotion/   recording quality, face resolvability
data/
  day1_scoring_raw/          your scoring, never modified
  day1_scoring_corrected/    audited copy — use this downstream
  block_measures_5min/       step1 output, 300 s blocks
  block_measures_6min/       step1 output, 360 s blocks (stimulus + rest)
slides/                  the lab-meeting deck
clips/                   short labelled example videos
```

Raw session videos (~128 GB of `.avi`) are **not** in the repository. They stay
on the acquisition machine.

---

## Scoring a session

Open `scoring/RUN_scoring.m` in MATLAB and press **F5**. Everything else is
dialogs.

Paths are set at the top of `RUN_scoring.m` in the `DAYS` table — edit those to
match your machine. Each row is `label, bottom folder, side folder, output
folder, day number`; rows whose folders hold no video are dropped
automatically, so add a row per recording day.

Keys:

```
HOLD   a attending     s licking/biting   d guarding   f escape/rearing
TAP    w withdrawal    e flinch
TAP    1 2 3 4         stimulus delivered, by type
HOLD   u               "I cannot tell"  (excluded from classifier training)
SPACE pause    <- -> step a frame    . faster    , slower
z undo last mark    BACKSPACE delete nearest mark    q stop and save
```

Drag the seek bar to jump anywhere in the session. A scrolling track under the
two videos shows the last 30 s of everything marked, so a stuck key or a missed
delivery is visible immediately.

Scoring is **blind**: treatment is never entered or displayed.

---

## Running the analysis

```bash
cd analysis

# 1. audit the scoring and write a corrected copy. Never edits the original.
python qc_and_correct_all.py --src ../data/day1_scoring_raw \
                             --out ../data/day1_scoring_corrected

# 2. per-block measures, both block definitions
python step1_block_measures.py ../data/day1_scoring_corrected \
                               --block-s 300 --out ../data/block_measures_5min
python step1_block_measures.py ../data/day1_scoring_corrected \
                               --block-s 360 --out ../data/block_measures_6min
python compare_block_windows.py --dir300 ../data/block_measures_5min \
                                --dir360 ../data/block_measures_6min \
                                --out ../figures/block_window

# 3. per-stimulus statistics, difference-in-differences, mixed model
python step2_block_stats.py ../data/block_measures_5min

# 4. Day 1 descriptive figures
python make_lab_meeting_figs.py
python make_extra_figs.py --day1 ../data/day1_scoring_corrected
```

Once Day 2 is scored, add:

```bash
python step3_day_comparison.py --day1 <d1> --day2 <d2> --out ../figures/day2
python step4_individual_stats.py --day1 <d1> --day2 <d2> --no-isolation \
                                 --out ../figures/day2
python make_day2_figs.py --day1 <d1> --day2 <d2> --out ../figures/day2
python make_full_ppt.py --real
```

`--no-isolation` on step4 is deliberate. The isolation filter keeps only
deliveries with 10 s of clearance, which is 122 of 535 — fine for the
population test, too thin for a per-mouse one. Dropping it is valid **for the
day comparison specifically**, because the mis-attribution inflates both days
equally (pacing is set by the protocol, not by the drug). It is *not* valid for
comparing one stimulus against another.

### Planning with a synthetic Day 2

```bash
python make_synthetic_day2.py <day1 corrected> <output folder>
python make_day2_figs.py --day1 <d1> --day2 <synth> --out <figs> --mockup
```

`--mockup` stamps every panel and appends `_MOCKUP` to every filename. **Do not
remove that flag.** Nothing synthetic is ever merged with real output.

---

## Known limits of this rig

Established by measurement, documented in `docs/qc/`:

- **The bottom view cannot support paw-level scoring.** Geometry, not
  processing — mesh pitch against paw size.
- **Automatic delivery detection is not usable.** 64 % of manual marks matched,
  and coverage ranged 45–85 % by stimulus, which would bias per-stimulus
  comparisons. A TTL or LED marker on the applicator solves timing, identity
  and camera sync at once.
- **The Mouse Grimace Scale cannot be scored.** The side view is a backlit
  silhouette; ears are visible as outlines but the eye is not resolved at all,
  so orbital tightening cannot be graded. MGS needs a front or 3/4 view,
  front-lit, head filling ≥ 300 × 300 px.
- **A grimace-like face does not distinguish pain from sedation.** Orbital
  tightening and flattened ears occur in both. Activity is the discriminator —
  and NTSR1 agonists are known to cause hypolocomotion, so for SBI-553 sedation
  is a live hypothesis. The `escape / rearing` counts are the built-in control.

---

Hansol Lim · HEAL mini1p / SBI-553
