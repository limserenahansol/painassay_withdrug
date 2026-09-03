# Repair report - fast repeated clicks merged

- merge gap: **0.35 s**
- originals: `C:/Users/hsollim/Documents/HEAL_mini1p_SBI553/videos/output` (unmodified)
- sessions repaired: 2, delivery files copied: 6

## Counts before and after

| file | behaviour | raw_events | merged_events | gaps_filled | counts_trustworthy |
|---|---|---|---|---|---|
| female1bottom000126_09_0212_00_19 | attending | 14 | 10 | 4 | yes |
| female1bottom000126_09_0212_00_19 | lickbite | 49 | 20 | 29 | approximate |
| female1bottom000126_09_0212_00_19 | guarding | 2 | 2 | 0 | yes |
| female1bottom000126_09_0212_00_19 | escape | 104 | 86 | 18 | approximate |
| female2bottom000226_09_0212_33_04_009 | attending | 33 | 19 | 14 | yes |
| female2bottom000226_09_0212_33_04_009 | lickbite | 79 | 51 | 28 | approximate |
| female2bottom000226_09_0212_33_04_009 | guarding | 30 | 24 | 6 | approximate |
| female2bottom000226_09_0212_33_04_009 | escape | 100 | 70 | 30 | approximate |

## Threshold sensitivity

Where the count plateaus across thresholds the merge choice is safe. Where it keeps falling there is no clean boundary between artefact and real gaps, so treat that count as approximate.

| file | behaviour | n_at_0s | n_at_0.13s | n_at_0.2s | n_at_0.35s | n_at_0.5s | n_at_1s |
|---|---|---|---|---|---|---|---|
| female1bottom000126_09_0212_00_19 | attending | 14 | 11 | 10 | 10 | 10 | 10 |
| female1bottom000126_09_0212_00_19 | lickbite | 49 | 25 | 22 | 20 | 19 | 18 |
| female1bottom000126_09_0212_00_19 | guarding | 2 | 2 | 2 | 2 | 2 | 2 |
| female1bottom000126_09_0212_00_19 | escape | 104 | 94 | 89 | 86 | 82 | 77 |
| female2bottom000226_09_0212_33_04_009 | attending | 33 | 19 | 19 | 19 | 19 | 19 |
| female2bottom000226_09_0212_33_04_009 | lickbite | 79 | 59 | 55 | 51 | 50 | 44 |
| female2bottom000226_09_0212_33_04_009 | guarding | 30 | 28 | 27 | 24 | 24 | 22 |
| female2bottom000226_09_0212_33_04_009 | escape | 100 | 77 | 72 | 70 | 68 | 56 |

## Read this before using the corrected files

- **Counts are the usable measure.** `guardMin` is set to 0 in the corrected `.mat`, so no duration filtering is applied.
- **Durations are NOT usable.** When a key is tapped rather than held, the merged span is the span of the clicking, not of the behaviour. Do not report `% of time`.
- The `> 2 s` guarding criterion is no longer applied by code. It is the scorer's judgement and must be stated as such in the methods.
