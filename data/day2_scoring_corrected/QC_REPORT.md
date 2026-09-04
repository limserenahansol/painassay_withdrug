# QC and correction report - Day 1 (no drug)

- originals: `..\videos\output_day2` **unmodified**
- corrected: `..\videos\output_day2_corrected`
- merge gap 0.35 s; blocks assigned by clock window 5-10 / 11-16 / 17-22 / 23-28 min
- guarding events given a nominal 1 s each (a tap records the keypress, not the behaviour); event counts unchanged
- stimulus names: F1=sheet, F2=sheet, F3=sheet, M1=typed, M2=sheet, M3=typed
- 6 session(s), 93 change(s), 3 flag(s)

Stimulus identity comes from the names TYPED WHILE SCORING, not from the randomisation sheet, because the delivery order on the day did not follow the sheet. Only the spelling was normalised.

## Session facts

| mouse | session | file | dur_s | n_del | runs_after_fix | n_miskey_fixed | labelled_frames | stimuli |
|---|---|---|---|---|---|---|---|---|
| F1 | 7 | female1bottom000726_09_0311_32_06_001 | 1682.6 | 58 | 4 | 4 | 1464 | Pin prick:17 / Heat:15 / Mild touch:13 / Light touch:13 |
| F2 | 8 | female2bottom000826_09_0312_01_40 | 1687.4 | 57 | 4 | 4 | 753 | Pin prick:12 / Heat:13 / Light touch:18 / Mild touch:14 |
| F3 | 9 | female3bottom000926_09_0312_31_11 | 1680.9 | 68 | 4 | 4 | 905 | Light touch:17 / Mild touch:18 / Pin prick:19 / Heat:14 |
| M1 | 10 | male1bottom001226_09_0314_00_32 | 1696.6 | 64 | 4 | 0 | 754 | Light touch:12 / Pin prick:16 / Heat:19 / Mild touch:17 |
| M2 | 11 | male2bottom001126_09_0313_30_46 | 1694.4 | 70 | 4 | 6 | 1665 | Heat:19 / Pin prick:17 / Mild touch:15 / Light touch:19 |
| M3 | 12 | male3bottom001026_09_0313_01_11 | 1689.2 | 71 | 4 | 4 | 1985 | Light touch:17 / Mild touch:17 / Heat:21 / Pin prick:16 |

## Behaviour events after merging fast clicks

| mouse | attending_before_merge | attending_events | escape_before_merge | escape_events | guarding_before_merge | guarding_events | lickbite_before_merge | lickbite_events |
|---|---|---|---|---|---|---|---|---|
| F1 | 3 | 3 | 12 | 12 | 19 | 15 | 14 | 13 |
| F2 | 8 | 8 | 3 | 2 | 12 | 9 | 7 | 7 |
| F3 | 8 | 7 | 10 | 9 | 7 | 6 | 1 | 1 |
| M1 | 6 | 5 | 11 | 11 | 1 | 1 | 1 | 1 |
| M2 | 13 | 13 | 22 | 21 | 4 | 3 | 3 | 3 |
| M3 | 20 | 19 | 23 | 22 | 13 | 12 | 9 | 9 |

## FLAGGED - not changed, needs your decision

| file | mouse | kind | detail | action |
|---|---|---|---|---|
| female2bottom000826_09_0312_01_40 | F2 | treatment differs from the sheet | sheet says 'Vehicle' for session 8, recorded as 'SBI-553' on your instruction that every mouse received it | wrote 'SBI-553' |
| male1bottom001226_09_0314_00_32 | M1 | treatment differs from the sheet | sheet says 'Vehicle' for session 10, recorded as 'SBI-553' on your instruction that every mouse received it | wrote 'SBI-553' |
| male3bottom001026_09_0313_01_11 | M3 | treatment differs from the sheet | sheet says 'Vehicle' for session 12, recorded as 'SBI-553' on your instruction that every mouse received it | wrote 'SBI-553' |

## Every change made

| file | mouse | kind | detail | before | after |
|---|---|---|---|---|---|
| female1bottom000726_09_0311_32_06_001 | F1 | stimulus name spelling only | slot 1 | pin | Pin prick |
| female1bottom000726_09_0311_32_06_001 | F1 | stimulus name spelling only | slot 2 | heat | Heat |
| female1bottom000726_09_0311_32_06_001 | F1 | stimulus name spelling only | slot 3 | mild | Mild touch |
| female1bottom000726_09_0311_32_06_001 | F1 | stimulus name spelling only | slot 4 | light | Light touch |
| female1bottom000726_09_0311_32_06_001 | F1 | stimulus mis-key (clock window) | t=1199.5s is in block 3 (1020-1320s) | Heat | Mild touch |
| female1bottom000726_09_0311_32_06_001 | F1 | stimulus mis-key (clock window) | t=1201.6s is in block 3 (1020-1320s) | Heat | Mild touch |
| female1bottom000726_09_0311_32_06_001 | F1 | stimulus mis-key (clock window) | t=1201.9s is in block 3 (1020-1320s) | Heat | Mild touch |
| female1bottom000726_09_0311_32_06_001 | F1 | stimulus mis-key (clock window) | t=1247.2s is in block 3 (1020-1320s) | Heat | Mild touch |
| female1bottom000726_09_0311_32_06_001 | F1 | fast clicks merged | lickbite: 1 gap(s) <= 0.35s filled | 14 | 13 |
| female1bottom000726_09_0311_32_06_001 | F1 | fast clicks merged | guarding: 4 gap(s) <= 0.35s filled | 19 | 15 |
| female1bottom000726_09_0311_32_06_001 | F1 | nominal duration applied | guarding: 9 event(s) extended to 1s each (a tap records the keypress, not the behaviour). Event COUNT is unchanged | 15.13s total | 19.93s total |
| female1bottom000726_09_0311_32_06_001 | F1 | metadata filled | sessionNo | (empty) | 7 |
| female1bottom000726_09_0311_32_06_001 | F1 | metadata filled | mouseID | (empty) | F1 |
| female1bottom000726_09_0311_32_06_001 | F1 | metadata filled | sexID | (empty) | F |
| female1bottom000726_09_0311_32_06_001 | F1 | metadata filled | phase | Baseline | Post-treatment |
| female1bottom000726_09_0311_32_06_001 | F1 | metadata filled | treatment | (empty) | SBI-553 |
| female1bottom000726_09_0311_32_06_001 | F1 | guardMin | tap scoring: a duration filter would discard nearly every mark | 2.0 | 0.0 |
| female2bottom000826_09_0312_01_40 | F2 | stimulus name spelling only | slot 1 | pin | Pin prick |
| female2bottom000826_09_0312_01_40 | F2 | stimulus name spelling only | slot 2 | heat | Heat |
| female2bottom000826_09_0312_01_40 | F2 | stimulus name spelling only | slot 3 | light | Light touch |
| female2bottom000826_09_0312_01_40 | F2 | stimulus name spelling only | slot 4 | mild | Mild touch |
| female2bottom000826_09_0312_01_40 | F2 | stimulus mis-key (clock window) | t=477.0s is in block 1 (300-600s) | Heat | Pin prick |
| female2bottom000826_09_0312_01_40 | F2 | stimulus mis-key (clock window) | t=1393.2s is in block 4 (1380-1687s) | Light touch | Mild touch |
| female2bottom000826_09_0312_01_40 | F2 | stimulus mis-key (clock window) | t=1395.0s is in block 4 (1380-1687s) | Light touch | Mild touch |
| female2bottom000826_09_0312_01_40 | F2 | stimulus mis-key (clock window) | t=1397.2s is in block 4 (1380-1687s) | Light touch | Mild touch |
| female2bottom000826_09_0312_01_40 | F2 | fast clicks merged | guarding: 3 gap(s) <= 0.35s filled | 12 | 9 |
| female2bottom000826_09_0312_01_40 | F2 | nominal duration applied | guarding: 7 event(s) extended to 1s each (a tap records the keypress, not the behaviour). Event COUNT is unchanged | 6.13s total | 10.27s total |
| female2bottom000826_09_0312_01_40 | F2 | fast clicks merged | escape: 1 gap(s) <= 0.35s filled | 3 | 2 |
| female2bottom000826_09_0312_01_40 | F2 | metadata filled | sessionNo | (empty) | 8 |
| female2bottom000826_09_0312_01_40 | F2 | metadata filled | mouseID | (empty) | F2 |
| female2bottom000826_09_0312_01_40 | F2 | metadata filled | sexID | (empty) | F |
| female2bottom000826_09_0312_01_40 | F2 | metadata filled | phase | Baseline | Post-treatment |
| female2bottom000826_09_0312_01_40 | F2 | metadata filled | treatment | (empty) | SBI-553 |
| female2bottom000826_09_0312_01_40 | F2 | guardMin | tap scoring: a duration filter would discard nearly every mark | 2.0 | 0.0 |
| female3bottom000926_09_0312_31_11 | F3 | stimulus name spelling only | slot 3 | pin | Pin prick |
| female3bottom000926_09_0312_31_11 | F3 | stimulus name spelling only | slot 4 | heat | Heat |
| female3bottom000926_09_0312_31_11 | F3 | stimulus mis-key (clock window) | t=671.6s is in block 2 (660-960s) | Light touch | Mild touch |
| female3bottom000926_09_0312_31_11 | F3 | stimulus mis-key (clock window) | t=672.4s is in block 2 (660-960s) | Light touch | Mild touch |
| female3bottom000926_09_0312_31_11 | F3 | stimulus mis-key (clock window) | t=674.2s is in block 2 (660-960s) | Light touch | Mild touch |
| female3bottom000926_09_0312_31_11 | F3 | stimulus mis-key (clock window) | t=678.7s is in block 2 (660-960s) | Light touch | Mild touch |
| female3bottom000926_09_0312_31_11 | F3 | fast clicks merged | attending: 1 gap(s) <= 0.35s filled | 8 | 7 |
| female3bottom000926_09_0312_31_11 | F3 | fast clicks merged | guarding: 1 gap(s) <= 0.35s filled | 7 | 6 |
| female3bottom000926_09_0312_31_11 | F3 | nominal duration applied | guarding: 5 event(s) extended to 1s each (a tap records the keypress, not the behaviour). Event COUNT is unchanged | 2.87s total | 5.33s total |
| female3bottom000926_09_0312_31_11 | F3 | fast clicks merged | escape: 1 gap(s) <= 0.35s filled | 10 | 9 |
| female3bottom000926_09_0312_31_11 | F3 | metadata filled | sessionNo | (empty) | 9 |
| female3bottom000926_09_0312_31_11 | F3 | metadata filled | mouseID | (empty) | F3 |
| female3bottom000926_09_0312_31_11 | F3 | metadata filled | sexID | (empty) | F |
| female3bottom000926_09_0312_31_11 | F3 | metadata filled | phase | Baseline | Post-treatment |
| female3bottom000926_09_0312_31_11 | F3 | metadata filled | treatment | (empty) | SBI-553 |
| female3bottom000926_09_0312_31_11 | F3 | guardMin | tap scoring: a duration filter would discard nearly every mark | 2.0 | 0.0 |
| male1bottom001226_09_0314_00_32 | M1 | stimulus name spelling only | slot 2 | pin | Pin prick |
| male1bottom001226_09_0314_00_32 | M1 | stimulus name spelling only | slot 4 | mild | Mild touch |
| male1bottom001226_09_0314_00_32 | M1 | fast clicks merged | attending: 1 gap(s) <= 0.35s filled | 6 | 5 |
| male1bottom001226_09_0314_00_32 | M1 | nominal duration applied | guarding: 1 event(s) extended to 1s each (a tap records the keypress, not the behaviour). Event COUNT is unchanged | 0.30s total | 0.30s total |
| male1bottom001226_09_0314_00_32 | M1 | metadata filled | sessionNo | (empty) | 10 |
| male1bottom001226_09_0314_00_32 | M1 | metadata filled | mouseID | (empty) | M1 |
| male1bottom001226_09_0314_00_32 | M1 | metadata filled | sexID | (empty) | M |
| male1bottom001226_09_0314_00_32 | M1 | metadata filled | phase | Baseline | Post-treatment |
| male1bottom001226_09_0314_00_32 | M1 | metadata filled | treatment | (empty) | SBI-553 |
| male1bottom001226_09_0314_00_32 | M1 | guardMin | tap scoring: a duration filter would discard nearly every mark | 2.0 | 0.0 |
| male2bottom001126_09_0313_30_46 | M2 | stimulus name spelling only | slot 1 | heat | Heat |
| male2bottom001126_09_0313_30_46 | M2 | stimulus name spelling only | slot 2 | pin | Pin prick |
| male2bottom001126_09_0313_30_46 | M2 | stimulus name spelling only | slot 3 | mild | Mild touch |
| male2bottom001126_09_0313_30_46 | M2 | stimulus name spelling only | slot 4 | light | Light touch |
| male2bottom001126_09_0313_30_46 | M2 | stimulus mis-key (clock window) | t=881.1s is in block 2 (660-960s) | Mild touch | Pin prick |
| male2bottom001126_09_0313_30_46 | M2 | stimulus mis-key (clock window) | t=1284.2s is in block 3 (1020-1320s) | Light touch | Mild touch |
| male2bottom001126_09_0313_30_46 | M2 | stimulus mis-key (clock window) | t=1285.7s is in block 3 (1020-1320s) | Light touch | Mild touch |
| male2bottom001126_09_0313_30_46 | M2 | stimulus mis-key (clock window) | t=1286.7s is in block 3 (1020-1320s) | Light touch | Mild touch |
| male2bottom001126_09_0313_30_46 | M2 | stimulus mis-key (clock window) | t=1287.8s is in block 3 (1020-1320s) | Light touch | Mild touch |
| male2bottom001126_09_0313_30_46 | M2 | stimulus mis-key (clock window) | t=1289.6s is in block 3 (1020-1320s) | Light touch | Mild touch |
| male2bottom001126_09_0313_30_46 | M2 | fast clicks merged | guarding: 1 gap(s) <= 0.35s filled | 4 | 3 |
| male2bottom001126_09_0313_30_46 | M2 | nominal duration applied | guarding: 3 event(s) extended to 1s each (a tap records the keypress, not the behaviour). Event COUNT is unchanged | 1.27s total | 2.83s total |
| male2bottom001126_09_0313_30_46 | M2 | fast clicks merged | escape: 1 gap(s) <= 0.35s filled | 22 | 21 |
| male2bottom001126_09_0313_30_46 | M2 | metadata filled | sessionNo | (empty) | 11 |
| male2bottom001126_09_0313_30_46 | M2 | metadata filled | mouseID | (empty) | M2 |
| male2bottom001126_09_0313_30_46 | M2 | metadata filled | sexID | (empty) | M |
| male2bottom001126_09_0313_30_46 | M2 | metadata filled | phase | Baseline | Post-treatment |
| male2bottom001126_09_0313_30_46 | M2 | metadata filled | treatment | (empty) | SBI-553 |
| male2bottom001126_09_0313_30_46 | M2 | guardMin | tap scoring: a duration filter would discard nearly every mark | 2.0 | 0.0 |
| male3bottom001026_09_0313_01_11 | M3 | stimulus mis-key (clock window) | t=700.1s is in block 2 (660-960s) | Light touch | Mild touch |
| male3bottom001026_09_0313_01_11 | M3 | stimulus mis-key (clock window) | t=702.0s is in block 2 (660-960s) | Light touch | Mild touch |
| male3bottom001026_09_0313_01_11 | M3 | stimulus mis-key (clock window) | t=1304.6s is in block 3 (1020-1320s) | Pin prick | Heat |
| male3bottom001026_09_0313_01_11 | M3 | stimulus mis-key (clock window) | t=1305.1s is in block 3 (1020-1320s) | Pin prick | Heat |
| male3bottom001026_09_0313_01_11 | M3 | fast clicks merged | attending: 1 gap(s) <= 0.35s filled | 20 | 19 |
| male3bottom001026_09_0313_01_11 | M3 | fast clicks merged | guarding: 1 gap(s) <= 0.35s filled | 13 | 12 |
| male3bottom001026_09_0313_01_11 | M3 | nominal duration applied | guarding: 12 event(s) extended to 1s each (a tap records the keypress, not the behaviour). Event COUNT is unchanged | 5.23s total | 8.03s total |
| male3bottom001026_09_0313_01_11 | M3 | fast clicks merged | escape: 1 gap(s) <= 0.35s filled | 23 | 22 |
| male3bottom001026_09_0313_01_11 | M3 | metadata filled | sessionNo | (empty) | 12 |
| male3bottom001026_09_0313_01_11 | M3 | metadata filled | mouseID | (empty) | M3 |
| male3bottom001026_09_0313_01_11 | M3 | metadata filled | sexID | (empty) | M |
| male3bottom001026_09_0313_01_11 | M3 | metadata filled | phase | Baseline | Post-treatment |
| male3bottom001026_09_0313_01_11 | M3 | metadata filled | treatment | (empty) | SBI-553 |
| male3bottom001026_09_0313_01_11 | M3 | guardMin | tap scoring: a duration filter would discard nearly every mark | 2.0 | 0.0 |
