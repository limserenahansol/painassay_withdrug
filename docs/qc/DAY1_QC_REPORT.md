# QC and correction report - Day 1 (no drug)

- originals: `C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos\output` **unmodified**
- corrected: `C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos\output_corrected`
- merge gap 0.35 s; blocks assigned by clock window 5-10 / 11-16 / 17-22 / 23-28 min
- guarding events given a nominal 1 s each (a tap records the keypress, not the behaviour); event counts unchanged
- stimulus names: F1=sheet, F2=sheet, F3=sheet, M1=typed, M2=sheet, M3=typed
- 6 session(s), 93 change(s), 2 flag(s)

Stimulus identity comes from the names TYPED WHILE SCORING, not from the randomisation sheet, because the delivery order on the day did not follow the sheet. Only the spelling was normalised.

## Session facts

| mouse | session | file | dur_s | n_del | runs_after_fix | n_miskey_fixed | labelled_frames | stimuli |
|---|---|---|---|---|---|---|---|---|
| F1 | 1 | female1bottom000126_09_0212_00_19 | 1680.9 | 87 | 4 | 0 | 2594 | Mild touch:26 / Pin prick:16 / Light touch:22 / Heat:23 |
| F2 | 2 | female2bottom000226_09_0212_33_04_009 | 1680.3 | 88 | 4 | 9 | 2856 | Heat:20 / Mild touch:20 / Pin prick:31 / Light touch:17 |
| F3 | 3 | female3bottom000326_09_0213_03_46_008 | 1691.9 | 85 | 4 | 0 | 4117 | Heat:15 / Light touch:30 / Pin prick:21 / Mild touch:19 |
| M1 | 4 | male1bottom000626_09_0214_37_16_006 | 1692.4 | 100 | 4 | 0 | 5900 | Mild touch:26 / Light touch:28 / Heat:22 / Pin prick:24 |
| M2 | 5 | male2bottom000526_09_0214_06_31_004 | 1692.1 | 83 | 4 | 3 | 6572 | Pin prick:23 / Mild touch:21 / Heat:16 / Light touch:23 |
| M3 | 6 | male3bottom000426_09_0213_35_27_011 | 1722.3 | 91 | 4 | 6 | 4953 | Pin prick:22 / Mild touch:24 / Light touch:25 / Heat:20 |

## Behaviour events after merging fast clicks

| mouse | attending_before_merge | attending_events | escape_before_merge | escape_events | guarding_before_merge | guarding_events | lickbite_before_merge | lickbite_events |
|---|---|---|---|---|---|---|---|---|
| F1 | 14 | 10 | 104 | 86 | 2 | 2 | 49 | 20 |
| F2 | 33 | 19 | 100 | 70 | 30 | 24 | 79 | 51 |
| F3 | 24 | 15 | 135 | 101 | 34 | 25 | 98 | 56 |
| M1 | 35 | 32 | 69 | 69 | 32 | 29 | 26 | 21 |
| M2 | 40 | 37 | 114 | 105 | 26 | 21 | 19 | 19 |
| M3 | 44 | 43 | 58 | 57 | 29 | 24 | 24 | 21 |

## FLAGGED - not changed, needs your decision

| file | mouse | kind | detail | action |
|---|---|---|---|---|
| male1bottom000626_09_0214_37_16_006 | M1 | names kept as TYPED, sheet differs | typed ['Mild touch', 'Light touch', 'Heat', 'Pin prick'] vs sheet ['Pin prick', 'Mild touch', 'Light touch', 'Heat'] - kept the typed names on your instruction | no change - this is the chosen source |
| male3bottom000426_09_0213_35_27_011 | M3 | names kept as TYPED, sheet differs | typed ['Pin prick', 'Mild touch', 'Light touch', 'Heat'] vs sheet ['Mild touch', 'Light touch', 'Heat', 'Pin prick'] - kept the typed names on your instruction | no change - this is the chosen source |

## Every change made

| file | mouse | kind | detail | before | after |
|---|---|---|---|---|---|
| female1bottom000126_09_0212_00_19 | F1 | fast clicks merged | attending: 4 gap(s) <= 0.35s filled | 14 | 10 |
| female1bottom000126_09_0212_00_19 | F1 | fast clicks merged | lickbite: 29 gap(s) <= 0.35s filled | 49 | 20 |
| female1bottom000126_09_0212_00_19 | F1 | nominal duration applied | guarding: 2 event(s) extended to 1s each (a tap records the keypress, not the behaviour). Event COUNT is unchanged | 0.57s total | 2.00s total |
| female1bottom000126_09_0212_00_19 | F1 | fast clicks merged | escape: 18 gap(s) <= 0.35s filled | 104 | 86 |
| female1bottom000126_09_0212_00_19 | F1 | metadata filled | sessionNo | (empty) | 1 |
| female1bottom000126_09_0212_00_19 | F1 | metadata filled | mouseID | (empty) | F1 |
| female1bottom000126_09_0212_00_19 | F1 | metadata filled | sexID | (empty) | F |
| female1bottom000126_09_0212_00_19 | F1 | metadata filled | dayNo | (empty) | 1 |
| female1bottom000126_09_0212_00_19 | F1 | metadata filled | phase | Baseline | Baseline (no injection) |
| female1bottom000126_09_0212_00_19 | F1 | metadata filled | treatment | (empty) | None |
| female1bottom000126_09_0212_00_19 | F1 | guardMin | tap scoring: a duration filter would discard nearly every mark | 2.0 | 0.0 |
| female2bottom000226_09_0212_33_04_009 | F2 | stimulus mis-key (clock window) | t=303.9s is in block 1 (300-600s) | Mild touch | Heat |
| female2bottom000226_09_0212_33_04_009 | F2 | stimulus mis-key (clock window) | t=385.5s is in block 1 (300-600s) | Mild touch | Heat |
| female2bottom000226_09_0212_33_04_009 | F2 | stimulus mis-key (clock window) | t=423.5s is in block 1 (300-600s) | Mild touch | Heat |
| female2bottom000226_09_0212_33_04_009 | F2 | stimulus mis-key (clock window) | t=496.3s is in block 1 (300-600s) | Mild touch | Heat |
| female2bottom000226_09_0212_33_04_009 | F2 | stimulus mis-key (clock window) | t=574.0s is in block 1 (300-600s) | Mild touch | Heat |
| female2bottom000226_09_0212_33_04_009 | F2 | stimulus mis-key (clock window) | t=577.7s is in block 1 (300-600s) | Mild touch | Heat |
| female2bottom000226_09_0212_33_04_009 | F2 | stimulus mis-key (clock window) | t=737.5s is in block 2 (660-960s) | Heat | Mild touch |
| female2bottom000226_09_0212_33_04_009 | F2 | stimulus mis-key (clock window) | t=807.5s is in block 2 (660-960s) | Heat | Mild touch |
| female2bottom000226_09_0212_33_04_009 | F2 | stimulus mis-key (clock window) | t=810.7s is in block 2 (660-960s) | Heat | Mild touch |
| female2bottom000226_09_0212_33_04_009 | F2 | fast clicks merged | attending: 14 gap(s) <= 0.35s filled | 33 | 19 |
| female2bottom000226_09_0212_33_04_009 | F2 | fast clicks merged | lickbite: 28 gap(s) <= 0.35s filled | 79 | 51 |
| female2bottom000226_09_0212_33_04_009 | F2 | fast clicks merged | guarding: 6 gap(s) <= 0.35s filled | 30 | 24 |
| female2bottom000226_09_0212_33_04_009 | F2 | nominal duration applied | guarding: 21 event(s) extended to 1s each (a tap records the keypress, not the behaviour). Event COUNT is unchanged | 10.70s total | 17.60s total |
| female2bottom000226_09_0212_33_04_009 | F2 | fast clicks merged | escape: 30 gap(s) <= 0.35s filled | 100 | 70 |
| female2bottom000226_09_0212_33_04_009 | F2 | metadata filled | sessionNo | (empty) | 2 |
| female2bottom000226_09_0212_33_04_009 | F2 | metadata filled | mouseID | (empty) | F2 |
| female2bottom000226_09_0212_33_04_009 | F2 | metadata filled | sexID | (empty) | F |
| female2bottom000226_09_0212_33_04_009 | F2 | metadata filled | dayNo | (empty) | 1 |
| female2bottom000226_09_0212_33_04_009 | F2 | metadata filled | phase | Baseline | Baseline (no injection) |
| female2bottom000226_09_0212_33_04_009 | F2 | metadata filled | treatment | (empty) | None |
| female2bottom000226_09_0212_33_04_009 | F2 | guardMin | tap scoring: a duration filter would discard nearly every mark | 2.0 | 0.0 |
| female3bottom000326_09_0213_03_46_008 | F3 | stimulus name from sheet | slot 2 (block 2 of the session) | Mild touch | Light touch |
| female3bottom000326_09_0213_03_46_008 | F3 | fast clicks merged | attending: 9 gap(s) <= 0.35s filled | 24 | 15 |
| female3bottom000326_09_0213_03_46_008 | F3 | fast clicks merged | lickbite: 42 gap(s) <= 0.35s filled | 98 | 56 |
| female3bottom000326_09_0213_03_46_008 | F3 | fast clicks merged | guarding: 9 gap(s) <= 0.35s filled | 34 | 25 |
| female3bottom000326_09_0213_03_46_008 | F3 | nominal duration applied | guarding: 22 event(s) extended to 1s each (a tap records the keypress, not the behaviour). Event COUNT is unchanged | 11.17s total | 16.83s total |
| female3bottom000326_09_0213_03_46_008 | F3 | fast clicks merged | escape: 34 gap(s) <= 0.35s filled | 135 | 101 |
| female3bottom000326_09_0213_03_46_008 | F3 | metadata filled | sessionNo | (empty) | 3 |
| female3bottom000326_09_0213_03_46_008 | F3 | metadata filled | mouseID | (empty) | F3 |
| female3bottom000326_09_0213_03_46_008 | F3 | metadata filled | sexID | (empty) | F |
| female3bottom000326_09_0213_03_46_008 | F3 | metadata filled | dayNo | (empty) | 1 |
| female3bottom000326_09_0213_03_46_008 | F3 | metadata filled | phase | Baseline | Baseline (no injection) |
| female3bottom000326_09_0213_03_46_008 | F3 | metadata filled | treatment | (empty) | None |
| female3bottom000326_09_0213_03_46_008 | F3 | guardMin | tap scoring: a duration filter would discard nearly every mark | 2.0 | 0.0 |
| male1bottom000626_09_0214_37_16_006 | M1 | stimulus name spelling only | slot 1 | mild touch | Mild touch |
| male1bottom000626_09_0214_37_16_006 | M1 | stimulus name spelling only | slot 2 | light | Light touch |
| male1bottom000626_09_0214_37_16_006 | M1 | fast clicks merged | attending: 3 gap(s) <= 0.35s filled | 35 | 32 |
| male1bottom000626_09_0214_37_16_006 | M1 | fast clicks merged | lickbite: 5 gap(s) <= 0.35s filled | 26 | 21 |
| male1bottom000626_09_0214_37_16_006 | M1 | fast clicks merged | guarding: 3 gap(s) <= 0.35s filled | 32 | 29 |
| male1bottom000626_09_0214_37_16_006 | M1 | nominal duration applied | guarding: 18 event(s) extended to 1s each (a tap records the keypress, not the behaviour). Event COUNT is unchanged | 28.37s total | 35.73s total |
| male1bottom000626_09_0214_37_16_006 | M1 | metadata filled | sessionNo | (empty) | 4 |
| male1bottom000626_09_0214_37_16_006 | M1 | metadata filled | mouseID | (empty) | M1 |
| male1bottom000626_09_0214_37_16_006 | M1 | metadata filled | sexID | (empty) | M |
| male1bottom000626_09_0214_37_16_006 | M1 | metadata filled | dayNo | (empty) | 1 |
| male1bottom000626_09_0214_37_16_006 | M1 | metadata filled | phase | Baseline | Baseline (no injection) |
| male1bottom000626_09_0214_37_16_006 | M1 | metadata filled | treatment | (empty) | None |
| male1bottom000626_09_0214_37_16_006 | M1 | guardMin | tap scoring: a duration filter would discard nearly every mark | 2.0 | 0.0 |
| male2bottom000526_09_0214_06_31_004 | M2 | stimulus mis-key (clock window) | t=1277.0s is in block 3 (1020-1320s) | Light touch | Heat |
| male2bottom000526_09_0214_06_31_004 | M2 | stimulus mis-key (clock window) | t=1278.2s is in block 3 (1020-1320s) | Light touch | Heat |
| male2bottom000526_09_0214_06_31_004 | M2 | stimulus mis-key (clock window) | t=1279.7s is in block 3 (1020-1320s) | Light touch | Heat |
| male2bottom000526_09_0214_06_31_004 | M2 | fast clicks merged | attending: 3 gap(s) <= 0.35s filled | 40 | 37 |
| male2bottom000526_09_0214_06_31_004 | M2 | fast clicks merged | guarding: 5 gap(s) <= 0.35s filled | 26 | 21 |
| male2bottom000526_09_0214_06_31_004 | M2 | nominal duration applied | guarding: 14 event(s) extended to 1s each (a tap records the keypress, not the behaviour). Event COUNT is unchanged | 20.03s total | 25.07s total |
| male2bottom000526_09_0214_06_31_004 | M2 | fast clicks merged | escape: 9 gap(s) <= 0.35s filled | 114 | 105 |
| male2bottom000526_09_0214_06_31_004 | M2 | metadata filled | sessionNo | (empty) | 5 |
| male2bottom000526_09_0214_06_31_004 | M2 | metadata filled | mouseID | (empty) | M2 |
| male2bottom000526_09_0214_06_31_004 | M2 | metadata filled | sexID | (empty) | M |
| male2bottom000526_09_0214_06_31_004 | M2 | metadata filled | dayNo | (empty) | 1 |
| male2bottom000526_09_0214_06_31_004 | M2 | metadata filled | phase | Baseline | Baseline (no injection) |
| male2bottom000526_09_0214_06_31_004 | M2 | metadata filled | treatment | (empty) | None |
| male2bottom000526_09_0214_06_31_004 | M2 | guardMin | tap scoring: a duration filter would discard nearly every mark | 2.0 | 0.0 |
| male3bottom000426_09_0213_35_27_011 | M3 | stimulus name spelling only | slot 1 | pin prick | Pin prick |
| male3bottom000426_09_0213_35_27_011 | M3 | stimulus name spelling only | slot 3 | light | Light touch |
| male3bottom000426_09_0213_35_27_011 | M3 | stimulus name spelling only | slot 4 | heat | Heat |
| male3bottom000426_09_0213_35_27_011 | M3 | stimulus mis-key (clock window) | t=371.2s is in block 1 (300-600s) | Mild touch | Pin prick |
| male3bottom000426_09_0213_35_27_011 | M3 | stimulus mis-key (clock window) | t=1022.5s is in block 3 (1020-1320s) | Mild touch | Light touch |
| male3bottom000426_09_0213_35_27_011 | M3 | stimulus mis-key (clock window) | t=1026.0s is in block 3 (1020-1320s) | Mild touch | Light touch |
| male3bottom000426_09_0213_35_27_011 | M3 | stimulus mis-key (clock window) | t=1027.4s is in block 3 (1020-1320s) | Mild touch | Light touch |
| male3bottom000426_09_0213_35_27_011 | M3 | stimulus mis-key (clock window) | t=1030.0s is in block 3 (1020-1320s) | Mild touch | Light touch |
| male3bottom000426_09_0213_35_27_011 | M3 | stimulus mis-key (clock window) | t=1031.3s is in block 3 (1020-1320s) | Mild touch | Light touch |
| male3bottom000426_09_0213_35_27_011 | M3 | fast clicks merged | attending: 1 gap(s) <= 0.35s filled | 44 | 43 |
| male3bottom000426_09_0213_35_27_011 | M3 | fast clicks merged | lickbite: 3 gap(s) <= 0.35s filled | 24 | 21 |
| male3bottom000426_09_0213_35_27_011 | M3 | fast clicks merged | guarding: 5 gap(s) <= 0.35s filled | 29 | 24 |
| male3bottom000426_09_0213_35_27_011 | M3 | nominal duration applied | guarding: 20 event(s) extended to 1s each (a tap records the keypress, not the behaviour). Event COUNT is unchanged | 11.73s total | 17.43s total |
| male3bottom000426_09_0213_35_27_011 | M3 | fast clicks merged | escape: 1 gap(s) <= 0.35s filled | 58 | 57 |
| male3bottom000426_09_0213_35_27_011 | M3 | metadata filled | sessionNo | (empty) | 6 |
| male3bottom000426_09_0213_35_27_011 | M3 | metadata filled | mouseID | (empty) | M3 |
| male3bottom000426_09_0213_35_27_011 | M3 | metadata filled | sexID | (empty) | M |
| male3bottom000426_09_0213_35_27_011 | M3 | metadata filled | dayNo | (empty) | 1 |
| male3bottom000426_09_0213_35_27_011 | M3 | metadata filled | phase | Baseline | Baseline (no injection) |
| male3bottom000426_09_0213_35_27_011 | M3 | metadata filled | treatment | (empty) | None |
| male3bottom000426_09_0213_35_27_011 | M3 | guardMin | tap scoring: a duration filter would discard nearly every mark | 2.0 | 0.0 |
