#!/bin/bash
# RUN_day2_finish_20261022.sh - replaces the post loop of RUN_day2_overnight_20261021.sh for the last steps:
#   1191: when its detection is done, check the seed registration against the activity projections
#         (ca_day2_seeds.register_best). Disagreement > 6 px -> redo 1191 (after the 5939 redo detection), else post.
#   then the Day-2 cohort analyses + the longitudinal analysis (they wait for every WAIT_<mouse>.flag),
#   the behaviour crossover analysis and all decks (EN + KR).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
OUT=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261021
L=$OUT/logs
until [ -f $OUT/1191_d2_detect_done.flag ] || [ -f $OUT/1191_d2_detect_failed.flag ]; do sleep 20; done
echo "1191 detection end seen $(date '+%F %T')"
dis=$(python -I -c "
import sys; sys.path.insert(0, r'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_pipeline')
import os, scipy.io as sio, ca_day2_seeds as SD
B = r'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_out'
g = lambda d, t, f, v: sio.loadmat(os.path.join(B, d, t, '3_extract', f), variable_names=[v])[v]
r = SD.register_best(g('20261019', '1191_d1', '1191_d1_extract.mat', 'mean_image'), g('20261021', '1191_d2', '1191_d2_extract.mat', 'mean_image'),
                     g('20261019', '1191_d1', '1191_d1_projections.mat', 'std_dff'), g('20261021', '1191_d2', '1191_d2_projections.mat', 'std_dff'))
print(r.get('disagreement_px', 0))
" 2> $L/finish_regcheck.err)
echo "1191: mean vs activity registration disagree by ${dis} px"
if python -I -c "import sys; sys.exit(0 if float('${dis:-0}') > 6 else 1)"; then
  echo "1191: redo with the activity registration"
  bash $PIPE/RUN_redo_day2_20261022.sh 1191 _v1_meanreg 5939_d2_detect_done.flag > $L/redo_1191.log 2>&1
else
  echo "post 1191_d2 start $(date '+%F %T')"
  bash $PIPE/ca_post_mouse_day2.sh 1191 > $L/post_1191_d2.log 2>&1
  echo "post 1191_d2 end $(date '+%F %T')"
fi
echo "cohort start $(date '+%F %T') (waits for WAIT flags: $(ls $OUT/WAIT_*.flag 2> /dev/null | xargs -n1 basename | tr '\n' ' '))"
CA_FORCE_COHORT=1 bash $PIPE/ca_post_cohort_day2.sh > $L/cohort_day2.log 2>&1
echo "cohort end $(date '+%F %T')"
python -I $PIPE/behavior_crossover_analysis.py > $L/behavior_crossover.log 2>&1
python -I $PIPE/ca_tracking_validation.py > $L/tracking_validation_all.log 2>&1
for m in 5941 5937 5939 1191; do python -I $PIPE/ca_confirm_sheet.py --mouse $m >> $L/confirm_sheets.log 2>&1; done
bash $PIPE/RUN_all_decks.sh day2 > $L/decks_day2.log 2>&1
bash $PIPE/RUN_all_decks.sh long > $L/decks_long.log 2>&1
echo "all done $(date '+%F %T')"
