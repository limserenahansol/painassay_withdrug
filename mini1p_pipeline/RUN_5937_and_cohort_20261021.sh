#!/bin/bash
# RUN_5937_and_cohort_20261021.sh - after the 5937 selection: the same rule as you chose for the other three
# (green + orange kept, rescued excluded), its traces / figures / videos, then the cohort analyses on all
# 4 mice (COHORT_HOLD released). Concurrency: 5937 starts after the 5939 post (with 1191 still running).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
OUT=$BASE/mini1p_out/20261019
T=/c/Users/hsollim/AppData/Local/Temp/claude/C--Users-hsollim/05a4ec72-aaab-4bd6-bf3c-2d10c312ad78/tasks
CONF=/c/Users/hsollim/AppData/Local/Temp/claude/C--Users-hsollim/05a4ec72-aaab-4bd6-bf3c-2d10c312ad78/scratchpad/confirmed_sets
DRV=$T/blhp7elw9.output; DS=$T/bcdzivrf1.output
L=$OUT/logs_v2_20261021
w() { cygpath -w "$1"; }
until grep -q "post 5937 end\|5937: detection failed" $DRV 2>/dev/null; do sleep 30; done
echo "5937 selection ready $(date '+%F %T')"
R=$OUT/5937_d1
IDS=$(python -I -c "
import pandas as pd, sys
R = pd.read_csv(sys.argv[1]); print(' '.join(str(int(i)) for i in R[(R.keep == 1) & (R.verdict == 'rescued')].id))" "$(w $R/4_cell_review/5937_d1_cell_review.csv)")
[ -n "$IDS" ] && python -I $PIPE/ca_exclude_rois.py --day-root "$(w $R)" --ids $IDS
python -I $PIPE/ca_day_maxproj.py --day-root "$(w $R)" --quadrants | head -1
python -I -c "
import pandas as pd, sys
R = pd.read_csv(sys.argv[1]); R[R.keep == 1][['id']].to_csv(sys.argv[2], index=False); print('5937 kept', int(R.keep.sum()))
" "$(w $R/4_cell_review/5937_d1_cell_review.csv)" "$(w $CONF/5937_confirmed_ids.csv)"
touch $OUT/5937_d1_CONFIRMED.flag
echo "5937 maps ready $(date '+%F %T')"
until grep -q "post 5939 end" $DS 2>/dev/null; do sleep 30; done
echo "post 5937 start $(date '+%F %T')"
bash $PIPE/ca_post_mouse.sh 5937 > $L/post_full_5937.log 2>&1
python -I -c "
import pandas as pd, sys
a = set(pd.read_csv(sys.argv[1]).id); R = pd.read_csv(sys.argv[2]); b = set(R[R.keep == 1].id)
print('5937: confirmed %d, kept now %d, only confirmed %s, only now %s' % (len(a), len(b), sorted(a - b), sorted(b - a)))
" "$(w $CONF/5937_confirmed_ids.csv)" "$(w $R/4_cell_review/5937_d1_cell_review.csv)"
echo "post 5937 end $(date '+%F %T')"
until grep -q "confirmed downstream + cohort (3 mice) done" $DS 2>/dev/null; do sleep 30; done
echo "cohort (4 mice) start $(date '+%F %T')"
CA_COHORT_RELEASE=1 CA_FORCE_COHORT=1 bash $PIPE/ca_post_cohort.sh > $L/cohort_4mice.log 2>&1
echo "#### cohort (4 mice) done $(date '+%F %T') ####"
