#!/bin/bash
# RUN_confirmed_downstream_20261021.sh - traces, figures, QC, videos for the mice whose whole-day selection
# you confirmed on the max projection (5941, 5939, 1191; green + orange, rescued excluded), then the
# cohort analyses on them (5937 is added when you confirm it). At most 2 heavy jobs at a time
# (5937 detection is still running): 5941 now, 5939 after the 5937 detection, 1191 after 5941.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
OUT=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261019
CONF=/c/Users/hsollim/AppData/Local/Temp/claude/C--Users-hsollim/05a4ec72-aaab-4bd6-bf3c-2d10c312ad78/scratchpad/confirmed_sets
L=$OUT/logs_v2_20261021
w() { cygpath -w "$1"; }
check() {   # the kept set after the post script must be the one you confirmed
  python -I -c "
import pandas as pd, sys
a = set(pd.read_csv(sys.argv[1]).id); R = pd.read_csv(sys.argv[2]); b = set(R[R.keep == 1].id)
print('$1: confirmed %d, kept now %d, only confirmed %s, only now %s' % (len(a), len(b), sorted(a - b), sorted(b - a)))
" "$(w $CONF/${1}_confirmed_ids.csv)" "$(w $OUT/${1}_d1/4_cell_review/${1}_d1_cell_review.csv)"
}
post() {
  echo "post $1 start $(date '+%F %T')"
  bash $PIPE/ca_post_mouse.sh $1 > $L/post_full_$1.log 2>&1
  check $1
  echo "post $1 end $(date '+%F %T')"
}
echo "#### confirmed downstream start $(date '+%F %T') ####"
( post 5941; post 1191 ) &
( until [ -f $OUT/5937_d1_v2_detect_done.flag ] || [ -f $OUT/5937_d1_v2_detect_failed.flag ]; do sleep 30; done; post 5939 ) &
wait
CA_FORCE_COHORT=1 bash $PIPE/ca_post_cohort.sh > $L/cohort_3mice.log 2>&1
echo "#### confirmed downstream + cohort (3 mice) done $(date '+%F %T') ####"
