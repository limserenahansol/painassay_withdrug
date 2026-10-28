#!/bin/bash
# RUN_5941_redo_20261021.sh - 5941 again with exactly the 67 neurons you confirmed (#140, a rescued cell that came
# back after its duplicate partner was excluded, is now excluded too). Starts after the 1191 post (2 jobs at a time);
# the cohort waits for it (WAIT_5941.flag).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
OUT=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261019
T=/c/Users/hsollim/AppData/Local/Temp/claude/C--Users-hsollim/05a4ec72-aaab-4bd6-bf3c-2d10c312ad78/tasks
CONF=/c/Users/hsollim/AppData/Local/Temp/claude/C--Users-hsollim/05a4ec72-aaab-4bd6-bf3c-2d10c312ad78/scratchpad/confirmed_sets
L=$OUT/logs_v2_20261021
w() { cygpath -w "$1"; }
until grep -q "post 1191 end" $T/bcdzivrf1.output 2>/dev/null; do sleep 30; done
echo "post 5941 (redo) start $(date '+%F %T')"
bash $PIPE/ca_post_mouse.sh 5941 > $L/post_full_5941_redo.log 2>&1
python -I -c "
import pandas as pd, sys
a = set(pd.read_csv(sys.argv[1]).id); R = pd.read_csv(sys.argv[2]); b = set(R[R.keep == 1].id)
print('5941: confirmed %d, kept now %d, only confirmed %s, only now %s' % (len(a), len(b), sorted(a - b), sorted(b - a)))
" "$(w $CONF/5941_confirmed_ids.csv)" "$(w $OUT/5941_d1/4_cell_review/5941_d1_cell_review.csv)"
rm -f $OUT/WAIT_5941.flag
echo "post 5941 (redo) end $(date '+%F %T')"
