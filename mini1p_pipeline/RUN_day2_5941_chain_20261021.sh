#!/bin/bash
# RUN_day2_5941_chain_20261021.sh - after the Day-2 5941 motion correction: Day-1 neurons as seeds, then detection.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
OUT=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261021
until [ -f $OUT/5941_d2_mc_ready.flag ] || [ -f $OUT/5941_d2_mc_failed.flag ]; do sleep 30; done
[ -f $OUT/5941_d2_mc_failed.flag ] && { echo "*** mc failed"; exit 1; }
echo "mc done $(date '+%F %T')"
python -I $PIPE/ca_day2_seeds.py --mouse 5941 || { echo "*** seeds failed"; exit 1; }
"/c/Program Files/MATLAB/R2024b/bin/matlab" -batch "M='5941'; STAGE='detect'; run('C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_pipeline\RUN_ca_day2_20261021.m')" > $OUT/logs/detect_5941_d2.log 2>&1
grep -E "done in|STOPPED|candidates|seeds" $OUT/logs/detect_5941_d2.log | tail -5
echo "detect end $(date '+%F %T')"
# selection for your confirmation (stops at the max-projection maps: SELECTION_ONLY.flag)
bash $PIPE/ca_post_mouse_day2.sh 5941 > $OUT/logs/post_5941_d2.log 2>&1
grep -E "transfer|kept|attention|selection only|\*\*\*" $OUT/logs/post_5941_d2.log | tail -6
echo "selection maps end $(date '+%F %T')"
