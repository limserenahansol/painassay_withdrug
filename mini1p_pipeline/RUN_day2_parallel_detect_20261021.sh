#!/bin/bash
# RUN_day2_parallel_detect_20261021.sh - a second detection worker for the LAST mouse (1191), started only when all
# motion correction is done (1191_d2_mc_ready.flag = the last MC) and the main detect loop is busy with another mouse:
# then exactly 2 heavy jobs run (two detections). The claim flag (1191_d2_detect_running.flag) is made here BEFORE the
# seeds, so the main loop's MATLAB skips 1191 when it gets there (guard in RUN_ca_overnight_20261021.m).
# If the main loop is idle at that moment it takes 1191 itself and this script does nothing.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
OUT=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261021
MATLAB="/c/Program Files/MATLAB/R2024b/bin/matlab.exe"
L=$OUT/logs
m=${1:-1191}; tag=${m}_d2
w() { cygpath -w "$1"; }
until [ -f $OUT/${tag}_mc_ready.flag ]; do
  ls $OUT/${tag}_*_failed.flag > /dev/null 2>&1 && { echo "$tag failed earlier - nothing to do"; exit 0; }
  sleep 30
done
[ -f $OUT/${tag}_detect_done.flag ] || [ -f $OUT/${tag}_detect_running.flag ] && { echo "$tag already done / running"; exit 0; }
busy=$(ls $OUT/*_d2_detect_running.flag 2> /dev/null | grep -v "/${tag}_" | wc -l)
[ "$busy" -eq 0 ] && { echo "main detect loop idle at $(date '+%T') - it takes $tag itself"; exit 0; }
touch $OUT/${tag}_detect_running.flag
echo "parallel detect $tag start $(date '+%F %T') (main loop busy: $(ls $OUT/*_d2_detect_running.flag | xargs -n1 basename | tr '\n' ' '))"
python -I $PIPE/ca_day2_seeds.py --mouse $m > $L/seeds_${tag}_parallel.log 2>&1 || echo "*** $tag: Day-1 seeds failed"
"$MATLAB" -batch "WORKER='detect'; M='$m'; CLAIMED=true; run('$(w $PIPE/RUN_ca_overnight_20261021.m)')" < /dev/null > $L/detect_${tag}_parallel.log 2>&1
echo "parallel detect $tag end $(date '+%F %T')"
