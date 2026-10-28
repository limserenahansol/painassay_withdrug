#!/bin/bash
# RUN_day2_detect_loop2_20261021.sh - replaces the detect loop of RUN_day2_overnight_20261021.sh (2026-10-21 20:40):
# the first 5941 detection collided with a leftover waiter of the old 5941 chain (stopped, its MATLAB killed), so 5941
# is detected again here, then 5937, 5939, 1191 as before (seeds -> EXTRACT + review). The guard in
# RUN_ca_overnight_20261021.m (<tag>_detect_running.flag) keeps the parallel 1191 worker and this loop apart.
# The overnight post loop had already skipped 5941, so 5941's post-processing is started here (in the background).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
OUT=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261021
MATLAB="/c/Program Files/MATLAB/R2024b/bin/matlab.exe"
L=$OUT/logs
w() { cygpath -w "$1"; }
RUNM=$(w $PIPE/RUN_ca_overnight_20261021.m)
for m in 5941 5937 5939 1191; do
  tag=${m}_d2
  until [ -f $OUT/${tag}_mc_ready.flag ]; do
    ls $OUT/${tag}_*_failed.flag > /dev/null 2>&1 && { echo "*** $tag: copy / mc failed - skipped"; continue 2; }
    sleep 60
  done
  if [ ! -f $OUT/${tag}_detect_done.flag ] && [ ! -f $OUT/${tag}_detect_running.flag ]; then
    echo "detect $tag start $(date '+%F %T')"
    python -I $PIPE/ca_day2_seeds.py --mouse $m > $L/seeds_$tag.log 2>&1 || echo "*** $tag: Day-1 seeds failed"
    "$MATLAB" -batch "WORKER='detect'; M='$m'; run('$RUNM')" < /dev/null > $L/detect_${tag}.log 2>&1
    echo "detect $tag end $(date '+%F %T')"
  else
    echo "$tag: detection done / running elsewhere"
  fi
  if [ "$m" = 5941 ] && [ -f $OUT/${tag}_detect_done.flag ]; then
    ( echo "post $tag start $(date '+%F %T')"; bash $PIPE/ca_post_mouse_day2.sh $m > $L/post_$tag.log 2>&1
      echo "post $tag end $(date '+%F %T')" ) &
  fi
done
wait
echo "detect loop 2 finished $(date '+%F %T')"
