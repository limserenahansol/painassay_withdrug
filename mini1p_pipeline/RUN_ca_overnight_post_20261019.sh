#!/bin/bash
# RUN_ca_overnight_post_20261019.sh - 4th overnight worker: waits for each mouse's detection
# (<tag>_detect_done.flag) and then runs ca_post_mouse.sh for it (decision transfer, kept cells,
# behaviour merge, traces, figures). ca_post_mouse.sh is read fresh for every mouse.
#   bash RUN_ca_overnight_post_20261019.sh [mouse ...]      default: 5941 5937 5939 1191
OUT=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261019
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
MICE=${@:-5941 5937 5939 1191}
for m in $MICE; do
  tag=${m}_d1
  while [ ! -f $OUT/${tag}_detect_done.flag ]; do
    if ls $OUT/${tag}_*_failed.flag >/dev/null 2>&1; then echo "*** $tag: an earlier worker failed - skipped"; continue 2; fi
    sleep 60
  done
  bash $PIPE/ca_post_mouse.sh $m
done
echo "OVERNIGHT POST FINISHED $(date '+%F %T')"
