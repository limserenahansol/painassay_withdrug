#!/bin/bash
# RUN_redo_day2_20261022.sh <mouse> <suffix> <wait-flag> - Day 2 of one mouse again with corrected seeds:
# waits for <wait-flag> (another heavy job finishing, so that at most 2 run), keeps the first run as *<suffix>,
# re-registers the seeds (activity projections of the first run used when they disagree with the mean image),
# re-detects (pass 1 re-used), then the post-processing. WAIT_<mouse>.flag makes the cohort wait.
#   bash RUN_redo_day2_20261022.sh 5939 _v1_meanreg 5937_d2_detect_done.flag
m=$1; SUF=$2; WAITFOR=$3
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
OUT=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261021
CD=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis/crossday
MATLAB="/c/Program Files/MATLAB/R2024b/bin/matlab.exe"
L=$OUT/logs
w() { cygpath -w "$1"; }
tag=${m}_d2; R=$OUT/$tag
touch $OUT/WAIT_${m}.flag
until [ -f $OUT/$WAITFOR ] && [ ! -f $OUT/${tag}_detect_running.flag ]; do sleep 60; done
echo "$tag redo start $(date '+%F %T')"
for d in 3_extract 4_cell_review 5_analysis; do
  [ -d $R/$d ] && [ ! -d $R/${d}${SUF} ] && mv $R/$d $R/${d}${SUF}
done
mkdir -p $R/3_extract/pass1
rm -f $OUT/${tag}_detect_done.flag $OUT/${tag}_post_done.flag
for g in $CD/${m}_*; do
  case "$g" in *_v1_*) ;; *) [ -f "$g" ] && mv "$g" "${g%.*}${SUF}.${g##*.}";; esac
done
touch $OUT/${tag}_detect_running.flag
python -I $PIPE/ca_day2_seeds.py --mouse $m > $L/seeds_${tag}_redo.log 2>&1
cat $L/seeds_${tag}_redo.log
"$MATLAB" -batch "M='$m'; SUF='$SUF'; run('$(w $PIPE/RUN_ca_redo_day2_20261022.m)')" < /dev/null > $L/detect_${tag}_redo.log 2>&1
grep -a -E "STOPPED|done in|candidates" $L/detect_${tag}_redo.log | tail -3
if [ -f $OUT/${tag}_detect_done.flag ]; then
  bash $PIPE/ca_post_mouse_day2.sh $m > $L/post_${tag}_redo.log 2>&1
  echo "$tag redo post end $(date '+%F %T')"
fi
rm -f $OUT/WAIT_${m}.flag
