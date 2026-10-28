#!/bin/bash
# ca_post_mouse_day2.sh <mouse>  -  Day 2 (2026-10-21): everything after the Day-2 detection of one mouse,
# the same steps as Day 1 (ca_post_mouse.sh), with the Day-1 confirmed selection (green + orange you confirmed):
#   1. ca_day2_decisions.py  Day 1 -> Day 2 transform (rotation / scale / shift, mean images, refined on the pairs),
#      each Day-1 neuron followed to the Day-2 candidate at its place and judged on DAY-2 evidence (local Cn / std
#      pattern, the Day-2 movie proof, the Day-2 cell rules) - position and shape alone prove nothing because the
#      Day-1 neurons were forced seeds of the Day-2 EXTRACT (analysis\day2_seeds). Green = same neuron (your Day-1
#      keep), orange = Day-1 orange / to check / new on Day 2 (clear neuron shape only), magenta on the tracking map =
#      Day-1 neuron the Day-2 rules deleted or did not find (shown, never forced - your rule). Replaces the footprint-
#      overlap transfer + fingerprint pairing of the first version (those were circular with forced seeds).
#   1b. duplicates resolved (cos >= 0.7: your decision > SNR), then the Day-1 <-> Day-2 pairs refreshed
#   STOP (SELECTION_ONLY.flag in mini1p_out\20261021) after the max-projection maps, until you confirm the mouse
#      (<tag>_CONFIRMED.flag) - as on Day 1; <tag>_PROVISIONAL.flag runs everything before your confirmation
#   2-6. keep + merge (Day-2 final scorings), traces (3 versions), figures, QC sheet, max projection, GUI data, videos
m=$1
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
OUT=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261021
D1=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261019
SEEDS=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis/day2_seeds
SCORE='C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261021_final'
MATLAB="/c/Program Files/MATLAB/R2024b/bin/matlab.exe"
declare -A SEX=([5941]=F [5937]=M [5939]=M [1191]=F)
declare -A ARM=([5941]=vehicle [5937]=sbi553 [5939]=vehicle [1191]=sbi553)
w() { cygpath -w "$1"; }

tag=${m}_d2; R=$OUT/$tag; t1=${m}_d1
echo "#### post $tag $(date '+%F %T') ####"
A=$R/5_analysis; mkdir -p $A/decisions $A/traces $A/figures
python -I $PIPE/ca_day2_decisions.py --mouse $m || { echo "*** $tag: Day-2 decisions failed"; exit 1; }
python -I $PIPE/ca_duplicate_check.py --day-root "$(w $R)" --resolve 0.7 || echo "*** $tag: duplicate resolution failed"
# neighbours you kept as two Day-1 neurons stay two (round forced seeds 8-10 px apart overlap at cos ~0.57)
python -I $PIPE/ca_day2_decisions.py --mouse $m --fix-dedup || echo "*** $tag: duplicate correction failed"
python -I $PIPE/ca_day2_decisions.py --mouse $m --refresh-pairs || echo "*** $tag: pair refresh failed"
python -I $PIPE/ca_day2_decisions.py --mouse $m --flag-review || echo "*** $tag: review flags failed"   # attention only
# <tag>_PROVISIONAL.flag (overnight 10-21, your request "whole pipeline"): run everything, but the selection is
# still waiting for your confirmation - outputs are labelled provisional
if [ -f $OUT/SELECTION_ONLY.flag ] && [ ! -f $OUT/${tag}_CONFIRMED.flag ] && [ ! -f $OUT/${tag}_PROVISIONAL.flag ]; then
  python -I $PIPE/ca_day_maxproj.py --day-root "$(w $R)" --quadrants || echo "*** $tag: max projection figure failed"
  echo "#### $tag: selection only (SELECTION_ONLY.flag) - stopped before traces / analyses $(date '+%F %T') ####"
  exit 0
fi
"$MATLAB" -batch "addpath('$(w $PIPE)'); cfg = struct('mouse_id', '$m', 'day', 2, 'sex', '${SEX[$m]}', 'arm', '${ARM[$m]}', 'out_root', '$(w $OUT)', 'scoring_dir', '$SCORE', 'stages', {{'keep', 'merge'}}, 'merge_opts', struct('allow_count_mismatch', true)); run_mini1p_day(cfg); try, cfg.stages = {'classify'}; run_mini1p_day(cfg); catch e, fprintf(2, 'classify (mini2p reference) failed: %s\n', e.message); end" < /dev/null 2>&1 | grep -v "Name is nonexistent"
python -I $PIPE/ca_trace_outputs.py --extract "$(w $R/3_extract/${tag}_extract.mat)" \
    --keep "$(w $R/4_cell_review/${tag}_cell_review.csv)" --out-dir "$(w $A/traces)" || echo "*** $tag: traces failed"
python -I $PIPE/ca_duplicate_check.py --day-root "$(w $R)" --out "$(w $A/decisions)" || echo "*** $tag: duplicate check failed"
python -I $PIPE/ca_pair_flag.py --day-root "$(w $R)" || echo "*** $tag: pair flag failed"   # anti-correlated overlapping neighbours -> attention
python -I $PIPE/ca_background_signal.py --day-root "$(w $R)" && python -I $PIPE/ca_trace_outputs.py --extract "$(w $R/3_extract/${tag}_extract.mat)" \
    --keep "$(w $R/4_cell_review/${tag}_cell_review.csv)" --out-dir "$(w $A/traces_bgcorr)" --bg-regress "$(w $A/traces)" \
    || echo "*** $tag: background-corrected traces failed"
python -I $PIPE/ca_roi_traces.py --day-root "$(w $R)" || echo "*** $tag: ROI traces failed"
python -I $PIPE/ca_session_figures.py --extract "$(w $R/3_extract/${tag}_extract.mat)" \
    --keep "$(w $R/4_cell_review/${tag}_cell_review.csv)" --projections "$(w $R/3_extract/${tag}_projections.mat)" \
    --traces-dir "$(w $A/traces)" --scoring-dir "$SCORE" --out-dir "$(w $A/figures)" || echo "*** $tag: figures failed"
python -I $PIPE/ca_qc_sheet.py --day-root "$(w $R)" || echo "*** $tag: QC sheet failed"
python -I $PIPE/ca_day_maxproj.py --day-root "$(w $R)" --quadrants || echo "*** $tag: max projection figure failed"
python -I $PIPE/ca_confirm_sheet.py --mouse $m || echo "*** $tag: confirmation sheet failed"
python -I $PIPE/ca_tracking_validation.py --mice $m > $A/decisions/${tag}_tracking_validation.txt 2>&1 || echo "*** $tag: tracking validation failed"
"$MATLAB" -batch "addpath('$(w $PIPE)'); prepare_curation_data('$(w $R/3_extract/${tag}_extract.mat)', '$(w $R/4_cell_review/${tag}_cell_review.csv)', '$(w $R/3_extract/${tag}_projections.mat)', '$(w $R/4_cell_review/${tag}_curation_data.mat)');" < /dev/null 2>&1 | grep -E "prepare_curation|Error"
VID=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis/ca_videos_day2
mkdir -p $VID
for s in pre post; do
  [ -f $A/traces/${tag}_${s}_traces.npz ] || continue
  for st in "Heat" "Light touch" "Mild touch" "Pin prick" "movement onset"; do
    python -I $PIPE/ca_behaviour_video.py --day-root "$(w $R)" --rid ${tag}_$s --scoring-dir "$SCORE" \
        --out-dir "$(w $VID)" --auto --dur 40 --stim "$st" 2>&1 | grep -v Warning | tail -1 || echo "*** ${tag}_$s $st: video failed"
  done
done
for b in $R/1_movie/*_detect_binned_*.h5; do
  [ -f "$b" ] && { echo "[disk] $tag: deleting the detection movie $(basename $b) ($(du -h "$b" | cut -f1))"; rm -f "$b"; }
done
touch $OUT/${tag}_post_done.flag
echo "#### post $tag done $(date '+%F %T') ####"
