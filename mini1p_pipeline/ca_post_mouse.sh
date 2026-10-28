#!/bin/bash
# ca_post_mouse.sh <mouse>  -  everything after the whole-day detection of one mouse (day 1)
#   1. ca_transfer_decisions.py  your decisions -> the whole-day candidates (5941: the 5-min test;
#      the others: their 1-min previews once you have reviewed them; until then the rules)
#   1b. rescue_missing_cells.m  cells you kept that the whole-day EXTRACT did not find again: your
#      footprint is added and every trace re-fitted jointly (least squares), then 1. again
#   1c. ca_duplicate_check.py --resolve 0.7  near-identical kept footprints: one neuron, one kept
#   2. run_mini1p_day {keep, merge, classify}  kept cells, behaviour merge (final scorings; a camera
#      that stopped a frame before the Inscopix is allowed, the counts are printed),
#      the mini2p classify_pain_neurons (reference)
#   3. ca_trace_outputs.py   local dF/F, z, deconvolution, events, per-cell summary, trace QC
#   3b. ca_duplicate_check.py pairs of kept cells that may be ONE neuron (strong footprint overlap)
#   3c. ca_background_signal.py + ca_trace_outputs --bg-regress -> traces_bgcorr (sensitivity version)
#   4. ca_session_figures.py cell map, every trace with behaviour, heatmaps
#   5. ca_qc_sheet.py        one QC page per recording (motion, filtering, cells, decisions, traces)
#   6. ca_behaviour_video.py side | bottom | Ca movie + traces, 40 s of the Heat (or Pin) block
# Safe to run again (e.g. after you review the previews): every step is re-done from the files.
m=$1
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
OUT=${CA_OUT:-/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261019}     # CA_OUT / CA_VID: tests only
PREV=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261019_preview3min
TEST=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/test_5941_5min
MARKS=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis/curation_marks_20261021
SCORE='C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261019_final'
MATLAB="/c/Program Files/MATLAB/R2024b/bin/matlab.exe"
declare -A SEX=([5941]=F [5937]=M [5939]=M [1191]=F)
declare -A ARM=([5941]=sbi553 [5937]=vehicle [5939]=sbi553 [1191]=vehicle)
w() { cygpath -w "$1"; }

tag=${m}_d1; R=$OUT/$tag
echo "#### post $tag $(date '+%F %T') ####"
A=$R/5_analysis; mkdir -p $A/decisions $A/traces $A/figures
REFS=""
if [ $m = 5941 ]; then
  REFS="--ref $(w $TEST/4c_extract_seeded_loose/5941_d1_extract.mat) $(w $TEST/5c_cell_review_loose/5941_d1_cell_review.csv) 5min_test"
fi
# your final preview decision (2026-10-21, reviewed on the max projection): <preview>/FINAL_VARIANT.txt
# names the subfolder; only the cells kept there count as your decisions (*_cell_review_user.csv),
# your circled marks are kept, and new candidates need a clear neuron shape (--strict-new)
FV=$PREV/${tag}_pre/FINAL_VARIANT.txt
if [ -f $FV ]; then
  v=$(tr -d ' \r\n' < $FV); p=$PREV/${tag}_pre/$v
  REFS="$REFS --ref $(w $p/3_extract/${tag}_extract.mat) $(w $p/4_review/${tag}_cell_review_user.csv) preview_pre_$v --strict-new"
  MK=$MARKS/marks_day_${tag}.csv
  [ -f $MK ] && REFS="$REFS --marks-day $(w $MK)"
else
  for s in pre post; do
    p=$PREV/${tag}_$s
    [ -f $p/3_extract/${tag}_extract.mat ] && REFS="$REFS --ref $(w $p/3_extract/${tag}_extract.mat) $(w $p/4_review/${tag}_cell_review.csv) preview_$s"
  done
fi
python -I $PIPE/ca_transfer_decisions.py --day-mat "$(w $R/3_extract/${tag}_extract.mat)" \
    --day-review "$(w $R/4_cell_review/${tag}_cell_review.csv)" $REFS --out "$(w $A/decisions)" --tag $tag \
    || { echo "*** $tag: transfer failed"; exit 1; }
# cells you kept that the whole-day run did not find: your footprint goes back in, joint LS traces
NMISS=$(python -I -c "import json,sys; print(len(json.load(open(sys.argv[1]))['user_keeps_not_found']))" "$(w $A/decisions/${tag}_decisions.json)")
if [ "$NMISS" -gt 0 ]; then
  echo "[rescue] $tag: $NMISS of your kept cells not found by the whole-day EXTRACT -> rescue"
  "$MATLAB" -batch "addpath('$(w $PIPE)'); rescue_missing_cells('$(w $R/3_extract/${tag}_extract.mat)', '$(w $R/4_cell_review/${tag}_cell_review.csv)', '$(w $A/decisions/${tag}_decisions.json)');" < /dev/null 2>&1 | grep -v "Name is nonexistent" | grep -E "rescue|Error|error"
  python -I $PIPE/ca_transfer_decisions.py --day-mat "$(w $R/3_extract/${tag}_extract.mat)" \
      --day-review "$(w $R/4_cell_review/${tag}_cell_review.csv)" $REFS --out "$(w $A/decisions)" --tag $tag \
      || { echo "*** $tag: transfer after rescue failed"; exit 1; }
fi
# near-identical kept footprints (cos >= 0.7) = one neuron: keep the one with your decision / higher SNR
python -I $PIPE/ca_duplicate_check.py --day-root "$(w $R)" --resolve 0.7 || echo "*** $tag: duplicate resolution failed"
# attention mark on every kept cell you have not decided yourself (rule only); keep is not changed
ST=$A/decisions/${tag}_decisions_strict_dryrun.csv
if [ -f $ST ]; then STARG="--strict $(w $ST)"; else STARG=""; fi
python -I $PIPE/ca_mark_attention.py --day-root "$(w $R)" $STARG || echo "*** $tag: attention marks failed"
# SELECTION_ONLY.flag in the day folder: stop here, after the cell selection and its max projection,
# so you can exclude ROIs before any trace / analysis is made (2026-10-21) - unless you confirmed this
# mouse's selection (<tag>_CONFIRMED.flag)
if [ -f $OUT/SELECTION_ONLY.flag ] && [ ! -f $OUT/${tag}_CONFIRMED.flag ]; then
  python -I $PIPE/ca_day_maxproj.py --day-root "$(w $R)" || echo "*** $tag: max projection figure failed"
  echo "#### $tag: selection only (SELECTION_ONLY.flag) - stopped before traces / analyses $(date '+%F %T') ####"
  exit 0
fi
"$MATLAB" -batch "addpath('$(w $PIPE)'); cfg = struct('mouse_id', '$m', 'day', 1, 'sex', '${SEX[$m]}', 'arm', '${ARM[$m]}', 'out_root', '$(w $OUT)', 'scoring_dir', '$SCORE', 'stages', {{'keep', 'merge'}}, 'merge_opts', struct('allow_count_mismatch', true)); run_mini1p_day(cfg); try, cfg.stages = {'classify'}; run_mini1p_day(cfg); catch e, fprintf(2, 'classify (mini2p reference) failed: %s\n', e.message); end" < /dev/null 2>&1 | grep -v "Name is nonexistent"
python -I $PIPE/ca_trace_outputs.py --extract "$(w $R/3_extract/${tag}_extract.mat)" \
    --keep "$(w $R/4_cell_review/${tag}_cell_review.csv)" --out-dir "$(w $A/traces)" || echo "*** $tag: traces failed"
python -I $PIPE/ca_duplicate_check.py --day-root "$(w $R)" --out "$(w $A/decisions)" || echo "*** $tag: duplicate check failed"
# sensitivity version: the background (outside every footprint) regressed out of each cell
python -I $PIPE/ca_background_signal.py --day-root "$(w $R)" && python -I $PIPE/ca_trace_outputs.py --extract "$(w $R/3_extract/${tag}_extract.mat)"     --keep "$(w $R/4_cell_review/${tag}_cell_review.csv)" --out-dir "$(w $A/traces_bgcorr)" --bg-regress "$(w $A/traces)"     || echo "*** $tag: background-corrected traces failed"
# third version without demixing: footprint pixels minus the local neuropil, from the movie (~5 min)
python -I $PIPE/ca_roi_traces.py --day-root "$(w $R)" || echo "*** $tag: ROI traces failed"
python -I $PIPE/ca_session_figures.py --extract "$(w $R/3_extract/${tag}_extract.mat)" \
    --keep "$(w $R/4_cell_review/${tag}_cell_review.csv)" --projections "$(w $R/3_extract/${tag}_projections.mat)" \
    --traces-dir "$(w $A/traces)" --scoring-dir "$SCORE" --out-dir "$(w $A/figures)" || echo "*** $tag: figures failed"
python -I $PIPE/ca_qc_sheet.py --day-root "$(w $R)" || echo "*** $tag: QC sheet failed"
# cells on the max dF/F projection, coloured by who decided (judge morphology there)
python -I $PIPE/ca_day_maxproj.py --day-root "$(w $R)" || echo "*** $tag: max projection figure failed"
# the curation GUI on the FINAL candidates (after rescue) and a launcher for the whole day
"$MATLAB" -batch "addpath('$(w $PIPE)'); prepare_curation_data('$(w $R/3_extract/${tag}_extract.mat)', '$(w $R/4_cell_review/${tag}_cell_review.csv)', '$(w $R/3_extract/${tag}_projections.mat)', '$(w $R/4_cell_review/${tag}_curation_data.mat)');" < /dev/null 2>&1 | grep -E "prepare_curation|Error"
cat > $R/RUN_cell_curation_${tag}_whole_day.m <<EOF2
%% RUN_cell_curation_${tag}_whole_day.m  -  review the WHOLE-DAY candidates of ${tag} (pre + post, one set)
% Open in the MATLAB desktop and press F5. The keep you see = the current decisions (yours carried over,
% rescued cells, the rules for new candidates, duplicates removed); the note column says why.
%   KEYS  k keep   d delete   -> / n next   <- / p previous   u next not reviewed   s save
% Saving writes keep + reviewed into 4_cell_review/${tag}_cell_review.csv; a re-run of ca_post_mouse.sh keeps
% every candidate you reviewed here as you decided.
addpath('$(w $PIPE)');
cell_curation_gui('$(w $R/4_cell_review/${tag}_curation_data.mat)');
EOF2
VID=${CA_VID:-/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/analysis/ca_videos}
mkdir -p $VID
for s in pre post; do
  python -I $PIPE/ca_behaviour_video.py --day-root "$(w $R)" --rid ${tag}_$s --scoring-dir "$SCORE" \
      --out-dir "$(w $VID)" --auto --dur 40 || echo "*** ${tag}_$s: video failed"
done
# space: the binned detection movie (~11 GB) is only needed to re-run EXTRACT; extract_1p remakes it
for b in $R/1_movie/*_detect_binned_*.h5; do
  [ -f "$b" ] && { echo "[disk] $tag: deleting the detection movie $(basename $b) ($(du -h "$b" | cut -f1))"; rm -f "$b"; }
done
touch $OUT/${tag}_post_done.flag
echo "#### post $tag done $(date '+%F %T') ####"
