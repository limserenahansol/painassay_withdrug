%% RUN_mini1p_20261019.m  -  the 2026-10-19 cohort (4 mice x baseline + post-injection)
% One mouse-day at a time; every stage is skipped when its files exist, so run this
% file again after each human step. See PIPELINE_mini1p.md for the whole picture.
%
%   STEP A (unattended, ~1-2 h per mouse)   STAGES = {'mc', 'sweep'}
%       lossless copy -> verified -> motion correction (QC printed) -> raw copy deleted
%       -> EXTRACT parameter sweep on an active 5-min piece
%   >>> YOU: open <mouse>_d1\2_sweep\<tag>_sweep_sheet.png, choose a setting -> EXTRACT_SET
%   STEP B (unattended, ~1-2 h)             STAGES = {'extract', 'review'}
%   >>> YOU: open 4_cell_review\<tag>_cells_map.png + pages, edit keep in <tag>_cell_review.csv
%   STEP C (behaviour, any time)            scoring_gui\RUN_scoring_mini1p_20261019.m
%   STEP D (minutes)                        STAGES = {'keep', 'merge', 'classify'}
%   STEP E                                  summarize_mini1p_cohort(OUT_ROOT, MICE)

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(here);

OUT_ROOT    = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_out\20261019';
SCORING_DIR = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261019';

% Raw IDAS movies (.isxd; the .gpio / .imu next to them are not needed for this path).
% Sex and arm from the 2026-10-19 sheet (sbi mini1p 1019).
MICE = struct( ...
  'id',  {'5941',    '5937',    '5939',    '1191'}, ...
  'sex', {'F',       'M',       'M',       'F'}, ...
  'arm', {'sbi553',  'vehicle', 'sbi553',  'vehicle'}, ...
  'pre_isxd',  {'D:\testHANSOL_20261019-111628\5941female_baseline_sBIday1_2026-10-19-11-58-32_video.isxd', ...
                'D:\testHANSOL_20261019-124926\5931malebaseline_2026-10-19-13-01-19_video.isxd', ...   % named 5931: it is 5937 (13:01 = behaviour 5937malebase 13-01-17)
                'D:\testHANSOL_20261019-134128\5939malebase_2026-10-19-13-48-58_video.isxd', ...
                'D:\testHANSOL_20261019-143901\1191femalebase_2026-10-19-14-44-13_video.isxd'}, ...
  'post_isxd', {'D:\testHANSOL_20261019-121826\sbi5941_2026-10-19-12-27-44_video.isxd', ...
                'D:\testHANSOL_20261019-132106\5937male_control_2026-10-19-13-21-29_video.isxd', ...
                'D:\testHANSOL_20261019-141205\5939SBI_2026-10-19-14-17-35_video.isxd', ...
                'D:\testHANSOL_20261019-150319\1191control_2026-10-19-15-04-45_video.isxd'});

DO_MICE = {'5941', '5937', '5939', '1191'};   % which mice to run now
% The cell recipe is settled (5941 test, user-approved 2026-10-20): no sweep; extract =
% extract_1p_seeded with the pass-1 setting below; review = the fitted keep rules + GUI data.
STAGES  = {'mc', 'extract', 'review'};

% EXTRACT setting chosen from the sweep sheet, per mouse ('suggested' = the sweep's pick).
% Example: EXTRACT_SET.m5941 = struct('avg_cell_radius', 13, 'T_min_snr', 7, ...
%                                     'cellfind_min_snr', 1, 'spatial_highpass_cutoff', 5);
RECIPE = struct('avg_cell_radius', 10, 'T_min_snr', 5, 'cellfind_min_snr', 1, 'spatial_highpass_cutoff', 8);
EXTRACT_SET = struct('m5941', RECIPE, 'm5937', RECIPE, 'm5939', RECIPE, 'm1191', RECIPE);

% If the first sweep finds few cells, delete 2_sweep\ and sweep again wider, e.g. the
% EXTRACT low-SNR advice (Tutorial 4: T_min_snr down to 3, cellfind_min_snr 0):
%   SWEEP_GRID = struct('avg_cell_radius', [10 13 16], 'T_min_snr', [3 4 5], ...
%                       'cellfind_min_snr', [0 1], 'spatial_highpass_cutoff', [3 5]);
SWEEP_GRID = [];                    % [] = default grid (radius 10/13 x T_min_snr 5/7 x high-pass 5/8/12)

% The extract stage refuses to run while the motion QC says CHECK (pre and post must be the
% same field of view to share one EXTRACT run). After LOOKING at 1_movie\*_mc_qc.png, list
% a mouse here to go ahead anyway.
ACCEPT_MC_CHECK = {};

for i = 1:numel(MICE)
    m = MICE(i);
    if ~ismember(m.id, DO_MICE), continue; end
    if any(strcmp({m.pre_isxd, m.post_isxd}, 'SET_ME')) && any(strcmp(STAGES, 'mc'))
        error('Set pre_isxd / post_isxd of mouse %s in this file first.', m.id);
    end
    cfg = struct('mouse_id', m.id, 'day', 1, 'sex', m.sex, 'arm', m.arm, ...
                 'pre_isxd', m.pre_isxd, 'post_isxd', m.post_isxd, ...
                 'out_root', OUT_ROOT, 'scoring_dir', SCORING_DIR, 'stages', {STAGES});
    key = ['m' m.id];
    if isfield(EXTRACT_SET, key), cfg.extract = EXTRACT_SET.(key); else, cfg.extract = 'suggested'; end
    cfg.accept_mc_check = ismember(m.id, ACCEPT_MC_CHECK);
    if ~isempty(SWEEP_GRID), cfg.sweep = struct('grid', SWEEP_GRID); end
    try
        run_mini1p_day(cfg);
    catch err                          % one mouse failing (e.g. the motion QC gate) must not stop the others
        fprintf(2, '\n*** mouse %s stopped: %s\n', m.id, err.message);
    end
end
