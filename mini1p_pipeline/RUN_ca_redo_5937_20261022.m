%% RUN_ca_redo_5937_20261022.m  -  5937 Day-2 detection again with CORRECT Day-1 seeds (2026-10-22 night)
% The first run's seeds were registered on the camera's fixed pixel pattern (shift = lens-crop difference (+2, +23);
% tissue shift (-3, -16)), so the 24 Day-1 neurons were seeded ~40 px off. ca_day2_seeds.py now median-filters and
% band-passes the mean images (5941 / 5939 unchanged). Pass 1 (EXTRACT alone, does not use the seeds) is re-used.
% The first run is kept as 3_extract_v1_sensorreg / 4_cell_review_v1_sensorreg / 5_analysis_v1_sensorreg.
here = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_pipeline';
addpath(here);
OUT_ROOT = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_out\20261021';
tag = '5937_d2';
R = fullfile(OUT_ROOT, tag);
t0 = tic;
try
    x1 = fullfile(R, '3_extract_v1_sensorreg');
    cfg = struct('mouse_id', '5937', 'day', 2, 'sex', 'M', 'arm', 'sbi553', ...
                 'pre_isxd', 'D:\testHANSOL_20261021-112133\5937malebase_2026-10-21-11-33-10_video.isxd', ...
                 'post_isxd', 'D:\testHANSOL_20261021-115306\5937mSBI_2026-10-21-11-53-42_video.isxd', ...
                 'out_root', OUT_ROOT, 'scoring_dir', 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261021_final', ...
                 'extract', struct('avg_cell_radius', 10, 'T_min_snr', 5, 'cellfind_min_snr', 1, 'spatial_highpass_cutoff', 8));
    cfg.mc = struct('delete_raw', false, 'mc_options', struct('crop_rect', 'auto', 'crop_margin_px', 40));
    cfg.stages = {'extract', 'review'};
    cfg.accept_mc_check = true;
    cfg.extract_common = struct('detect_prebin', true, 'use_gpu', false);
    seed = struct('k_sd', 2, 'min_sep', 1.0, 'extra_replace', true);
    K = readtable('C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis\day2_seeds\marks_day2_5937_d2.csv');
    seed.extra_rc = [K.fov_row, K.fov_col];
    fprintf('%s: %d Day-1 neurons as forced seeds (corrected registration)\n', tag, height(K));
    cfg.extract_seed = seed;
    cfg.review_qc = struct('piece_min_frac', 0.25, 'pieces_max', 4, 'axis_max', 4, 'area_max', 7);
    cfg.extract_pass1_mat = fullfile(x1, 'pass1', [tag '_extract.mat']);
    run_mini1p_day(cfg);
    fclose(fopen(fullfile(OUT_ROOT, [tag '_detect_done.flag']), 'w'));
    fprintf('#### redo detect %s done in %.1f h ####\n', tag, toc(t0) / 3600);
catch err
    fprintf(2, '\n*** redo detect %s STOPPED after %.1f h: %s\n', tag, toc(t0) / 3600, err.message);
    for k = 1:numel(err.stack), fprintf(2, '   %s line %d\n', err.stack(k).name, err.stack(k).line); end
    fclose(fopen(fullfile(OUT_ROOT, [tag '_detect_failed.flag']), 'w'));
end
if isfile(fullfile(OUT_ROOT, [tag '_detect_running.flag'])), delete(fullfile(OUT_ROOT, [tag '_detect_running.flag'])); end
