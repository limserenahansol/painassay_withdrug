%% RUN_ca_redo_day2_20261022.m  -  Day-2 detection of ONE mouse again with corrected Day-1 seeds (2026-10-22 night)
% M = mouse id, SUF = suffix the first run was moved to (its pass 1 = EXTRACT alone, independent of the seeds, is
% re-used). The seeds (analysis\day2_seeds\marks_day2_<tag>.csv) were re-registered by ca_day2_seeds.py
% (sensor-pattern-proof band-pass; activity projections preferred when they disagree with the mean image).
% Settings = that mouse's Day-1 settings (5941 default seeds; 5937 / 5939 / 1191 v2 seeds + review rules).
%   matlab -batch "M='5939'; SUF='_v1_meanreg'; run('...\RUN_ca_redo_day2_20261022.m')"
here = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_pipeline';
addpath(here);
OUT_ROOT = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_out\20261021';
D = struct('id', {'5941', '5937', '5939', '1191'}, 'sex', {'F', 'M', 'M', 'F'}, 'arm', {'vehicle', 'sbi553', 'vehicle', 'sbi553'}, ...
  'pre_isxd',  {'E:\testHANSOL_20261021-103656\day2_5941base_2026-10-21-10-40-47_video.isxd', ...
                'D:\testHANSOL_20261021-112133\5937malebase_2026-10-21-11-33-10_video.isxd', ...
                'D:\testHANSOL_20261021-121237\5939malebase_2026-10-21-12-17-49_video.isxd', ...
                'D:\testHANSOL_20261021-131558\1191femalebase_2026-10-21-13-16-33_video.isxd'}, ...
  'post_isxd', {'D:\testHANSOL_20261021-110055\5941control_2026-10-21-11-01-55_video.isxd', ...
                'D:\testHANSOL_20261021-115306\5937mSBI_2026-10-21-11-53-42_video.isxd', ...
                'D:\testHANSOL_20261021-123706\5939malecontol_2026-10-21-12-37-46_video.isxd', ...
                'D:\testHANSOL_20261021-134339\1191SBIreal_2026-10-21-13-44-58_video.isxd'}, ...
  'v2', {false, true, true, true});
m = D(strcmp({D.id}, M));
tag = sprintf('%s_d2', m.id);
R = fullfile(OUT_ROOT, tag);
t0 = tic;
try
    cfg = struct('mouse_id', m.id, 'day', 2, 'sex', m.sex, 'arm', m.arm, 'pre_isxd', m.pre_isxd, 'post_isxd', m.post_isxd, ...
                 'out_root', OUT_ROOT, 'scoring_dir', 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261021_final', ...
                 'extract', struct('avg_cell_radius', 10, 'T_min_snr', 5, 'cellfind_min_snr', 1, 'spatial_highpass_cutoff', 8));
    cfg.mc = struct('delete_raw', false, 'mc_options', struct('crop_rect', 'auto', 'crop_margin_px', 40));
    cfg.stages = {'extract', 'review'};
    cfg.accept_mc_check = true;
    cfg.extract_common = struct('detect_prebin', true, 'use_gpu', false);
    if m.v2
        seed = struct('k_sd', 2, 'min_sep', 1.0, 'extra_replace', true);
        cfg.review_qc = struct('piece_min_frac', 0.25, 'pieces_max', 4, 'axis_max', 4, 'area_max', 7);
    else
        seed = struct('extra_replace', true);
    end
    K = readtable(fullfile('C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis\day2_seeds', ['marks_day2_' tag '.csv']));
    seed.extra_rc = [K.fov_row, K.fov_col];
    fprintf('%s: %d Day-1 neurons as forced seeds (corrected registration)\n', tag, height(K));
    cfg.extract_seed = seed;
    cfg.extract_pass1_mat = fullfile(R, ['3_extract' SUF], 'pass1', [tag '_extract.mat']);
    run_mini1p_day(cfg);
    fclose(fopen(fullfile(OUT_ROOT, [tag '_detect_done.flag']), 'w'));
    fprintf('#### redo detect %s done in %.1f h ####\n', tag, toc(t0) / 3600);
catch err
    fprintf(2, '\n*** redo detect %s STOPPED after %.1f h: %s\n', tag, toc(t0) / 3600, err.message);
    for k = 1:numel(err.stack), fprintf(2, '   %s line %d\n', err.stack(k).name, err.stack(k).line); end
    fclose(fopen(fullfile(OUT_ROOT, [tag '_detect_failed.flag']), 'w'));
end
if isfile(fullfile(OUT_ROOT, [tag '_detect_running.flag'])), delete(fullfile(OUT_ROOT, [tag '_detect_running.flag'])); end
