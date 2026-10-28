%% RUN_ca_day_v2_20261021.m  -  whole-day detection again for 5937 / 5939 / 1191 (2026-10-21)
% After your review of the 1-min previews on the max projection:
%   seeds   k_sd 2 (was 3), min_sep 1.0 radius (was 1.5), your circled positions forced in
%           (analysis\curation_marks_20261021\marks_day_<tag>.csv, day coordinates) and replacing
%           an automatic seed sitting on them
%   review  pieces >= 25 % of the biggest (was 5 %), not_cell above 4 pieces, axis 4, size 7x
%   pass 1 and the projections of the first day run are re-used (same movie, same recipe)
% The keep decision is made afterwards by ca_post_mouse.sh: your preview keeps + your marks, and
% new candidates only with a clear neuron shape (ca_transfer_decisions --strict-new).
% The first day run is kept as 3_extract_v1_20261020 / 4_cell_review_v1_20261020 / 5_analysis_v1_20261020.
% Run: matlab -batch "DO = {'5939'}; run('...\RUN_ca_day_v2_20261021.m')"

here = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_pipeline';
addpath(here);
OUT_ROOT    = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_out\20261019';
SCORING_DIR = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261019_final';
MARKS = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis\curation_marks_20261021';
V1 = '_v1_20261020';
MICE = struct( ...
  'id',  {'5941', '5937', '5939', '1191'}, ...
  'sex', {'F', 'M', 'M', 'F'}, ...
  'arm', {'sbi553', 'vehicle', 'sbi553', 'vehicle'}, ...
  'pre_isxd',  {'D:\testHANSOL_20261019-111628\5941female_baseline_sBIday1_2026-10-19-11-58-32_video.isxd', ...
                'D:\testHANSOL_20261019-124926\5931malebaseline_2026-10-19-13-01-19_video.isxd', ...
                'D:\testHANSOL_20261019-134128\5939malebase_2026-10-19-13-48-58_video.isxd', ...
                'D:\testHANSOL_20261019-143901\1191femalebase_2026-10-19-14-44-13_video.isxd'}, ...
  'post_isxd', {'D:\testHANSOL_20261019-121826\sbi5941_2026-10-19-12-27-44_video.isxd', ...
                'D:\testHANSOL_20261019-132106\5937male_control_2026-10-19-13-21-29_video.isxd', ...
                'D:\testHANSOL_20261019-141205\5939SBI_2026-10-19-14-17-35_video.isxd', ...
                'D:\testHANSOL_20261019-150319\1191control_2026-10-19-15-04-45_video.isxd'});
RECIPE = struct('avg_cell_radius', 10, 'T_min_snr', 5, 'cellfind_min_snr', 1, 'spatial_highpass_cutoff', 8);
if ~exist('DO', 'var'), DO = {'5939', '1191', '5937'}; end

for i = 1:numel(DO)
    m = MICE(strcmp({MICE.id}, DO{i}));
    tag = sprintf('%s_d1', m.id);
    R = fullfile(OUT_ROOT, tag);
    t0 = tic;
    fprintf('\n#### day v2 %s  %s ####\n', tag, char(datetime('now')));
    try
        % keep the first run
        for d = {'3_extract', '4_cell_review', '5_analysis'}
            src = fullfile(R, d{1}); dst = fullfile(R, [d{1} V1]);
            if isfolder(src) && ~isfolder(dst), movefile(src, dst); end
        end
        x1 = fullfile(R, ['3_extract' V1]);
        xd = fullfile(R, '3_extract');
        if ~isfolder(fullfile(xd, 'projections')), mkdir(fullfile(xd, 'projections')); end
        copyfile(fullfile(x1, 'projections', '*.mat'), fullfile(xd, 'projections'));
        seed = struct('k_sd', 2, 'min_sep', 1.0, 'extra_replace', true);
        mf = fullfile(MARKS, ['marks_day_' tag '.csv']);
        if isfile(mf)
            K = readtable(mf);
            if height(K), seed.extra_rc = [K.fov_row, K.fov_col]; end
            fprintf('%s: %d of your marks as seeds\n', tag, height(K));
        end
        cfg = struct('mouse_id', m.id, 'day', 1, 'sex', m.sex, 'arm', m.arm, 'pre_isxd', m.pre_isxd, ...
                     'post_isxd', m.post_isxd, 'out_root', OUT_ROOT, 'scoring_dir', SCORING_DIR, 'extract', RECIPE);
        cfg.mc = struct('delete_raw', false, 'mc_options', struct('crop_rect', 'auto', 'crop_margin_px', 40));
        cfg.stages = {'extract', 'review'};
        cfg.accept_mc_check = true;
        cfg.extract_common = struct('detect_prebin', true, 'use_gpu', false);
        cfg.extract_seed = seed;
        cfg.extract_pass1_mat = fullfile(x1, 'pass1', [tag '_extract.mat']);
        cfg.review_qc = struct('piece_min_frac', 0.25, 'pieces_max', 4, 'axis_max', 4, 'area_max', 7);
        run_mini1p_day(cfg);
        fclose(fopen(fullfile(OUT_ROOT, [tag '_v2_detect_done.flag']), 'w'));
        fprintf('#### day v2 %s detection done in %.1f h ####\n', tag, toc(t0) / 3600);
    catch err
        fprintf(2, '\n*** day v2 %s STOPPED after %.1f h: %s\n', tag, toc(t0) / 3600, err.message);
        for k = 1:numel(err.stack), fprintf(2, '   %s line %d\n', err.stack(k).name, err.stack(k).line); end
        fclose(fopen(fullfile(OUT_ROOT, [tag '_v2_detect_failed.flag']), 'w'));
    end
end
