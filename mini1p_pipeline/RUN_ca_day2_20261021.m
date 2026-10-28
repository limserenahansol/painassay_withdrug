%% RUN_ca_day2_20261021.m  -  Day 2 (2026-10-21) Ca pipeline, the same recipe as Day 1
% Recordings arrive one by one; this runs whatever .isxd of a mouse is listed (pre now, post later).
%   STAGE = 'mc'      lossless .isxd -> h5 (verified), NoRMCorre with the lens crop, motion QC, QC stage media
%   STAGE = 'detect'  EXTRACT seeded (pass 1 + Cn x PNR seeds + the Day-1 confirmed neurons registered onto
%                     the Day-2 field of view: analysis\day2_seeds\marks_day2_<tag>.csv), then the review metrics
% Same settings as Day 1 (radius 10, T_min_snr 5, cellfind 1, high-pass 8, prebinned 1-partition detection, CPU).
% Run: matlab -batch "M='5941'; STAGE='mc'; run('...\RUN_ca_day2_20261021.m')"

here = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_pipeline';
addpath(here);
OUT_ROOT    = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_out\20261021';
SCORING_DIR = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261021_day2';
SEEDS       = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis\day2_seeds';
% Day-2 recordings (add the post .isxd when it is copied)
D2 = struct( ...
  'id',  {'5941'}, 'sex', {'F'}, 'arm', {'day2: vehicle (crossover)'}, ...
  'pre_isxd',  {'E:\testHANSOL_20261021-103656\day2_5941base_2026-10-21-10-40-47_video.isxd'}, ...
  'post_isxd', {''});
RECIPE = struct('avg_cell_radius', 10, 'T_min_snr', 5, 'cellfind_min_snr', 1, 'spatial_highpass_cutoff', 8);
if ~exist('M', 'var'), M = '5941'; end
if ~exist('STAGE', 'var'), STAGE = 'mc'; end
if ~isfolder(OUT_ROOT), mkdir(OUT_ROOT); end
m = D2(strcmp({D2.id}, M));
tag = sprintf('%s_d2', m.id);
t0 = tic;
fprintf('\n#### day 2 %s %s  %s ####\n', STAGE, tag, char(datetime('now')));
try
    cfg = struct('mouse_id', m.id, 'day', 2, 'sex', m.sex, 'arm', m.arm, 'pre_isxd', m.pre_isxd, ...
                 'post_isxd', m.post_isxd, 'out_root', OUT_ROOT, 'scoring_dir', SCORING_DIR, 'extract', RECIPE);
    cfg.mc = struct('delete_raw', false, 'mc_options', struct('crop_rect', 'auto', 'crop_margin_px', 40));
    switch STAGE
        case 'mc'
            cfg.stages = {'mc'};
            S = run_mini1p_day(cfg);
            js = jsondecode(fileread(S.files.day_json));
            recs = js.recordings; if ~iscell(recs), recs = num2cell(recs); end
            mcdir = fullfile(S.dirs.root, '1_movie');
            for r = 1:numel(recs)
                rid = recs{r}.recording_id;
                mc_h5 = fullfile(mcdir, [rid '_mc.h5']); raw_h5 = fullfile(mcdir, [rid '_raw.h5']);
                qd = fullfile(S.dirs.root, '2_qc');
                if ~isfile(fullfile(qd, [rid '_qc_stages.png'])) && isfile(raw_h5)
                    try
                        qc_stage_media(struct('raw_h5', raw_h5, 'mc_h5', mc_h5, ...
                            'frames_csv', fullfile(mcdir, [rid '_raw_frames.csv']), 'out_dir', qd, ...
                            'tag', rid, 'window_s', [195 225]));
                    catch e
                        fprintf(2, '[qc] %s: QC media failed: %s\n', rid, e.message);
                    end
                end
                if isfile(raw_h5) && recs{r}.lossless_verification.passed && isfile(mc_h5)
                    b = dir(raw_h5).bytes; delete(raw_h5);
                    fprintf('[disk] deleted the verified raw copy %s (%.1f GB)\n', raw_h5, b / 1e9);
                end
            end
            fclose(fopen(fullfile(OUT_ROOT, [tag '_mc_ready.flag']), 'w'));
        case 'detect'
            cfg.stages = {'extract', 'review'};
            cfg.accept_mc_check = true;
            cfg.extract_common = struct('detect_prebin', true, 'use_gpu', false);
            % the 5941 Day-1 settings (you, 10-21): default seeding (k_sd, min_sep) and default review metrics,
            % plus the Day-1 neurons as forced seeds so the same neurons can be followed
            seed = struct('extra_replace', true);
            mf = fullfile(SEEDS, ['marks_day2_' tag '.csv']);
            if isfile(mf)
                K = readtable(mf);
                if height(K), seed.extra_rc = [K.fov_row, K.fov_col]; end
                fprintf('%s: %d Day-1 neurons as seeds\n', tag, height(K));
            end
            cfg.extract_seed = seed;
            run_mini1p_day(cfg);
            fclose(fopen(fullfile(OUT_ROOT, [tag '_detect_done.flag']), 'w'));
    end
    fprintf('#### day 2 %s %s done in %.1f h ####\n', STAGE, tag, toc(t0) / 3600);
catch err
    fprintf(2, '\n*** day 2 %s %s STOPPED after %.1f h: %s\n', STAGE, tag, toc(t0) / 3600, err.message);
    for k = 1:numel(err.stack), fprintf(2, '   %s line %d\n', err.stack(k).name, err.stack(k).line); end
    fclose(fopen(fullfile(OUT_ROOT, [tag '_' STAGE '_failed.flag']), 'w'));
end
