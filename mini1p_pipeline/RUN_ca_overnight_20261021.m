%% RUN_ca_overnight_20261021.m  -  Day 2 (2026-10-21) whole-session Ca pipeline, all 4 mice, overnight
% The Day-1 overnight recipe (RUN_ca_overnight_20261019.m) with the Day-2 recordings:
%   copy    lossless .isxd -> h5 copies, verified pixel by pixel, one mouse ahead of 'mc'
%   mc      inscopix_day (re-uses the copies) + NoRMCorre with the lens crop, motion QC, QC media of both
%           sessions while the raw copy exists, then the raw copy is deleted
%   detect  ONE mouse (M): EXTRACT seeded with the SAME settings that mouse had on Day 1
%             5941              default seeds (k_sd 3, min_sep 1.5) + default review rules
%             5937 / 5939 / 1191 the v2 settings you confirmed on 10-21 (k_sd 2, min_sep 1.0, review pieces 25 %,
%                               <= 4 pieces, axis 4, area 7x)
%           plus the Day-1 confirmed neurons registered onto the Day-2 field of view as forced seeds
%           (analysis\day2_seeds\marks_day2_<tag>.csv, written by ca_day2_seeds.py before this runs)
% Pre and post of a mouse share one template, one crop and one EXTRACT run (cell k = the same cell in both).
% Crossover: 5941 / 5939 vehicle, 5937 / 1191 SBI-553 on Day 2.
% 1191 post: the 13-43-55 file (418 MB, a few seconds) was a false start; the session is 13-44-58.
% Start:  matlab -batch "WORKER='copy'; run('...\RUN_ca_overnight_20261021.m')"   (and 'mc')
%         matlab -batch "WORKER='detect'; M='5937'; run('...\RUN_ca_overnight_20261021.m')"

here = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_pipeline';
addpath(here);
OUT_ROOT    = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_out\20261021';
SCORING_DIR = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261021_final';
SEEDS       = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis\day2_seeds';
MICE = struct( ...
  'id',  {'5941', '5937', '5939', '1191'}, ...
  'sex', {'F', 'M', 'M', 'F'}, ...
  'arm', {'vehicle', 'sbi553', 'vehicle', 'sbi553'}, ...
  'pre_isxd',  {'E:\testHANSOL_20261021-103656\day2_5941base_2026-10-21-10-40-47_video.isxd', ...
                'D:\testHANSOL_20261021-112133\5937malebase_2026-10-21-11-33-10_video.isxd', ...
                'D:\testHANSOL_20261021-121237\5939malebase_2026-10-21-12-17-49_video.isxd', ...
                'D:\testHANSOL_20261021-131558\1191femalebase_2026-10-21-13-16-33_video.isxd'}, ...
  'post_isxd', {'D:\testHANSOL_20261021-110055\5941control_2026-10-21-11-01-55_video.isxd', ...
                'D:\testHANSOL_20261021-115306\5937mSBI_2026-10-21-11-53-42_video.isxd', ...
                'D:\testHANSOL_20261021-123706\5939malecontol_2026-10-21-12-37-46_video.isxd', ...
                'D:\testHANSOL_20261021-134339\1191SBIreal_2026-10-21-13-44-58_video.isxd'}, ...
  'v2', {false, true, true, true});
RECIPE = struct('avg_cell_radius', 10, 'T_min_snr', 5, 'cellfind_min_snr', 1, 'spatial_highpass_cutoff', 8);
if ~exist('WORKER', 'var'), error('set WORKER = ''copy'' | ''mc'' | ''detect'''); end
flag = @(tag, what) fullfile(OUT_ROOT, sprintf('%s_%s.flag', tag, what));
if ~isfolder(OUT_ROOT), mkdir(OUT_ROOT); end
if strcmp(WORKER, 'detect'), order = find(strcmp({MICE.id}, M)); else, order = 1:numel(MICE); end

for i = order
    m = MICE(i);
    tag = sprintf('%s_d2', m.id);
    mcdir = fullfile(OUT_ROOT, tag, '1_movie');
    t0 = tic;
    try
        switch WORKER
            case 'copy'
                if isfile(flag(tag, 'copied')), continue; end
                fprintf('\n#### copy %s  %s ####\n', tag, char(datetime('now')));
                as = {'pre', 'post'}; fs = {m.pre_isxd, m.post_isxd};
                for r = 1:2
                    rid = sprintf('%s_%s', tag, as{r});
                    if isfile(fullfile(mcdir, [rid '_mc.h5'])), continue; end      % already corrected
                    li = isxd_to_h5(struct('isxd', fs{r}, 'recording_id', rid, 'assay', as{r}, ...
                                           'name', rid, 'out_dir', mcdir, 'verify', 'full'));
                    fprintf('[copy] %s: %d frames, verification passed %d, max |diff| %g\n', rid, ...
                            li.n_valid_total, li.verification.passed, li.verification.max_abs_diff);
                end
                fclose(fopen(flag(tag, 'copied'), 'w'));
            case 'mc'
                if isfile(flag(tag, 'mc_ready')), continue; end
                wait_flag(flag(tag, 'copied'));
                fprintf('\n#### mc %s  %s ####\n', tag, char(datetime('now')));
                cfg = day_cfg(m, OUT_ROOT, SCORING_DIR, RECIPE);
                cfg.stages = {'mc'};
                S = run_mini1p_day(cfg);
                js = jsondecode(fileread(S.files.day_json));
                recs = js.recordings; if ~iscell(recs), recs = num2cell(recs); end
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
                fclose(fopen(flag(tag, 'mc_ready'), 'w'));
            case 'detect'
                if isfile(flag(tag, 'detect_done')), continue; end
                % a second detection worker may already run this mouse (parallel detection once MC is done)
                if isfile(flag(tag, 'detect_running')) && ~(exist('CLAIMED', 'var') && CLAIMED)   % CLAIMED: the caller made the flag
                    fprintf('%s: detection already running in another MATLAB - skipped here\n', tag); continue;
                end
                fclose(fopen(flag(tag, 'detect_running'), 'w'));
                fprintf('\n#### detect %s  %s ####\n', tag, char(datetime('now')));
                cfg = day_cfg(m, OUT_ROOT, SCORING_DIR, RECIPE);
                cfg.stages = {'extract', 'review'};
                cfg.accept_mc_check = true;       % the verdicts are printed; Claude checks the QC images
                cfg.extract_common = struct('detect_prebin', true, 'use_gpu', false);
                if m.v2
                    seed = struct('k_sd', 2, 'min_sep', 1.0, 'extra_replace', true);
                    cfg.review_qc = struct('piece_min_frac', 0.25, 'pieces_max', 4, 'axis_max', 4, 'area_max', 7);
                else
                    seed = struct('extra_replace', true);
                end
                mf = fullfile(SEEDS, ['marks_day2_' tag '.csv']);
                if isfile(mf)
                    K = readtable(mf);
                    if height(K), seed.extra_rc = [K.fov_row, K.fov_col]; end
                    fprintf('%s: %d Day-1 neurons as forced seeds (%s)\n', tag, height(K), mf);
                else
                    fprintf(2, '%s: no Day-1 seed file %s - automatic seeds only\n', tag, mf);
                end
                cfg.extract_seed = seed;
                run_mini1p_day(cfg);
                fclose(fopen(flag(tag, 'detect_done'), 'w'));
                if isfile(flag(tag, 'detect_running')), delete(flag(tag, 'detect_running')); end
        end
        fprintf('#### %s %s done in %.1f h ####\n', WORKER, tag, toc(t0) / 3600);
    catch err
        fprintf(2, '\n*** %s %s STOPPED after %.1f h: %s\n', WORKER, tag, toc(t0) / 3600, err.message);
        for k = 1:numel(err.stack), fprintf(2, '   %s line %d\n', err.stack(k).name, err.stack(k).line); end
        fclose(fopen(flag(tag, [WORKER '_failed']), 'w'));
        if strcmp(WORKER, 'detect') && isfile(flag(tag, 'detect_running')), delete(flag(tag, 'detect_running')); end
    end
end
fprintf('\nOVERNIGHT WORKER %s FINISHED %s\n', WORKER, char(datetime('now')));


function cfg = day_cfg(m, out_root, scoring_dir, recipe)
cfg = struct('mouse_id', m.id, 'day', 2, 'sex', m.sex, 'arm', m.arm, 'pre_isxd', m.pre_isxd, ...
             'post_isxd', m.post_isxd, 'out_root', out_root, 'scoring_dir', scoring_dir, 'extract', recipe);
cfg.mc = struct('delete_raw', false, 'mc_options', struct('crop_rect', 'auto', 'crop_margin_px', 40));
end


function wait_flag(f)
% wait for the flag of the previous worker; give up if it reported a failure
[d, nm] = fileparts(f);
stem = regexprep(nm, '_(copied|mc_ready)$', '');
while ~isfile(f)
    if ~isempty(dir(fullfile(d, [stem '_*_failed.flag'])))
        error('previous step failed for %s', stem);
    end
    pause(30);
end
end
