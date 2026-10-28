%% RUN_ca_overnight_20261019.m  -  whole-session Ca pipeline, all 4 mice of 2026-10-19
% Three workers run this same file with WORKER = 'copy' | 'mc' | 'detect' and hand the mice
% to each other through flag files, so the heavy steps never overlap (no crash):
%   copy    lossless .isxd -> h5 copies, verified pixel by pixel, one mouse ahead of 'mc'
%           (inscopix_day re-uses a verified copy: same file, size and recording id)
%   mc      inscopix_day (re-uses the copies) + NoRMCorre with the lens crop, motion QC,
%           QC media of both sessions while the raw copy exists, then the raw copy is deleted
%   detect  EXTRACT recipe of the 5941 test (extract_1p_seeded: radius 10, T_min_snr 5,
%           cellfind 1, high-pass 8 + Cn x PNR seeds), detection on a movie binned in time 4 x
%           and space 2 x so EXTRACT runs in ONE partition (no seams), on the CPU; then the
%           review rules fitted to your 5941 decisions and the curation GUI data
% Pre and post of a mouse share one template, one crop and one EXTRACT run: cell k is the
% same cell in both sessions.
% Start (three processes):
%   matlab -batch "WORKER='copy'; run('...\RUN_ca_overnight_20261019.m')"   (and 'mc', 'detect')

here = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_pipeline';
addpath(here);
OUT_ROOT    = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_out\20261019';
SCORING_DIR = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261019_final';
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
if ~exist('WORKER', 'var'), error('set WORKER = ''copy'' | ''mc'' | ''detect'''); end
flag = @(tag, what) fullfile(OUT_ROOT, sprintf('%s_%s.flag', tag, what));
waitfor_flag = @(f) wait_flag(f);

for i = 1:numel(MICE)
    m = MICE(i);
    tag = sprintf('%s_d1', m.id);
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
                waitfor_flag(flag(tag, 'copied'));
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
                waitfor_flag(flag(tag, 'mc_ready'));
                fprintf('\n#### detect %s  %s ####\n', tag, char(datetime('now')));
                cfg = day_cfg(m, OUT_ROOT, SCORING_DIR, RECIPE);
                cfg.stages = {'extract', 'review'};
                cfg.accept_mc_check = true;       % the verdicts are printed; Claude checks the QC images
                cfg.extract_common = struct('detect_prebin', true, 'use_gpu', false);
                run_mini1p_day(cfg);
                fclose(fopen(flag(tag, 'detect_done'), 'w'));
        end
        fprintf('#### %s %s done in %.1f h ####\n', WORKER, tag, toc(t0) / 3600);
    catch err
        fprintf(2, '\n*** %s %s STOPPED after %.1f h: %s\n', WORKER, tag, toc(t0) / 3600, err.message);
        fclose(fopen(flag(tag, [WORKER '_failed']), 'w'));
    end
end
fprintf('\nOVERNIGHT WORKER %s FINISHED %s\n', WORKER, char(datetime('now')));


function cfg = day_cfg(m, out_root, scoring_dir, recipe)
cfg = struct('mouse_id', m.id, 'day', 1, 'sex', m.sex, 'arm', m.arm, 'pre_isxd', m.pre_isxd, ...
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
