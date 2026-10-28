%% RUN_ca_batch_20261019.m  -  unattended Ca pipeline for all 4 mice of 2026-10-19 (8 .isxd)
% Per mouse-day (baseline 'pre' + injection 'post' share ONE template, ONE lens crop and ONE
% EXTRACT run, so neuron k is the same cell in both sessions):
%   1. lossless .isxd -> h5 (pixel-exact, verified) and NoRMCorre motion correction with the
%      lens crop (motion_correct_h5 crop_rect 'auto': the lit lens area + 40 px; saves disk
%      and EXTRACT time). The raw copy is kept until step 2 has made its QC media.
%   2. QC media per recording (qc_stage_media): raw vs motion-corrected video, filtered video,
%      stage images, seed images; then the verified raw copy is deleted.
%   3. EXTRACT with the settled recipe (extract_1p_seeded, pass 1 radius 10, T_min_snr 5,
%      cellfind 1, high-pass 8 + Cn x PNR seeds) and the cell review (rules fitted to your
%      5941 decisions; keep pre-filled) + data for cell_curation_gui.
% A mouse that stops (e.g. the motion QC gate) is logged and the next mouse runs.
% Run with:  matlab -batch "run('...\RUN_ca_batch_20261019.m')"

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
if ~exist('DO_MICE', 'var'), DO_MICE = {MICE.id}; end
if ~exist('ACCEPT_MC_CHECK', 'var'), ACCEPT_MC_CHECK = {}; end

for i = 1:numel(MICE)
    m = MICE(i);
    if ~ismember(m.id, DO_MICE), continue; end
    t0 = tic;
    fprintf('\n################ mouse %s (%s)  %s ################\n', m.id, m.arm, char(datetime('now')));
    try
        cfg = struct('mouse_id', m.id, 'day', 1, 'sex', m.sex, 'arm', m.arm, ...
                     'pre_isxd', m.pre_isxd, 'post_isxd', m.post_isxd, 'out_root', OUT_ROOT, ...
                     'scoring_dir', SCORING_DIR, 'extract', RECIPE, ...
                     'accept_mc_check', ismember(m.id, ACCEPT_MC_CHECK));
        cfg.mc = struct('delete_raw', false, 'mc_options', struct('crop_rect', 'auto', 'crop_margin_px', 40));
        % 1. copy + motion correction
        cfg.stages = {'mc'};
        S = run_mini1p_day(cfg);
        % 2. QC media while the raw copy still exists, then delete the verified raw copy
        js = jsondecode(fileread(S.files.day_json));
        recs = js.recordings; if ~iscell(recs), recs = num2cell(recs); end
        qd = fullfile(S.dirs.root, '2_qc');
        for r = 1:numel(recs)
            R = recs{r};
            rid = R.recording_id;
            mc_h5 = fullfile(S.dirs.mc, [rid '_mc.h5']);
            raw_h5 = fullfile(S.dirs.mc, [rid '_raw.h5']);
            if ~isfile(fullfile(qd, [rid '_qc_stages.png']))
                try
                    qc_stage_media(struct('raw_h5', raw_h5, 'mc_h5', mc_h5, ...
                        'frames_csv', fullfile(S.dirs.mc, [rid '_raw_frames.csv']), 'out_dir', qd, ...
                        'tag', rid, 'window_s', [195 225]));
                catch e
                    fprintf(2, '[qc] %s: QC media failed: %s\n', rid, e.message);
                end
            end
            if isfile(raw_h5) && R.lossless_verification.passed && isfile(mc_h5)
                b = dir(raw_h5).bytes;
                delete(raw_h5);
                fprintf('[disk] deleted the verified raw copy %s (%.1f GB freed)\n', raw_h5, b / 1e9);
            elseif isfile(raw_h5)
                fprintf(2, '[disk] %s KEPT: verification did not pass\n', raw_h5);
            end
        end
        % 3. EXTRACT + cell review
        cfg.stages = {'extract', 'review'};
        run_mini1p_day(cfg);
        fprintf('################ mouse %s DONE in %.1f h ################\n', m.id, toc(t0) / 3600);
    catch err
        fprintf(2, '\n*** mouse %s STOPPED after %.1f h: %s\n', m.id, toc(t0) / 3600, err.message);
    end
end
fprintf('\nCA BATCH FINISHED %s\n', char(datetime('now')));
