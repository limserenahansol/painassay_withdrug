%% RUN_ca_preview3min_20261019.m  -  detection check of EVERY recording before trusting the day runs
% The cell recipe was optimised on 5 min of ONE recording (5941 SBI). Noise and signal can differ
% between recordings, so here each of the 8 recordings is processed alone on its first 3 min
% (5400 frames) with exactly the same steps as the day pipeline:
%   lossless copy (verified) -> NoRMCorre + lens crop -> QC media -> extract_1p_seeded (the recipe)
%   -> extract_cell_review (rules fitted to your 5941 decisions) -> review video
% and the noise / signal levels are measured, so the recordings can be compared.
% Per recording (mini1p_out\20261019_preview3min\<recording>\):
%   <rec>_detection_review_2x.mp4   left: motion-corrected movie; right: what EXTRACT sees (dF/F +
%                                   high-pass); every candidate outlined and numbered (green cell,
%                                   yellow check, red not a cell), 2x real time
%   4_review\<tag>_cells_map.png + pages, 3_extract\seeds\seeds.png, 2_qc\<rec>_qc_stages.png
%   <rec>_preview.json              noise, signal, motion and detection numbers
% Run: matlab -batch "DO_REC = [1 3 5 7]; run('...\RUN_ca_preview3min_20261019.m')"

here = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_pipeline';
addpath(here);
OUT = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_out\20261019_preview3min';
REC = {'5941', 'pre',  'D:\testHANSOL_20261019-111628\5941female_baseline_sBIday1_2026-10-19-11-58-32_video.isxd';
       '5941', 'post', 'D:\testHANSOL_20261019-121826\sbi5941_2026-10-19-12-27-44_video.isxd';
       '5937', 'pre',  'D:\testHANSOL_20261019-124926\5931malebaseline_2026-10-19-13-01-19_video.isxd';
       '5937', 'post', 'D:\testHANSOL_20261019-132106\5937male_control_2026-10-19-13-21-29_video.isxd';
       '5939', 'pre',  'D:\testHANSOL_20261019-134128\5939malebase_2026-10-19-13-48-58_video.isxd';
       '5939', 'post', 'D:\testHANSOL_20261019-141205\5939SBI_2026-10-19-14-17-35_video.isxd';
       '1191', 'pre',  'D:\testHANSOL_20261019-143901\1191femalebase_2026-10-19-14-44-13_video.isxd';
       '1191', 'post', 'D:\testHANSOL_20261019-150319\1191control_2026-10-19-15-04-45_video.isxd'};
if ~exist('N_FRAMES', 'var'), N_FRAMES = 1800; end   % 1 min at 30 Hz (user 2026-10-20: 1 min is enough; 5941 ran 3 min)
if ~exist('DO_REC', 'var'), DO_REC = 1:size(REC, 1); end

for i = DO_REC
    m = REC{i, 1}; as = REC{i, 2}; isxd = REC{i, 3};
    rid = sprintf('%s_d1_%s', m, as);
    od = fullfile(OUT, rid);
    jf = fullfile(od, [rid '_preview.json']);
    if isfile(jf), fprintf('%s: already done\n', rid); continue; end
    t0 = tic;
    fprintf('\n######## preview %s  %s ########\n', rid, char(datetime('now')));
    try
        mv = fullfile(od, '1_movie');
        mc_h5 = fullfile(mv, [rid '_mc.h5']);
        mcj = fullfile(mv, [rid '_mc.json']);
        fcsv = fullfile(mv, [rid '_raw_frames.csv']);
        done_mc = false;
        if isfile(mcj) && isfile(mc_h5)
            jj = jsondecode(fileread(mcj));
            done_mc = isfield(jj, 'status') && strcmp(jj.status, 'ok');
        end
        if done_mc
            % copy + motion correction already finished (a restart): re-use them
            mi = jsondecode(fileread(mcj));
            if ~iscell(mi.recordings), mi.recordings = num2cell(mi.recordings); end
            fprintf('[preview] %s re-uses its verified copy + motion correction\n', rid);
        else
            li = isxd_to_h5(struct('isxd', isxd, 'recording_id', rid, 'assay', as, 'name', rid, ...
                                   'out_dir', mv, 'max_frames', N_FRAMES, 'verify', 'full'));
            fprintf('[preview] %s lossless: passed %d, max |diff| %g\n', rid, li.verification.passed, ...
                    li.verification.max_abs_diff);
            mi = motion_correct_h5(struct('inputs', {{li.files.h5}}, 'frames_csv', {{li.files.frames_csv}}, ...
                                          'names', {{rid}}, 'tag', rid, 'out_dir', mv, 'crop_rect', 'auto', ...
                                          'crop_margin_px', 40));
            try
                qc_stage_media(struct('raw_h5', li.files.h5, 'mc_h5', mc_h5, 'frames_csv', li.files.frames_csv, ...
                                      'out_dir', fullfile(od, '2_qc'), 'tag', rid, 'window_s', [20 50]));
            catch e
                fprintf(2, '[preview] %s QC media failed: %s\n', rid, e.message);
            end
            if li.verification.passed && isfile(li.files.h5), delete(li.files.h5); end
            fcsv = li.files.frames_csv;
        end
        ex = extract_1p_seeded(struct('h5', mc_h5, 'frames_csv', fcsv, 'mouse_id', m, ...
                                      'day', 1, 'out_dir', fullfile(od, '3_extract'), 'window_s', [15 N_FRAMES / 30], ...
                                      'detect_prebin', true, 'use_gpu', false));   % one partition (no seams), CPU (GPU was oversubscribed)
        rv = extract_cell_review(ex.files.mat, fullfile(od, '4_review'));
        vid = fullfile(od, [rid '_detection_review_2x.mp4']);
        fov_review_video(ex.files.mat, rv.csv, vid, struct('bin_time', 2));
        % the curation GUI (as the 5941 test, 5c_cell_review_loose) + its launcher
        tg = sprintf('%s_d1', m);
        gd = fullfile(od, '4_review', [tg '_curation_data.mat']);
        prepare_curation_data(ex.files.mat, rv.csv, ex.projections_mat, gd);
        lf = fullfile(od, ['RUN_cell_curation_' rid '.m']);
        fid = fopen(lf, 'w');
        fprintf(fid, '%%%% RUN_cell_curation_%s.m  -  keep / delete every candidate of %s (first 3 min)\n', rid, rid);
        fprintf(fid, '%% Open in the MATLAB desktop and press F5. Same GUI as the 5941 5-min test.\n');
        fprintf(fid, '%%   menu: movie dF/F | movie raw | max dF/F | mean | max (bg removed) | Cn x PNR\n');
        fprintf(fid, '%%   KEYS  k keep   d delete   -> / n next   <- / p previous   u next not reviewed\n');
        fprintf(fid, '%%         space play / pause   + / - speed   s save (writes keep in 4_review\\%s_cell_review.csv)\n', tg);
        fprintf(fid, 'addpath(''%s'');\ncell_curation_gui(''%s'');\n', here, gd);
        fclose(fid);
        % ---- noise / signal / motion / detection numbers --------------------------------------
        P = load(ex.projections_mat);
        E = load(ex.files.mat, 'movie_mask', 'mean_image');
        mk = logical(E.movie_mask);
        if isempty(mk) || ~any(mk(:)), mk = true(size(P.Cn)); end
        R = readtable(rv.csv, 'TextType', 'char');
        kept = ~strcmp(R.verdict, 'not_cell');
        sh = cellfun(@(r) r.stats.abs_shift_max, mi.recordings);
        if ~isfield(mi, 'crop_fraction_kept'), mi.crop_fraction_kept = NaN; end
        met = struct('recording_id', rid, 'frames', N_FRAMES, 'crop_rect', mi.crop_rect, ...
            'crop_fraction_kept', mi.crop_fraction_kept, 'max_shift_px', max(sh), ...
            'corr_after_median', mi.recordings{1}.stats.corr_after_median, ...
            'brightness_median_counts', median(double(E.mean_image(mk))), ...
            'noise_std_dff_median', median(double(P.std_dff(mk))), ...
            'Cn_median', median(double(P.Cn(mk))), 'Cn_p99', prctile(double(P.Cn(mk)), 99), ...
            'PNR_median', median(double(P.PNR(mk))), 'PNR_p99', prctile(double(P.PNR(mk)), 99), ...
            'n_pass1', ex.n_pass1, 'n_seeds', ex.n_seeds, 'n_candidates', ex.n_cells, ...
            'n_cell', rv.n_cell, 'n_check', rv.n_check, 'n_not_cell', rv.n_not_cell, ...
            'snr_median_kept', median(R.snr(kept), 'omitnan'), ...
            'event_r_median_kept', median(R.event_image_corr(kept), 'omitnan'), ...
            'video', vid, 'cells_map', rv.map_png, 'minutes', toc(t0) / 60);
        fid = fopen(jf, 'w'); fprintf(fid, '%s', jsonencode(met, 'PrettyPrint', true)); fclose(fid);
        fprintf('[preview] %s DONE in %.0f min: %d candidates -> %d cell, %d check, %d not_cell\n', rid, ...
                toc(t0) / 60, ex.n_cells, rv.n_cell, rv.n_check, rv.n_not_cell);
    catch err
        fprintf(2, '\n*** preview %s STOPPED after %.0f min: %s\n', rid, toc(t0) / 60, err.message);
    end
end
fprintf('\nPREVIEW BATCH FINISHED %s\n', char(datetime('now')));
