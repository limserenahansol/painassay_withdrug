%% RUN_ca_preview_finish_20261019.m  -  finish the 1-min previews that stopped at the GUI data
% prepare_curation_data chose a 120 s window and failed on 1-min movies (fixed: a shorter
% movie now uses all of it). For every preview with EXTRACT + review done but no
% <rec>_preview.json, this makes the curation GUI data, the launcher and the metrics JSON,
% without re-running anything else.

here = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_pipeline';
addpath(here);
OUT = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_out\20261019_preview3min';
L = dir(fullfile(OUT, '*_d1_*'));
logs = dir(fullfile(OUT, 'preview_*.log'));
logtxt = '';
for k = 1:numel(logs), logtxt = [logtxt fileread(fullfile(OUT, logs(k).name))]; end %#ok<AGROW>
for i = 1:numel(L)
    if ~L(i).isdir, continue; end
    rid = L(i).name;
    od = fullfile(OUT, rid);
    jf = fullfile(od, [rid '_preview.json']);
    tag = regexprep(rid, '_(pre|post)$', '');
    exm = fullfile(od, '3_extract', [tag '_extract.mat']);
    rvc = fullfile(od, '4_review', [tag '_cell_review.csv']);
    pm = fullfile(od, '3_extract', [tag '_projections.mat']);
    gd = fullfile(od, '4_review', [tag '_curation_data.mat']);
    if isfile(jf) || ~isfile(exm) || ~isfile(rvc), continue; end
    if ~contains(logtxt, ['preview ' rid ' STOPPED']), continue; end   % still running, or not started
    t0 = tic;
    try
        if ~isfile(gd), prepare_curation_data(exm, rvc, pm, gd); end
        lf = fullfile(od, ['RUN_cell_curation_' rid '.m']);
        fid = fopen(lf, 'w');
        fprintf(fid, '%%%% RUN_cell_curation_%s.m  -  keep / delete every candidate of %s (first 1 min)\n', rid, rid);
        fprintf(fid, '%% Open in the MATLAB desktop and press F5. Same GUI as the 5941 5-min test.\n');
        fprintf(fid, '%%   menu: movie dF/F | movie raw | max dF/F | mean | max (bg removed) | Cn x PNR\n');
        fprintf(fid, '%%   KEYS  k keep   d delete   -> / n next   <- / p previous   u next not reviewed\n');
        fprintf(fid, '%%         space play / pause   + / - speed   s save (writes keep in 4_review\\%s_cell_review.csv)\n', tag);
        fprintf(fid, '%% The pre-filled keep = the rules fitted to your 5941 decisions (cell / check = 1, not_cell = 0).\n');
        fprintf(fid, 'addpath(''%s'');\ncell_curation_gui(''%s'');\n', here, gd);
        fclose(fid);
        % metrics, as the preview script writes them
        mi = jsondecode(fileread(fullfile(od, '1_movie', [rid '_mc.json'])));
        recs = mi.recordings; if ~iscell(recs), recs = num2cell(recs); end
        P = load(pm);
        E = load(exm, 'movie_mask', 'mean_image', 'temporal_raw');
        mk = logical(E.movie_mask);
        if isempty(mk) || ~any(mk(:)), mk = true(size(P.Cn)); end
        R = readtable(rvc, 'TextType', 'char');
        kept = ~strcmp(R.verdict, 'not_cell');
        tok = regexp(logtxt, ['extract_1p_seeded ' tag ': pass 1 (\d+) cells \+ (\d+) seeds -> (\d+) candidates -> [^\n]*' ...
                              regexptranslate('escape', rid)], 'tokens');
        n1 = NaN; ns = NaN;
        if ~isempty(tok), n1 = str2double(tok{end}{1}); ns = str2double(tok{end}{2}); end
        if ~isfield(mi, 'crop_fraction_kept'), mi.crop_fraction_kept = NaN; end
        sh = cellfun(@(r) r.stats.abs_shift_max, recs);
        met = struct('recording_id', rid, 'frames', size(E.temporal_raw, 1), 'crop_rect', mi.crop_rect, ...
            'crop_fraction_kept', mi.crop_fraction_kept, 'max_shift_px', max(sh), ...
            'corr_after_median', recs{1}.stats.corr_after_median, ...
            'brightness_median_counts', median(double(E.mean_image(mk))), ...
            'noise_std_dff_median', median(double(P.std_dff(mk))), ...
            'Cn_median', median(double(P.Cn(mk))), 'Cn_p99', prctile(double(P.Cn(mk)), 99), ...
            'PNR_median', median(double(P.PNR(mk))), 'PNR_p99', prctile(double(P.PNR(mk)), 99), ...
            'n_pass1', n1, 'n_seeds', ns, 'n_candidates', height(R), ...
            'n_cell', nnz(strcmp(R.verdict, 'cell')), 'n_check', nnz(strcmp(R.verdict, 'check')), ...
            'n_not_cell', nnz(strcmp(R.verdict, 'not_cell')), ...
            'snr_median_kept', median(R.snr(kept), 'omitnan'), ...
            'event_r_median_kept', median(R.event_image_corr(kept), 'omitnan'), ...
            'video', fullfile(od, [rid '_detection_review_2x.mp4']), ...
            'cells_map', fullfile(od, '4_review', [tag '_cells_map.png']), 'minutes', NaN);
        fid = fopen(jf, 'w'); fprintf(fid, '%s', jsonencode(met, 'PrettyPrint', true)); fclose(fid);
        fprintf('[finish] %s DONE in %.1f min: %d candidates -> %d cell, %d check, %d not_cell\n', rid, ...
                toc(t0) / 60, met.n_candidates, met.n_cell, met.n_check, met.n_not_cell);
    catch err
        fprintf(2, '*** finish %s failed: %s\n', rid, err.message);
    end
end
fprintf('PREVIEW FINISH DONE %s\n', char(datetime('now')));
