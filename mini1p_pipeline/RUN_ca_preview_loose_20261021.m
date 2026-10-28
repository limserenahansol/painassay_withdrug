%% RUN_ca_preview_loose_20261021.m  -  the 1-min previews again with looser detection + your marks
% Your review of the previews (2026-10-21 morning): green cells fine, but some red ones are
% neurons and some neurons were not found at all. Changed:
%   seeds   k_sd 3 -> 2 (dimmer Cn x PNR spots), min_sep 1.5 -> 1.0 radius (neighbours), and every
%           position you circled on the screenshots (analysis\curation_marks_20261021\marks*_<rid>.csv,
%           all screenshots of that recording) forced in as a seed
%   rules   a 'piece' must be >= 25 % of the biggest one (was 5 %), not_cell only with > 4 pieces
%           (was > 2), elongated > 4 (was 3), too large > 7x (was 5x)
%   (v2 kept far too many non-neuron shapes; the keep decision is then redone by
%    ca_preview_rule_v3.py: original greens + your marks + clear neuron shapes only)
% Pass 1 and the projections are re-used (same movie). Results next to the old ones:
%   <preview>\<rid>\<VNAME>\3_extract, \4_review, launcher RUN_cell_curation_<rid>_<VNAME>.m
% Run: matlab -batch "DO = {'5937_d1_pre'}; VNAME = 'v4_marks'; run('...\RUN_ca_preview_loose_20261021.m')"

here = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_pipeline';
addpath(here);
OUT = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_out\20261019_preview3min';
MARKS = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis\curation_marks_20261021';
if ~exist('DO', 'var'), DO = {'5939_d1_pre', '1191_d1_pre', '5937_d1_pre'}; end
if ~exist('VNAME', 'var'), VNAME = 'v2_loose'; end
SEED = struct('k_sd', 2, 'min_sep', 1.0);
QC = struct('piece_min_frac', 0.25, 'pieces_max', 4, 'axis_max', 4, 'area_max', 7);

for i = 1:numel(DO)
    rid = DO{i};
    t0 = tic;
    m = rid(1:4);
    tg = sprintf('%s_d1', m);
    od = fullfile(OUT, rid);
    vd = fullfile(od, VNAME);
    fprintf('\n######## loose preview %s -> %s  %s ########\n', rid, VNAME, char(datetime('now')));
    try
        mv = fullfile(od, '1_movie');
        mc_h5 = fullfile(mv, [rid '_mc.h5']);
        fcsv = fullfile(mv, [rid '_raw_frames.csv']);
        xd = fullfile(vd, '3_extract');
        if ~isfolder(fullfile(xd, 'projections')), mkdir(fullfile(xd, 'projections')); end
        copyfile(fullfile(od, '3_extract', 'projections', '*.mat'), fullfile(xd, 'projections'));
        sd = SEED;
        mfs = dir(fullfile(MARKS, ['marks*_' rid '.csv']));
        xr = zeros(0, 2);
        for q = 1:numel(mfs)
            K = readtable(fullfile(MARKS, mfs(q).name));
            if height(K), xr = [xr; K.fov_row, K.fov_col]; end %#ok<AGROW>
        end
        if ~isempty(xr)
            sd.extra_rc = xr;
            fprintf('%s: %d marked positions from %d screenshot(s)\n', rid, size(xr, 1), numel(mfs));
        end
        ex = extract_1p_seeded(struct('h5', mc_h5, 'frames_csv', fcsv, 'mouse_id', m, 'day', 1, ...
            'out_dir', xd, 'window_s', [15 60], 'detect_prebin', true, 'use_gpu', false, ...
            'pass1_mat', fullfile(od, '3_extract', 'pass1', [tg '_extract.mat']), 'seed', sd));
        rv = extract_cell_review(ex.files.mat, fullfile(vd, '4_review'), struct('qc', QC));
        gd = fullfile(vd, '4_review', [tg '_curation_data.mat']);
        prepare_curation_data(ex.files.mat, rv.csv, ex.projections_mat, gd);
        lf = fullfile(od, ['RUN_cell_curation_' rid '_' VNAME '.m']);
        fid = fopen(lf, 'w');
        fprintf(fid, '%%%% RUN_cell_curation_%s_%s.m  -  %s, first 1 min, looser detection + your marks (2026-10-21)\n', rid, VNAME, rid);
        fprintf(fid, '%% Open in the MATLAB desktop and press F5. Same GUI as before.\n');
        fprintf(fid, '%%   KEYS  k keep   d delete   -> / n next   <- / p previous   u next not reviewed\n');
        fprintf(fid, '%%         space play / pause   + / - speed   s save (writes keep in %s\\4_review\\%s_cell_review.csv)\n', VNAME, tg);
        fprintf(fid, 'addpath(''%s'');\ncell_curation_gui(''%s'');\n', here, gd);
        fclose(fid);
        fprintf('[loose] %s DONE in %.1f min: %d candidates -> %d cell, %d check, %d not_cell\n', rid, ...
                toc(t0) / 60, ex.n_cells, rv.n_cell, rv.n_check, rv.n_not_cell);
    catch err
        fprintf(2, '\n*** loose preview %s STOPPED after %.1f min: %s\n', rid, toc(t0) / 60, err.message);
        for k = 1:numel(err.stack), fprintf(2, '   %s line %d\n', err.stack(k).name, err.stack(k).line); end
    end
end
fprintf('\nLOOSE PREVIEW BATCH FINISHED %s\n', char(datetime('now')));
