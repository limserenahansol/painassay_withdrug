%% RUN_ca_preview_curation_20261019.m  -  curation GUI data for every finished 3-min preview
% For each recording in mini1p_out\20261019_preview3min that has finished
% (<rec>_preview.json), this makes the data of cell_curation_gui (the same GUI as the 5941
% 5-min test: movie dF/F / raw, max dF/F, mean, Cn x PNR; keep k / delete d) and writes a
% launcher RUN_cell_curation_<rec>.m into that recording's folder: open it, press F5.
% Safe to run again: existing curation data is kept.

here = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_pipeline';
addpath(here);
OUT = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_out\20261019_preview3min';
L = dir(fullfile(OUT, '*_d1_*'));
for i = 1:numel(L)
    if ~L(i).isdir, continue; end
    rid = L(i).name;
    od = fullfile(OUT, rid);
    if ~isfile(fullfile(od, [rid '_preview.json'])), continue; end
    tag = regexprep(rid, '_(pre|post)$', '');
    ex = fullfile(od, '3_extract', [tag '_extract.mat']);
    rv = fullfile(od, '4_review', [tag '_cell_review.csv']);
    pm = fullfile(od, '3_extract', [tag '_projections.mat']);
    gd = fullfile(od, '4_review', [tag '_curation_data.mat']);
    if ~isfile(gd)
        try
            prepare_curation_data(ex, rv, pm, gd);
        catch e
            fprintf(2, '%s: curation data failed: %s\n', rid, e.message);
            continue;
        end
    end
    lf = fullfile(od, ['RUN_cell_curation_' rid '.m']);
    fid = fopen(lf, 'w');
    fprintf(fid, '%%%% RUN_cell_curation_%s.m  -  keep / delete every candidate of %s (first 3 min)\n', rid, rid);
    fprintf(fid, '%% Open in the MATLAB desktop and press F5. Same GUI as the 5941 5-min test:\n');
    fprintf(fid, '%%   LEFT  whole field; menu: movie dF/F | movie raw | max dF/F | mean | max (bg removed) | Cn x PNR\n');
    fprintf(fid, '%%         green = keep, red = delete, dashed = not reviewed yet\n');
    fprintf(fid, '%%   KEYS  k keep   d delete   -> / n next   <- / p previous   u next not reviewed\n');
    fprintf(fid, '%%         space play / pause   + / - speed   s save\n');
    fprintf(fid, '%% Saving writes the keep column of 4_review\\%s_cell_review.csv (a backup is kept).\n', tag);
    fprintf(fid, '%% The pre-filled keep = the rules fitted to your 5941 decisions (cell / check = 1, not_cell = 0).\n\n');
    fprintf(fid, 'addpath(''%s'');\n', here);
    fprintf(fid, 'cell_curation_gui(''%s'');\n', gd);
    fclose(fid);
    fprintf('%s: GUI ready -> %s\n', rid, lf);
end
