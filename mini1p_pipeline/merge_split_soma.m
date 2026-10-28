function out = merge_split_soma(extract_mat, keep_id, merge_ids, opts)
%MERGE_SPLIT_SOMA  One soma that EXTRACT cut into several components -> ONE neuron (you, 2026-10-22: 1191 Day 1 #1 + #2,
%   the brightest neuron of the field). The footprint of keep_id becomes the sum of the max-normalised footprints
%   (keep_id + merge_ids), max 1; then every trace is re-fitted jointly by least squares WITHOUT the merged-away
%   components (ls_traces_joint, as rescue_missing_cells), so the merged neuron gets the whole soma's signal. The
%   merged-away components keep their old footprint and trace (not kept; the review table says "merged into #k").
%
%   out = merge_split_soma(extract_mat, keep_id, merge_ids)
%   Backup first: <tag>_extract_before_merge.mat. opts: block (1000), use_gpu ('auto')
if nargin < 4, opts = struct(); end
block = getopt(opts, 'block', 1000);
use_gpu = getopt(opts, 'use_gpu', 'auto');
m = load(extract_mat);
[H, W, C] = size(m.spatial_weights);
S = single(m.spatial_weights);
f = S(:, :, keep_id) / max(max(S(:, :, keep_id)));
for j = merge_ids(:)'
    f = f + S(:, :, j) / max(max(S(:, :, j)));
end
f(f < 0) = 0;
S(:, :, keep_id) = f / max(f(:));
use = setdiff(1:C, merge_ids);
c1 = m.config.extract_1p;
tbd = fullfile(getenv('USERPROFILE'), 'OneDrive', 'Desktop', 'cursor', 'pain_2plane_split');
if isfield(c1, 'toolbox_dir') && isfolder(c1.toolbox_dir), tbd = c1.toolbox_dir; end
addpath(tbd);
mini2p_toolbox_paths();
h5 = c1.h5;
mask = [];
pb = isfield(c1, 'detect_prebin') && ~isempty(c1.detect_prebin) && logical(c1.detect_prebin);
dss1 = isfield(c1, 'detect_downsample_space_by') && isequal(c1.detect_downsample_space_by, 1);
if (pb || dss1) && isfield(m, 'movie_mask') && ~isempty(m.movie_mask) && ~all(m.movie_mask(:))
    mask = logical(m.movie_mask);
end
radius = double(c1.avg_cell_radius);
hp = double(m.config.extract.spatial_highpass_cutoff);
fprintf('merge_split_soma: #%d + #%s -> #%d; joint least squares of %d components on %s\n', keep_id, ...
        strjoin(arrayfun(@num2str, merge_ids(:)', 'UniformOutput', false), ' #'), keep_id, numel(use), h5);
t0 = tic;
T = [];
if ~isequal(use_gpu, false)
    try
        if gpuDeviceCount > 0
            T = ls_traces_joint(h5, S(:, :, use), single(m.mean_image), mask, radius, hp, true, 250);
            fprintf('  least squares on the GPU\n');
        end
    catch e
        fprintf('  GPU least squares failed (%s) - CPU\n', e.message);
        T = [];
    end
end
if isempty(T)
    T = ls_traces_joint(h5, S(:, :, use), single(m.mean_image), mask, radius, hp, false, block);
end
fprintf('  least squares %.1f min\n', toc(t0) / 60);
ft = m.frame_table;
valid_rows = find(ft.valid == 1);
[d, nm] = fileparts(extract_mat);
bk = fullfile(d, [regexprep(nm, '_extract$', '') '_extract_before_merge.mat']);
if ~isfile(bk), copyfile(extract_mat, bk); end
tr = m.temporal_raw;
tr(valid_rows, use) = T(ft.h5_row(valid_rows), :);
m.temporal_raw = tr;
m.temporal_extract(valid_rows, keep_id) = tr(valid_rows, keep_id);
m.spatial_weights = S;
if isfield(m, 'F_per_cell')
    s2 = double(reshape(S(:, :, keep_id), [], 1));
    m.F_per_cell(keep_id) = (s2 / max(sum(s2), eps))' * double(m.mean_image(:));
end
if isfield(m, 'cell_qc') && isstruct(m.cell_qc)
    bw = S(:, :, keep_id) > 0.3;
    rp = regionprops(bw, 'Area', 'Centroid', 'MajorAxisLength', 'MinorAxisLength');
    [~, b] = max([rp.Area]);
    m.cell_qc.centroid_xy(keep_id, :) = rp(b).Centroid;
    m.cell_qc.area_px(keep_id, 1) = rp(b).Area;
    m.cell_qc.axis_ratio(keep_id, 1) = rp(b).MajorAxisLength / max(rp(b).MinorAxisLength, eps);
    m.cell_qc.n_blobs(keep_id, 1) = numel(rp);
    m.cell_qc.area_over_expected(keep_id, 1) = rp(b).Area / (pi * radius ^ 2);
end
m.merged = struct('keep_id', keep_id, 'merge_ids', merge_ids, 'created', datestr(now, 'yyyy-mm-dd HH:MM'), ...
                  'note', 'one soma cut into pieces by EXTRACT; footprint = sum, traces = joint least squares without the pieces');
fmt = '-v7';
w = whos('m');
if w.bytes > 1.9e9, fmt = '-v7.3'; end
save(extract_mat, '-struct', 'm', fmt);
c = m.cell_qc.centroid_xy(keep_id, :);
out = struct('keep_id', keep_id, 'merge_ids', merge_ids, 'row', c(2), 'col', c(1), 'backup', bk);
fprintf('merge_split_soma: done (centre row %.1f, col %.1f) -> %s\n', c(2), c(1), extract_mat);
end


function v = getopt(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
