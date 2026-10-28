function out = rescue_footprints(extract_mat, review_csv, add_mat, opts)
%RESCUE_FOOTPRINTS  Add footprints that are already in the day frame (ca_day2_recover.py: Day-1 confirmed neurons
%   not found on Day 2, their Day-1 footprint moved to Day 2) and re-fit EVERY trace jointly by least squares, as
%   rescue_missing_cells does (same movie, high-pass, mask; ls_traces_joint). EXTRACT's iterations cannot remove them.
%
%   out = rescue_footprints(extract_mat, review_csv, add_mat)
%   add_mat : add (H x W x K, max 1), day1_id (K)
%   Backups first (<tag>_extract_before_recover.mat, <tag>_cell_review_before_recover.csv); the review table gets
%   one row per added footprint (verdict 'rescued', keep 1, note 'recovered: Day-1 #k ...'). The footprints that
%   are mostly outside the lit lens area are skipped (printed).
%   opts: block (1000), use_gpu ('auto')
if nargin < 4, opts = struct(); end
block = getopt(opts, 'block', 1000);
use_gpu = getopt(opts, 'use_gpu', 'auto');
A = load(add_mat);
add = single(A.add);
did = double(A.day1_id(:));
m = load(extract_mat);
[H, W, C] = size(m.spatial_weights);
if size(add, 1) ~= H || size(add, 2) ~= W
    error('rescue_footprints:size', 'footprints %d x %d, day frame %d x %d', size(add, 1), size(add, 2), H, W);
end
K = size(add, 3);
c1 = m.config.extract_1p;
tbd = fullfile(getenv('USERPROFILE'), 'OneDrive', 'Desktop', 'cursor', 'pain_2plane_split');
if isfield(c1, 'toolbox_dir') && isfolder(c1.toolbox_dir), tbd = c1.toolbox_dir; end
addpath(tbd);
mini2p_toolbox_paths();          % EXTRACT's spatial_bandpass, without ActSort shadowing it
h5 = c1.h5;
mask = [];
pb = isfield(c1, 'detect_prebin') && ~isempty(c1.detect_prebin) && logical(c1.detect_prebin);
dss1 = isfield(c1, 'detect_downsample_space_by') && isequal(c1.detect_downsample_space_by, 1);
if (pb || dss1) && isfield(m, 'movie_mask') && ~isempty(m.movie_mask) && ~all(m.movie_mask(:))
    mask = logical(m.movie_mask);
end
keepk = true(1, K);
if isfield(m, 'movie_mask') && ~isempty(m.movie_mask) && ~all(m.movie_mask(:))
    lm = single(logical(m.movie_mask));
    for j = 1:K
        inside = sum(sum(add(:, :, j) .* lm)) / max(sum(sum(add(:, :, j))), eps);
        if inside < 0.5
            fprintf('  Day-1 #%d: %.0f%% inside the lens mask - not added\n', did(j), 100 * inside);
            keepk(j) = false;
        end
    end
end
add = add(:, :, keepk); did = did(keepk); K = size(add, 3);
if K == 0
    fprintf('rescue_footprints: nothing to add\n'); out = struct('n_added', 0); return;
end
S = cat(3, single(m.spatial_weights), add);
radius = double(c1.avg_cell_radius);
hp = double(m.config.extract.spatial_highpass_cutoff);
fprintf('rescue_footprints: %d footprints added to %d; joint least squares on %s\n', K, C, h5);
t0 = tic;
T = [];
if ~isequal(use_gpu, false)
    try
        if gpuDeviceCount > 0
            T = ls_traces_joint(h5, S, single(m.mean_image), mask, radius, hp, true, 250);
            fprintf('  least squares on the GPU\n');
        end
    catch e
        fprintf('  GPU least squares failed (%s) - CPU\n', e.message);
        T = [];
    end
end
if isempty(T)
    T = ls_traces_joint(h5, S, single(m.mean_image), mask, radius, hp, false, block);
end
fprintf('  least squares %.1f min\n', toc(t0) / 60);
ft = m.frame_table;
valid_rows = find(ft.valid == 1);
n_full = numel(ft.isx_frame);
tr = NaN(n_full, C + K, 'single');
tr(valid_rows, :) = T(ft.h5_row(valid_rows), :);
[d, nm] = fileparts(extract_mat);
bk = fullfile(d, [regexprep(nm, '_extract$', '') '_extract_before_recover.mat']);
if ~isfile(bk), copyfile(extract_mat, bk); end
[d2, nm2] = fileparts(review_csv);
bk2 = fullfile(d2, [nm2 '_before_recover.csv']);
if ~isfile(bk2), copyfile(review_csv, bk2); end
m.spatial_weights = S;
m.temporal_raw = tr;
te = NaN(n_full, K, 'single');
te(valid_rows, :) = tr(valid_rows, C + 1:end);
m.temporal_extract = [m.temporal_extract, te];
if isfield(m, 'F_per_cell')
    S2 = reshape(double(add), H * W, K);
    m.F_per_cell = [m.F_per_cell(:); (S2 ./ max(sum(S2, 1), eps))' * double(m.mean_image(:))];
end
if isfield(m, 'cell_qc') && isstruct(m.cell_qc)
    for j = 1:K
        bw = add(:, :, j) > 0.3;
        rp = regionprops(bw, 'Area', 'Centroid', 'MajorAxisLength', 'MinorAxisLength');
        [~, b] = max([rp.Area]);
        m.cell_qc.centroid_xy(end + 1, :) = rp(b).Centroid;
        m.cell_qc.area_px(end + 1, 1) = rp(b).Area;
        m.cell_qc.axis_ratio(end + 1, 1) = rp(b).MajorAxisLength / max(rp(b).MinorAxisLength, eps);
        m.cell_qc.n_blobs(end + 1, 1) = numel(rp);
        m.cell_qc.area_over_expected(end + 1, 1) = rp(b).Area / (pi * radius ^ 2);
    end
end
m.recovered = struct('ids', C + (1:K), 'day1_id', did, 'add_mat', add_mat, ...
                     'note', 'Day-1 footprint moved to Day 2 (ca_day2_recover.py); trace = joint least squares with every candidate');
Rt = readtable(review_csv, 'TextType', 'char', 'Delimiter', ',');
if height(Rt) ~= C
    error('rescue_footprints:table', '%s has %d rows, the EXTRACT file %d components', review_csv, height(Rt), C);
end
for v = {'verdict', 'note', 'reasons'}
    if ismember(v{1}, Rt.Properties.VariableNames) && ~iscell(Rt.(v{1}))
        x = Rt.(v{1}); c = repmat({''}, height(Rt), 1); ok = ~ismissing(x); c(ok) = cellstr(string(x(ok))); Rt.(v{1}) = c;
    end
end
for j = 1:K
    row = Rt(1, :);
    for v = Rt.Properties.VariableNames
        x = row.(v{1});
        if iscell(x), row.(v{1}) = {''}; elseif isnumeric(x) || islogical(x), row.(v{1}) = NaN; end
    end
    row.id = C + j;
    cxy = m.cell_qc.centroid_xy(C + j, :);
    if ismember('row', Rt.Properties.VariableNames), row.row = cxy(2); end
    if ismember('col', Rt.Properties.VariableNames), row.col = cxy(1); end
    row.verdict = {'rescued'};
    row.keep = 1;
    if ismember('note', Rt.Properties.VariableNames)
        row.note = {sprintf('recovered: Day-1 #%d footprint moved to Day 2 (not found by the Day-2 EXTRACT)', did(j))};
    end
    Rt = [Rt; row]; %#ok<AGROW>
end
fmt = '-v7';
w = whos('m');
if w.bytes > 1.9e9, fmt = '-v7.3'; warning('rescue_footprints:v73', 'saved as -v7.3 (too large for -v7)'); end
save(extract_mat, '-struct', 'm', fmt);
writetable(Rt, review_csv);
out = struct('n_added', K, 'ids', C + (1:K), 'day1_id', did, 'backup_mat', bk, 'backup_csv', bk2);
fprintf('rescue_footprints: %d Day-1 footprints added as #%d-#%d -> %s\n', K, C + 1, C + K, extract_mat);
end


function v = getopt(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
