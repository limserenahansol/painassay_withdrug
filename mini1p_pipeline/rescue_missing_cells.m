function out = rescue_missing_cells(extract_mat, review_csv, decisions_json, opts)
%RESCUE_MISSING_CELLS  Put back the cells you KEPT on a reference movie that the whole-day
%   EXTRACT run did not find again.
%
%   out = rescue_missing_cells(extract_mat, review_csv, decisions_json)
%   out = rescue_missing_cells(..., opts)
%
%   decisions_json : <tag>_decisions.json of ca_transfer_decisions.py: user_keeps_not_found
%                    (reference label + cell id) and, per reference, its EXTRACT file and the
%                    registration shift (day pixel (y, x) = reference pixel (y + dy, x + dx)).
%   Each such footprint is shifted into the day frame (whole pixels), normalised to max 1 and
%   added to the day's footprints; then ALL traces are re-fitted jointly by least squares,
%   exactly as extract_1p does it (per-pixel mean removed, the same spatial high-pass and mask,
%   T = (S'S)^-1 S'M, divided by the footprint-weighted mean F), so a rescued cell and its
%   neighbours share their overlapping pixels correctly. Nothing else changes.
%   The day files are replaced after a backup (<tag>_extract_before_rescue.mat,
%   <tag>_cell_review_before_rescue.csv); the review table gets one row per rescued cell
%   (verdict 'rescued', keep 1). Running it again does nothing (the rescued cells now match).
%
%   opts (default): min_area_px (20) - footprints that end up outside the cropped day frame
%                   are skipped; block (1000) frames per LS block (CPU); use_gpu ('auto': GPU, 250-frame blocks, CPU fallback)

if nargin < 4, opts = struct(); end
min_area = getopt(opts, 'min_area_px', 20);
block = getopt(opts, 'block', 1000);
use_gpu = getopt(opts, 'use_gpu', 'auto');      % the spatial filter on the GPU in 250-frame blocks
                                                % (3.3x faster, ~3 GB; falls back to the CPU on any error)
js = jsondecode(fileread(decisions_json));
miss = js.user_keeps_not_found;
out = struct('n_rescued', 0, 'rescued', {{}});
if isempty(miss)
    fprintf('rescue_missing_cells: nothing to rescue (%s)\n', decisions_json);
    return;
end
if ~iscell(miss), miss = num2cell(miss); end
regs = js.registration;
if ~iscell(regs), regs = num2cell(regs); end
m = load(extract_mat);
[H, W, C] = size(m.spatial_weights);
add = zeros(H, W, 0, 'single');
info = {};
refcache = containers.Map();
for k = 1:numel(miss)
    q = miss{k};
    rg = [];
    for r = 1:numel(regs)
        if strcmp(regs{r}.label, q.ref), rg = regs{r}; end
    end
    if isempty(rg), continue; end
    if isfield(q, 'best_cos') && q.best_cos >= 0.5
        bd = NaN;
        if isfield(q, 'best_day_id'), bd = q.best_day_id; end
        fprintf('  %s #%d: overlaps day component #%d (cos %.2f) - not added (would split one cell)\n', ...
                q.ref, q.ref_id, bd, q.best_cos);
        continue;
    end
    if ~isKey(refcache, rg.mat)
        refcache(rg.mat) = load(rg.mat, 'spatial_weights');
    end
    R = refcache(rg.mat).spatial_weights;
    fr = single(R(:, :, q.ref_id));
    [Hr, Wr] = size(fr);
    D = zeros(H, W, 'single');
    ys = (1:H) + rg.dy; xs = (1:W) + rg.dx;          % day (y, x) = ref (y + dy, x + dx)
    oy = ys >= 1 & ys <= Hr; ox = xs >= 1 & xs <= Wr;
    D(oy, ox) = fr(ys(oy), xs(ox));
    D(D < 0) = 0;
    if nnz(D > 0.3 * max(D(:))) < min_area || max(D(:)) <= 0
        fprintf('  %s #%d: outside the day frame - skipped\n', q.ref, q.ref_id);
        continue;
    end
    add(:, :, end + 1) = D / max(D(:)); %#ok<AGROW>
    info{end + 1} = sprintf('%s #%d', q.ref, q.ref_id); %#ok<AGROW>
end
K = size(add, 3);
if K == 0
    fprintf('rescue_missing_cells: no footprint inside the day frame\n');
    return;
end
S = cat(3, single(m.spatial_weights), add);
c1 = m.config.extract_1p;
tbd = fullfile(getenv('USERPROFILE'), 'OneDrive', 'Desktop', 'cursor', 'pain_2plane_split');
if isfield(c1, 'toolbox_dir') && isfolder(c1.toolbox_dir), tbd = c1.toolbox_dir; end
addpath(tbd);
mini2p_toolbox_paths();          % EXTRACT's spatial_bandpass, without ActSort shadowing it
h5 = c1.h5;
if ~isfile(h5)
    h5 = fullfile(fileparts(fileparts(extract_mat)), '1_movie', [regexprep(char(m.mouse_id), '\s', '') ...
         sprintf('_d%d', m.day) '_mc_day.h5']);
end
% extract_1p masks the LS movie only when EXTRACT itself saw the mask (binned detection movie,
% or detection at input resolution)
mask = [];
pb = isfield(c1, 'detect_prebin') && ~isempty(c1.detect_prebin) && logical(c1.detect_prebin);
dss1 = isfield(c1, 'detect_downsample_space_by') && isequal(c1.detect_downsample_space_by, 1);
if (pb || dss1) && isfield(m, 'movie_mask') && ~isempty(m.movie_mask) && ~all(m.movie_mask(:))
    mask = logical(m.movie_mask);
end
% a footprint (mostly) outside the lit lens area has no signal there (the 3-min test: a rescued
% cell with 0.0002 % of its mass inside the mask got a flat zero trace) - not added
if isfield(m, 'movie_mask') && ~isempty(m.movie_mask) && ~all(m.movie_mask(:))
    lm = single(logical(m.movie_mask));
    inside = arrayfun(@(j) sum(sum(add(:, :, j) .* lm)) / max(sum(sum(add(:, :, j))), eps), 1:K);
    for j = find(inside < 0.5)
        fprintf('  %s: %.0f%% inside the lens mask - not added\n', info{j}, 100 * inside(j));
    end
    add = add(:, :, inside >= 0.5);
    info = info(inside >= 0.5);
    K = size(add, 3);
    if K == 0
        fprintf('rescue_missing_cells: no footprint inside the lens\n');
        return;
    end
    S = cat(3, single(m.spatial_weights), add);
end
radius = double(c1.avg_cell_radius);
hp = double(m.config.extract.spatial_highpass_cutoff);
fprintf('rescue_missing_cells: %d footprints added to %d; joint least squares on %s\n', K, C, h5);
t0 = tic;
T = [];
if ~isequal(use_gpu, false)
    try
        if gpuDeviceCount > 0
            T = ls_traces(h5, S, single(m.mean_image), mask, radius, hp, true, 250);
            fprintf('  least squares on the GPU\n');
        end
    catch e
        fprintf('  GPU least squares failed (%s) - CPU\n', e.message);
        T = [];
    end
end
if isempty(T)
    T = ls_traces(h5, S, single(m.mean_image), mask, radius, hp, false, block);
end
fprintf('  least squares %.1f min\n', toc(t0) / 60);
ft = m.frame_table;
valid_rows = find(ft.valid == 1);
n_full = numel(ft.isx_frame);
tr = NaN(n_full, C + K, 'single');
tr(valid_rows, :) = T(ft.h5_row(valid_rows), :);
% backups, then the augmented files
[d, nm] = fileparts(extract_mat);
bk = fullfile(d, [regexprep(nm, '_extract$', '') '_extract_before_rescue.mat']);
if ~isfile(bk), copyfile(extract_mat, bk); end
[d2, nm2] = fileparts(review_csv);
bk2 = fullfile(d2, [nm2 '_before_rescue.csv']);
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
m.rescued = struct('ids', C + (1:K), 'source', {info}, 'decisions_json', decisions_json, ...
                   'note', 'footprint from your reference decision; trace = joint least squares with every candidate');
Rt = readtable(review_csv, 'TextType', 'char', 'Delimiter', ',');
if height(Rt) ~= C
    error('rescue_missing_cells:table', '%s has %d rows, the EXTRACT file %d components', review_csv, height(Rt), C);
end
for v = {'verdict', 'note', 'reasons'}                % text columns: always cell arrays
    if ismember(v{1}, Rt.Properties.VariableNames) && ~iscell(Rt.(v{1}))
        x = Rt.(v{1});
        c = repmat({''}, height(Rt), 1);
        ok = ~ismissing(x);
        c(ok) = cellstr(string(x(ok)));
        Rt.(v{1}) = c;
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
        row.note = {sprintf('rescued: your keep %s, not found by the whole-day EXTRACT', info{j})};
    end
    Rt = [Rt; row]; %#ok<AGROW>
end
fmt = '-v7';                                       % as extract_1p: the Python steps read it with scipy
w = whos('m');
if w.bytes > 1.9e9, fmt = '-v7.3'; warning('rescue_missing_cells:v73', 'saved as -v7.3 (too large for -v7)'); end
save(extract_mat, '-struct', 'm', fmt);
writetable(Rt, review_csv);
out = struct('n_rescued', K, 'rescued', {info}, 'ids', C + (1:K), 'backup_mat', bk, 'backup_csv', bk2);
fprintf('rescue_missing_cells: %d cells rescued as #%d-#%d -> %s\n', K, C + 1, C + K, extract_mat);
end


function T = ls_traces(h5, S, F, mask, radius, hp_cutoff, use_gpu, block)
% the least-squares step of extract_1p (least_squares_traces), unchanged
[H, W, C] = size(S);
hi = h5info(h5, '/mov');
n = hi.Dataspace.Size(3);
S2 = reshape(double(S), H * W, C);
Ssp = sparse(S2);
G = full(Ssp' * Ssp);
B = zeros(C, n);
for i0 = 1:block:n
    nb = min(block, n - i0 + 1);
    Mb = h5read(h5, '/mov', [1 1 i0], [H W nb]);
    Mb = bsxfun(@minus, single(Mb), single(F));
    if ~isinf(hp_cutoff)
        Mb = spatial_bandpass(Mb, radius, hp_cutoff, inf, use_gpu);
        if use_gpu, Mb = gather(Mb); end
    end
    if ~isempty(mask)
        Mb = bsxfun(@times, Mb, single(mask));
    end
    B(:, i0:i0 + nb - 1) = Ssp' * double(reshape(Mb, H * W, nb));
end
if rcond(G) > 1e-10
    X = G \ B;
else
    X = pinv(G) * B;
end
F_cell = (S2 ./ max(sum(S2, 1), eps))' * double(F(:));
T = single(bsxfun(@rdivide, X', F_cell'));
end


function v = getopt(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
