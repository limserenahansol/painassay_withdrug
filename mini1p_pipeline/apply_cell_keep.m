function out = apply_cell_keep(extract_mat, review_csv, out_mat)
%APPLY_CELL_KEEP  Keep only the components you marked keep = 1 in the cell review table.
%
%   out = apply_cell_keep(extract_mat, review_csv)
%   out = apply_cell_keep(extract_mat, review_csv, out_mat)
%
%   extract_mat : <tag>_extract.mat of extract_1p (never modified)
%   review_csv  : <tag>_cell_review.csv of extract_cell_review, after your edits
%   out_mat     : default <tag>_extract_kept.mat next to extract_mat
%
%   The kept file has every field of the EXTRACT file, with spatial_weights,
%   temporal_raw, temporal_extract (and the per-cell vectors) reduced to the kept
%   components, plus
%     kept_ids     original component numbers (the numbers on the review map / pages)
%     review       the review table as applied
%   merge_mini1p_scoring uses kept_ids as the cell id, so every later table and figure
%   names a cell by the number you saw in the review.

if nargin < 3 || isempty(out_mat)
    out_mat = regexprep(extract_mat, '_extract\.mat$', '_extract_kept.mat');
    if strcmp(out_mat, extract_mat)
        out_mat = strrep(extract_mat, '.mat', '_kept.mat');
    end
end
if strcmp(out_mat, extract_mat)
    error('apply_cell_keep:path', 'refusing to overwrite the EXTRACT file');
end
m = load(extract_mat);
C = size(m.spatial_weights, 3);
R = readtable(review_csv, 'TextType', 'char');
if ~all(ismember({'id', 'keep'}, R.Properties.VariableNames))
    error('apply_cell_keep:csv', '%s needs columns id and keep', review_csv);
end
if height(R) ~= C || ~isequal(R.id(:), (1:C)')
    error('apply_cell_keep:csv', ['%s has %d rows (ids %d..%d) but the EXTRACT file has %d ' ...
          'components: was it made from this file?'], review_csv, height(R), min(R.id), max(R.id), C);
end
k = R.keep;
if iscell(k), k = str2double(k); end
if any(~ismember(k, [0 1]))
    error('apply_cell_keep:csv', 'keep must be 0 or 1 in every row (rows %s)', ...
          mat2str(find(~ismember(k, [0 1]))'));
end
keep = logical(k);
S = full(m.spatial_weights);
m.spatial_weights = S(:, :, keep);
m.temporal_raw = m.temporal_raw(:, keep);
m.temporal_extract = m.temporal_extract(:, keep);
f = fieldnames(m);
for i = 1:numel(f)
    v = m.(f{i});
    if (isnumeric(v) || islogical(v)) && isvector(v) && numel(v) == C && C > 1 && ...
            ~ismember(f{i}, {'fps', 'day'})
        m.(f{i}) = v(keep);
    end
end
if isfield(m, 'cell_qc') && isstruct(m.cell_qc)
    g = fieldnames(m.cell_qc);
    for i = 1:numel(g)
        v = m.cell_qc.(g{i});
        if size(v, 1) == C, m.cell_qc.(g{i}) = v(keep, :); end
    end
end
m.kept_ids = find(keep);
m.review = R;
m.review_csv = review_csv;
m.source_extract_mat = extract_mat;
w = whos('m');
fmt = '-v7';
if w.bytes > 1.9e9, fmt = '-v7.3'; end
save(out_mat, '-struct', 'm', fmt);
fprintf('apply_cell_keep: kept %d of %d components -> %s\n', nnz(keep), C, out_mat);
out = struct('file', out_mat, 'n_kept', nnz(keep), 'n_total', C, 'kept_ids', m.kept_ids);
end
