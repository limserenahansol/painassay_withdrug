function out = extract_1p_seeded(cfg)
%EXTRACT_1P_SEEDED  The cell-detection recipe settled on the 5941 SBI test (2026-10-20):
%   EXTRACT alone missed dim somata in the bright middle of the lens (it found them but its
%   own quality thresholds removed them), so:
%     1. EXTRACT as tuned by the sweep                              -> <out_dir>\pass1
%     2. whole-movie projections per recording (window_s, binned 4x) -> <out_dir>\projections
%        (Cn, PNR, max dF/F, mean...; the max over the recordings)
%     3. every bright Cn x PNR spot not already covered becomes a seed  (seed_footprints)
%     4. EXTRACT again from pass 1 + seeds as starting footprints, NO automatic elimination
%        (2 refinement iterations, permissive thresholds)          -> <out_dir>\<tag>_extract.mat
%   extract_cell_review then sorts the candidates (rules fitted to the user's review of 5941:
%   91 / 91 agreement); cell_curation_gui is the human check.
%
%   out = extract_1p_seeded(cfg)
%   cfg: day_json (or h5 + frames_csv + mouse_id + day), out_dir, and optionally
%     pass1       struct of extract_1p fields (default radius 10, T_min_snr 5, cellfind 1, hp 8)
%     seed        struct (radius 13, k_sd 3)
%     loose       struct of extract_1p fields for pass 2 (radius 13, hp 8, T_min_snr 3,
%                 cellfind 0, eccent 10)
%     window_s    ([180 780]) projection window within each recording (the stimulus blocks)

def1 = struct('avg_cell_radius', 10, 'T_min_snr', 5, 'cellfind_min_snr', 1, 'spatial_highpass_cutoff', 8);
p1 = merge(def1, getopt(cfg, 'pass1', struct()));
sd = merge(struct('radius', 13, 'k_sd', 3), getopt(cfg, 'seed', struct()));
lo = merge(struct('avg_cell_radius', 13, 'T_min_snr', 3, 'cellfind_min_snr', 0, ...
                  'spatial_highpass_cutoff', 8, 'eccent_thresh', 10), getopt(cfg, 'loose', struct()));
win = getopt(cfg, 'window_s', [180 780]);
od = cfg.out_dir;
if ~isfolder(od), mkdir(od); end
base = rmfield_if(cfg, {'pass1', 'seed', 'loose', 'window_s', 'out_dir', 'pass1_mat'});
if isfield(base, 'day_json') && ~isempty(base.day_json)
    js = jsondecode(fileread(base.day_json));
    jd = fileparts(base.day_json);
    h5 = relocate(js.files.h5, jd);
    fcsv = relocate(js.files.frames_csv, jd);
    tag = sprintf('%s_d%d', js.mouse_id, js.day);
else
    h5 = base.h5; fcsv = base.frames_csv;
    tag = sprintf('%s_d%d', base.mouse_id, base.day);
end

% 1. pass 1 (cfg.pass1_mat: re-use an earlier pass-1 result of the same movie)
p1mat = getopt(cfg, 'pass1_mat', '');
if ~isempty(p1mat) && isfile(p1mat)
    e1 = load(p1mat, 'spatial_weights');
    r1 = struct('files', struct('mat', p1mat), 'n_cells', size(e1.spatial_weights, 3));
    fprintf('extract_1p_seeded: pass 1 re-used (%d cells) from %s\n', r1.n_cells, p1mat);
else
    c1 = merge(base, p1);
    c1.out_dir = fullfile(od, 'pass1');
    c1.movie_mask = 'auto';
    r1 = extract_1p(c1);
end

% 2. projections, per recording, max over recordings
pd = fullfile(od, 'projections');
if ~isfolder(pd), mkdir(pd); end
ft = readtable(fcsv, 'TextType', 'char');
ids = unique(cellstr(string(ft.recording_id)), 'stable');
P = [];
for i = 1:numel(ids)
    f = fullfile(pd, sprintf('%s_projections.mat', ids{i}));
    if ~isfile(f)
        fov_projection_images(h5, fcsv, f, struct('radius', p1.avg_cell_radius, 'window_s', win, ...
                                                  'recording_id', ids{i}));
    end
    Q = load(f);
    if isempty(P)
        P = Q;
    else
        P.Cn = max(P.Cn, Q.Cn); P.PNR = max(P.PNR, Q.PNR);
        P.max_dff = max(P.max_dff, Q.max_dff); P.max_bgsub = max(P.max_bgsub, Q.max_bgsub);
        P.std_dff = max(P.std_dff, Q.std_dff); P.mean_img = (P.mean_img * (i - 1) + Q.mean_img) / i;
    end
end
proj_mat = fullfile(od, [tag '_projections.mat']);
save(proj_mat, '-struct', 'P', '-v7');

% 3. seeds
s = seed_footprints(proj_mat, r1.files.mat, fullfile(od, 'seeds'), sd);
L = load(s.file, 'S_init');

% 4. pass 2, no automatic elimination
c2 = merge(base, lo);
c2.out_dir = od;
c2.movie_mask = 'auto';
c2.extract_overrides = struct('S_init', L.S_init, 'max_iter', 2, 'thresholds', struct( ...
    'T_min_snr', 1, 'size_lower_limit', 0, 'size_upper_limit', inf, 'temporal_corrupt_thresh', inf, ...
    'spatial_corrupt_thresh', inf, 'eccent_thresh', inf, 'low_ST_index_thresh', -inf, ...
    'low_ST_corr_thresh', -inf));
r2 = extract_1p(c2);
out = struct('files', r2.files, 'projections_mat', proj_mat, 'pass1_mat', r1.files.mat, ...
             'n_pass1', r1.n_cells, 'n_seeds', s.n_seed, 'n_cells', r2.n_cells);
fprintf('extract_1p_seeded %s: pass 1 %d cells + %d seeds -> %d candidates -> %s\n', tag, r1.n_cells, ...
        s.n_seed, r2.n_cells, r2.files.mat);
end


function s = merge(a, b)
s = a;
f = fieldnames(b);
for k = 1:numel(f), s.(f{k}) = b.(f{k}); end
end


function s = rmfield_if(s, f)
s = rmfield(s, intersect(fieldnames(s), f));
end


function v = getopt(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end


function p = relocate(p, d)
if ~isfile(p)
    [~, nm, ext] = fileparts(p);
    p = fullfile(d, [nm ext]);
end
end
