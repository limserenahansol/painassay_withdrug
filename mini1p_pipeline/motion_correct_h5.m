function info = motion_correct_h5(cfg)
%MOTION_CORRECT_H5  NoRMCorre motion correction of lossless raw h5 movies, one template per day.
%
%   info = motion_correct_h5(cfg)      correct one mouse-day (normally pre then post)
%   motion_correct_h5('selftest')      synthetic test, errors (non-zero exit under -batch) on failure
%
%   Standard 1p practice: shifts are ESTIMATED on a filtered copy and APPLIED to the raw
%   values, so the output is the raw movie translated and nothing else.
%     estimation copy  3x3 median (hot/defective pixels must not pin the registration),
%                      then the NoRMCorre/CaImAn 1p high-pass kernel (Gaussian of sigma
%                      filter_sigma_px minus its mean over its central disk), then, in rigid
%                      mode, 2x2 binning (estimate_bin): the cross-correlation is 4x cheaper
%                      and us_fac 50 still resolves 0.04 full-resolution px
%     template         from the template recording (cfg.template_from, normally 'pre'):
%                      template_blocks contiguous blocks of template_block_frames frames spread
%                      over the recording, registered to their median, averaged, registered
%                      again, and the best-correlated 80 % averaged
%     shifts           normcorre_batch on the estimation copy against that ONE fixed template
%                      (upd_template = false) for EVERY recording, so pre and post end up on
%                      one pixel grid and cells keep their positions
%     output           apply_shifts (NoRMCorre, FFT sub-pixel translation, edges filled by
%                      copying the border, values clamped to the frame's own range) on the
%                      RAW full-resolution frames, with the rigid shifts scaled by
%                      estimate_bin and the global phase term set to 0 (pure translation)
%   Never the whole movie in RAM: chunk_frames frames at a time.
%
%   Required cfg fields
%     inputs        cellstr: raw h5 files of isxd_to_h5 (<name>_raw.h5), one per recording
%   Optional (default)
%     frames_csv    cellstr (<name>_raw_frames.csv next to each h5)
%     names         cellstr output stems (input stem without '_raw')
%     tag           day tag for the template, log and day QC files (names{1})
%     out_dir       (folder of inputs{1})
%     template_from (1)      index into inputs
%     mode          ('rigid') | 'piecewise' (NoRMCorre patches; estimated at full resolution)
%     max_shift_px  (40)     full-resolution px. The 2024 'post' file needs ~31 px.
%     us_fac        (50)
%     filter_sigma_px (7)    full-resolution px (about half the 13 px cell radius)
%     median_px     (3)      0 = off
%     estimate_bin  (2)      rigid only; piecewise always estimates at full resolution
%     grid_size_px  ([128 128]), overlap_px ([32 32]), max_dev_px ([3 3])   piecewise only
%     template_blocks (10), template_block_frames (50)
%     chunk_frames  ([])     [] = from free RAM (multiple of the h5 chunk depth, 50..1000)
%     output_class  ('uint16') | 'single'. uint16 = rounded and clamped (rounding error
%                            <= 0.5 count against ~37 counts of shot noise per pixel)
%     deflate       (1)      lossless gzip level of /mov (+ shuffle); 0 = none. Measured on
%                            the 2024 nVista3 movie: level 1 = 1.77x smaller, read 380 MB/s
%                            (2026-10-19). uint16 + deflate 1 = 1.16 MB/frame vs 4.1 single
%     crop_rect     ([])     crop of the OUTPUT movie, applied after the shift: [] = whole frame,
%                            [r0 r1 c0 c1] (1-based, inclusive, raw grid) or 'auto' = the lit
%                            lens area of the template's raw-space mean (Otsu, largest region)
%                            + crop_margin_px. Shifts, template and verification use the whole
%                            frame; the crop is the same for every recording of the day and is
%                            stored in /mov attribute crop_rect and in the JSON
%                            (<tag>_mc_crop_qc.png shows it). Saves disk and EXTRACT time.
%     crop_margin_px (40)
%     workers       (8)      parallel pool for NoRMCorre's parfor: an existing pool is used,
%                            else a thread pool of this size is started (and closed after)
%     use_gpu       ('auto') GPU for the estimation-copy filtering only
%     overwrite     (false)  false re-uses outputs of an identical earlier run
%     toolbox_dir   folder of mini2p_toolbox_paths.m (EXTRACT + NoRMCorre, strips ActSort)
%
%   Outputs per recording (<name> from cfg.names)
%     <name>_mc.h5          /mov [H W N] single (or uint16): row k of the mc movie is row k
%                           of the raw h5, so the raw frame table's h5_row is valid for both;
%                           /mean_image, /max_image, /frame_mean (single)
%     <name>_mc_shifts.csv  recording_id, isx_frame, h5_row, x_shift_px, y_shift_px,
%                           corr_before, corr_after (and patch_dev_max_px in piecewise mode).
%                           Shifts are what NoRMCorre APPLIED to the frame (full-resolution
%                           px, x = columns, y = rows); the tissue moved by minus that.
%                           corr_* = Pearson r of the estimation copy with the template, inner
%                           region, before and after registration.
%     <name>_mc_qc.png      mean image before/after, shift and correlation traces
%   Per day
%     <tag>_mc_template.mat template (estimation space), raw-space template mean, frames used
%     <tag>_mc_qc.png       template; every recording's corrected mean against the template
%     <tag>_mc.json         parameters, shift and correlation statistics, sizes, runtimes,
%                           peak RAM

if nargin == 1 && (ischar(cfg) || isstring(cfg)) && strcmpi(cfg, 'selftest')
    info = run_selftest();
    return;
end

t_all = tic;
cfg = apply_defaults(cfg);
addpath(cfg.toolbox_dir);
tb = mini2p_toolbox_paths();
files = day_files(cfg);

prev = reusable(files, cfg);
if ~isempty(prev)
    info = prev;
    info.reused = true;
    fprintf('motion_correct_h5: %s re-used\n', cfg.tag);
    return;
end

info = struct('module', 'motion_correct_h5', 'status', 'running', ...
    'created', char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss')), ...
    'matlab_version', version, 'tag', cfg.tag, 'params', param_record(cfg), 'files', files, ...
    'toolboxes', tb, 'reused', false);
info.warnings = {};
info.runtime_s = struct();
info.method = ['NoRMCorre normcorre_batch on a median + 1p high-pass (+ binned) copy against one ', ...
               'fixed template from the template recording; NoRMCorre apply_shifts on the raw values'];
pool_cleanup = start_pool(cfg.workers); %#ok<NASGU>
try
    info = run_mc(cfg, files, info);
    info.status = 'ok';
catch err
    info.status = 'failed';
    info.error = err.message;
    info.error_report = getReport(err, 'extended', 'hyperlinks', 'off');
    info.runtime_s.total = toc(t_all);
    write_json(files.json, info);
    rethrow(err);
end
info.runtime_s.total = toc(t_all);
info.peak_ram_gb = peak_ram_gb();
write_json(files.json, info);
fprintf('motion_correct_h5: %s done in %.1f s\n', cfg.tag, info.runtime_s.total);
end


% =====================================================================================
function info = run_mc(cfg, files, info)
n_rec = numel(cfg.inputs);
R = cell(1, n_rec);
for r = 1:n_rec
    R{r} = describe_input(cfg.inputs{r}, cfg.frames_csv{r}, cfg.names{r}, cfg.out_dir);
end
H = R{1}.H;
W = R{1}.W;
for r = 2:n_rec
    if R{r}.H ~= H || R{r}.W ~= W
        error('motion_correct_h5:size', '%s is %dx%d, %s is %dx%d: one pixel grid is required', ...
              R{1}.h5, H, W, R{r}.h5, R{r}.H, R{r}.W);
    end
end
P = estimation_setup(cfg, H, W);
info.estimation = rmfield(P, intersect(fieldnames(P), {'otf', 'opt_est', 'opt_apply'}));
chunk = cfg.chunk_frames;
if isempty(chunk)
    chunk = auto_chunk(H, W, R{1}.chunk(3));
end
info.chunk_frames = chunk;
fprintf('motion_correct_h5: %d recording(s) %dx%d, %s, estimation %dx%d (bin %d), chunk %d frames, GPU %d\n', ...
        n_rec, H, W, cfg.mode, P.He, P.We, P.bin, chunk, P.use_gpu);

% ---- 1. template ---------------------------------------------------------------
t = tic;
T = build_template(R{cfg.template_from}, P, cfg);
info.template = rmfield(T, intersect(fieldnames(T), {'template', 'raw_mean'}));
info.template.recording_id = R{cfg.template_from}.recording_id;
template = T.template;
save(files.template_mat, '-struct', 'T', '-v7.3');
info.runtime_s.template = toc(t);
fprintf('  template from %s: %d frames, median r %.3f -> kept %d\n', info.template.recording_id, ...
        T.n_frames, T.corr_median, T.n_kept);

% ---- 1b. output crop (the lens), one rectangle for the whole day ----------------------
crop = choose_crop(cfg, T, H, W, files);
info.crop_rect = crop;
info.crop_fraction_kept = (crop(2) - crop(1) + 1) * (crop(4) - crop(3) + 1) / (H * W);
fprintf('  output crop rows %d-%d, cols %d-%d (%.0f%% of the frame)\n', crop, 100 * info.crop_fraction_kept);

% ---- 2. every recording against that template --------------------------------------
for r = 1:n_rec
    t = tic;
    R{r} = correct_one(R{r}, template, P, cfg, chunk, crop);
    R{r}.runtime_s = toc(t);
    R{r}.ms_per_frame = 1000 * R{r}.runtime_s / R{r}.N;
    fprintf('  %s: %d frames in %.0f s (%.1f ms/frame); |shift| median %.2f, max %.2f px; r %.3f -> %.3f\n', ...
            R{r}.recording_id, R{r}.N, R{r}.runtime_s, R{r}.ms_per_frame, R{r}.stats.abs_shift_median, ...
            R{r}.stats.abs_shift_max, R{r}.stats.corr_before_median, R{r}.stats.corr_after_median);
    try
        plot_recording_qc(R{r}, P);
    catch e
        info.warnings{end+1} = sprintf('%s: QC figure failed: %s', R{r}.recording_id, e.message);
    end
    if R{r}.stats.abs_shift_max > 0.9 * P.max_shift_px
        info.warnings{end+1} = sprintf(['%s: a shift of %.1f px reaches max_shift_px (%g): raise it or ', ...
            'check the recording'], R{r}.recording_id, R{r}.stats.abs_shift_max, P.max_shift_px);
    end
    if R{r}.stats.corr_after_median < 0.5
        info.warnings{end+1} = sprintf(['%s: median correlation with the day template after registration ', ...
            'is %.2f (different focal plane or field of view?)'], R{r}.recording_id, R{r}.stats.corr_after_median);
    end
end
try
    plot_day_qc(files.qc_png, R, T, P, cfg.tag);
catch e
    info.warnings{end+1} = ['day QC figure failed: ' e.message];
end
info.recordings = cellfun(@(x) rmfield(x, intersect(fieldnames(x), {'ft', 'mean_before', 'mean_after', ...
                  'shifts', 'corr_before', 'corr_after'})), R, 'UniformOutput', false);
info.peak_ram_gb_during_mc = peak_ram_gb();
end


% =====================================================================================
function rec = describe_input(h5, frames_csv, name, out_dir)
hi = h5info(h5, '/mov');
sz = hi.Dataspace.Size;
if numel(sz) ~= 3
    error('motion_correct_h5:input', '%s:/mov must be H x W x T', h5);
end
rec.h5 = h5;
rec.frames_csv = frames_csv;
rec.name = name;
rec.H = sz(1);
rec.W = sz(2);
rec.N = sz(3);
rec.in_class = hi.Datatype.Type;
rec.chunk = [sz(1) sz(2) 50];
if ~isempty(hi.ChunkSize)
    rec.chunk = hi.ChunkSize;
end
ft = read_frame_table(frames_csv);
v = find(ft.valid == 1);
if numel(v) ~= rec.N || ~isequal(ft.h5_row(v), (1:rec.N)')
    error('motion_correct_h5:frames', '%s: %d valid rows, h5 has %d frames; h5_row must be 1..N', ...
          frames_csv, numel(v), rec.N);
end
ids = unique(ft.recording_id, 'stable');
rec.recording_id = strjoin(ids, '+');
rec.ft = ft;
rec.mc_h5 = fullfile(out_dir, [name '_mc.h5']);
rec.shifts_csv = fullfile(out_dir, [name '_mc_shifts.csv']);
rec.qc_png = fullfile(out_dir, [name '_mc_qc.png']);
end


function P = estimation_setup(cfg, H, W)
P = struct();
P.mode = lower(cfg.mode);
P.bin = cfg.estimate_bin;
if strcmp(P.mode, 'piecewise')
    P.bin = 1;
end
P.max_shift_px = cfg.max_shift_px;
P.filter_sigma_px = cfg.filter_sigma_px;
P.median_px = cfg.median_px;
P.us_fac = cfg.us_fac;
P.use_gpu = resolve_gpu(cfg.use_gpu);
P.H = H;
P.W = W;
P.He = floor(H / P.bin);
P.We = floor(W / P.bin);
% NoRMCorre / CaImAn 1p high-pass kernel (demo_1p_low_RAM.m)
gSig = cfg.filter_sigma_px;
gSiz = 3 * gSig;
psf = fspecial('gaussian', round(2 * gSiz), gSig);
ind = psf(:) >= max(psf(:, 1));
psf = psf - mean(psf(ind));
psf(~ind) = 0;
P.kernel_size = size(psf, 1);
pad0 = ceil(size(psf, 1) / 2) + 1;
Hp = next_smooth(H + 2 * pad0);
Wp = next_smooth(W + 2 * pad0);
P.pad = [floor((Hp - H) / 2), Hp - H - floor((Hp - H) / 2), floor((Wp - W) / 2), Wp - W - floor((Wp - W) / 2)];
P.otf = single(psf2otf(psf, [Hp Wp]));
% inner region for the correlations: away from the copied border and the filter edge
P.margin = ceil((P.max_shift_px + pad0) / P.bin);
ms = ceil(P.max_shift_px / P.bin);
common = {'us_fac', P.us_fac, 'iter', 1, 'correct_bidir', false, 'upd_template', false, ...
          'print_msg', false, 'output_type', 'mat', 'boundary', 'copy', 'shifts_method', 'fft', ...
          'phase_flag', false, 'bin_width', 200, 'init_batch', 50};
if strcmp(P.mode, 'piecewise')
    P.grid_size_px = cfg.grid_size_px;
    P.overlap_px = cfg.overlap_px;
    P.max_dev_px = cfg.max_dev_px;
    pw = {'grid_size', cfg.grid_size_px, 'overlap_pre', cfg.overlap_px, 'overlap_post', cfg.overlap_px, ...
          'mot_uf', 4, 'max_dev', cfg.max_dev_px};
    P.opt_est = NoRMCorreSetParms('d1', H, 'd2', W, 'max_shift', ms, common{:}, pw{:});
    P.opt_apply = P.opt_est;
else
    P.opt_est = NoRMCorreSetParms('d1', P.He, 'd2', P.We, 'max_shift', ms, common{:});
    P.opt_apply = NoRMCorreSetParms('d1', H, 'd2', W, 'max_shift', P.max_shift_px, common{:});
end
end


function n = next_smooth(n)
% Smallest integer >= n whose prime factors are 2, 3, 5 or 7 (fast FFT sizes).
while true
    m = n;
    for p = [2 3 5 7]
        while mod(m, p) == 0
            m = m / p;
        end
    end
    if m == 1
        return;
    end
    n = n + 1;
end
end


function c = auto_chunk(H, W, depth)
% About 10 copies of a chunk are alive at once (raw, filtered, NoRMCorre's cell copies,
% the output); keep them under ~15 % of the free RAM.
free = free_ram_bytes();
c = floor(0.15 * free / (10 * H * W * 4));
c = max(depth, min(1000, floor(c / depth) * depth));
end


function F = estimation_copy(Y, P)
% Y: H x W x n single (raw counts). Returns He x We x n single.
[H, W, n] = size(Y);
F = zeros(P.He, P.We, n, 'single');
sub = 32;
otf = P.otf;
if P.use_gpu
    otf = gpuArray(otf);
end
for i0 = 1:sub:n
    i1 = min(n, i0 + sub - 1);
    g = Y(:, :, i0:i1);
    if P.use_gpu
        g = gpuArray(g);
    end
    if P.median_px > 1
        % symmetric edges by hand: the gpuArray medfilt2 takes no padding option
        q = floor(P.median_px / 2);
        for j = 1:size(g, 3)
            gj = medfilt2(padarray(g(:, :, j), [q q], 'symmetric'), [P.median_px P.median_px]);
            g(:, :, j) = gj(q + (1:H), q + (1:W));
        end
    end
    g = padarray(g, [P.pad(1) P.pad(3)], 'symmetric', 'pre');
    g = padarray(g, [P.pad(2) P.pad(4)], 'symmetric', 'post');
    g = real(ifft2(fft2(g) .* otf));
    g = g(P.pad(1) + (1:H), P.pad(3) + (1:W), :);
    if P.bin > 1
        b = P.bin;
        g = g(1:P.He * b, 1:P.We * b, :);
        g = reshape(sum(sum(reshape(g, b, P.He, b, P.We, []), 1), 3), P.He, P.We, []) / b^2;
    end
    F(:, :, i0:i1) = gather(g);
end
end


function [Mr, sh] = register(F, template, opt)
% normcorre_batch against a fixed template; it errors on a 1-frame movie, so a lone
% frame is registered as a pair with itself.
nthr = maxNumCompThreads;
restore = onCleanup(@() maxNumCompThreads(nthr));   % normcorre_batch sets it to 1
% NoRMCorre's parfor triggers a harmless code-analysis warning on every call
ws = warning('off', 'MATLAB:mir_warning_maybe_uninitialized_temporary');
restore_w = onCleanup(@() warning(ws));
single_frame = size(F, 3) == 1;
if single_frame
    F = cat(3, F, F);
end
[Mr, sh] = normcorre_batch(F, opt, template);
if single_frame
    Mr = Mr(:, :, 1);
    sh = sh(1);
end
end


function out = apply_raw(Y, sh, P)
% NoRMCorre apply_shifts on the raw frames: rigid shifts scaled to full resolution,
% global phase term 0 (a pure translation of the raw values).
for k = 1:numel(sh)
    sh(k).shifts = sh(k).shifts * P.bin;
    if isfield(sh, 'shifts_up') && ~isempty(sh(k).shifts_up)
        sh(k).shifts_up = sh(k).shifts_up * P.bin;
    else
        sh(k).shifts_up = sh(k).shifts;
    end
    sh(k).diff = zeros(size(sh(k).diff));
end
nthr = maxNumCompThreads;
restore = onCleanup(@() maxNumCompThreads(nthr));
single_frame = size(Y, 3) == 1;
if single_frame
    Y = cat(3, Y, Y);
    sh = [sh; sh];
end
out = apply_shifts(Y, sh, P.opt_apply);
if single_frame
    out = out(:, :, 1);
end
end


function [dx, dy, dev] = shift_xy(sh, P)
n = numel(sh);
dx = zeros(n, 1);
dy = zeros(n, 1);
dev = zeros(n, 1);
for k = 1:n
    s = sh(k).shifts;
    sy = s(:, :, :, 1);
    sx = s(:, :, :, 2);
    dy(k) = mean(sy(:)) * P.bin;
    dx(k) = mean(sx(:)) * P.bin;
    dev(k) = max(hypot(sx(:) - mean(sx(:)), sy(:) - mean(sy(:)))) * P.bin;
end
end


function c = inner_corr(F, template, m)
% Pearson r of every frame with the template on the inner region.
[h, w, n] = size(F);
rr = (m + 1):(h - m);
cc = (m + 1):(w - m);
if numel(rr) < 8 || numel(cc) < 8
    rr = 1:h;
    cc = 1:w;
end
t = double(template(rr, cc));
t = t(:) - mean(t(:));
t = t / max(norm(t), eps);
X = reshape(double(F(rr, cc, :)), [], n);
X = X - mean(X, 1);
c = (t' * X) ./ max(sqrt(sum(X.^2, 1)), eps);
c = c(:);
end


function T = build_template(rec, P, cfg)
% Contiguous blocks (whole h5 chunks) spread over the recording; random single frames
% would read ~chunk-depth times more data.
nb = cfg.template_blocks;
L = cfg.template_block_frames;
N = rec.N;
if nb * L >= N
    starts = 1;
    L = N;
else
    starts = round(linspace(1, N - L + 1, nb));
end
rows = [];
F = [];
Yraw_sum = zeros(rec.H, rec.W);
Ylist = cell(1, numel(starts));
for b = 1:numel(starts)
    Y = single(h5read(rec.h5, '/mov', [1 1 starts(b)], [rec.H rec.W L]));
    F = cat(3, F, estimation_copy(Y, P));
    Ylist{b} = Y;
    rows = [rows, starts(b):starts(b) + L - 1]; %#ok<AGROW>
end
t0 = median(F, 3);
[R1, ~] = register(F, t0, P.opt_est);
t1 = mean(R1, 3);
[R2, sh2] = register(F, t1, P.opt_est);
c = inner_corr(R2, t1, P.margin);
keep = c >= prctile(c, 20);
template = mean(R2(:, :, keep), 3);
% raw-space mean of the same frames, translated: what the template looks like in counts
j0 = 0;
nk = 0;
for b = 1:numel(Ylist)
    n = size(Ylist{b}, 3);
    kk = keep(j0 + (1:n));
    if any(kk)
        A = apply_raw(Ylist{b}(:, :, kk), sh2(j0 + find(kk)), P);
        Yraw_sum = Yraw_sum + sum(double(A), 3);
        nk = nk + nnz(kk);
    end
    j0 = j0 + n;
end
[dx, dy] = shift_xy(sh2, P);
T = struct('template', single(template), 'raw_mean', single(Yraw_sum / max(1, nk)), ...
    'h5_rows', rows, 'n_frames', numel(rows), 'n_kept', nnz(keep), 'corr_median', median(c), ...
    'corr_min_kept', min(c(keep)), 'block_starts', starts, 'block_frames', L, ...
    'shift_x_range_px', [min(dx) max(dx)], 'shift_y_range_px', [min(dy) max(dy)], ...
    'source_h5', rec.h5, 'estimation_bin', P.bin, 'filter_sigma_px', P.filter_sigma_px, ...
    'median_px', P.median_px);
end


function rec = correct_one(rec, template, P, cfg, chunk, crop)
H = rec.H;
W = rec.W;
N = rec.N;
if nargin < 6 || isempty(crop), crop = [1 H 1 W]; end
rr = crop(1):crop(2);
cc = crop(3):crop(4);
Hc = numel(rr);
Wc = numel(cc);
tmp = [rec.mc_h5 '.partial'];
delete_if_exists(tmp);
cls = cfg.output_class;
h5args = {'Datatype', cls, 'ChunkSize', min([Hc Wc rec.chunk(3)], [Hc Wc N])};
if cfg.deflate > 0
    h5args = [h5args, {'Deflate', cfg.deflate, 'Shuffle', true}];
end
h5create(tmp, '/mov', [Hc Wc N], h5args{:});
rec.output_class = cls;
dx = zeros(N, 1);
dy = zeros(N, 1);
dev = zeros(N, 1);
cb = zeros(N, 1);
ca = zeros(N, 1);
sum_before = zeros(H, W);
sum_after = zeros(H, W);
max_after = -inf(Hc, Wc, 'single');
frame_mean = zeros(1, N, 'single');
n_clamped = 0;
t0 = tic;
for i0 = 1:chunk:N
    nb = min(chunk, N - i0 + 1);
    rows = i0:i0 + nb - 1;
    Y = single(h5read(rec.h5, '/mov', [1 1 i0], [H W nb]));
    F = estimation_copy(Y, P);
    [Mr, sh] = register(F, template, P.opt_est);
    cb(rows) = inner_corr(F, template, P.margin);
    ca(rows) = inner_corr(Mr, template, P.margin);
    [dx(rows), dy(rows), dev(rows)] = shift_xy(sh, P);
    A = apply_raw(Y, sh, P);
    if strcmp(cls, 'uint16')
        n_clamped = n_clamped + nnz(A < 0 | A > 65535);
        A = uint16(round(A));
    end
    if ~all(isfinite(A(:)))
        error('motion_correct_h5:nan', '%s: non-finite values after apply_shifts (rows %d..%d)', ...
              rec.recording_id, rows(1), rows(end));
    end
    sum_before = sum_before + sum(double(Y), 3);
    sum_after = sum_after + sum(double(A), 3);
    A = A(rr, cc, :);
    h5write(tmp, '/mov', A, [1 1 i0], [Hc Wc nb]);
    max_after = max(max_after, single(max(A, [], 3)));
    frame_mean(rows) = squeeze(mean(single(A), [1 2]))';
    fprintf('    %s: %d / %d frames (%.0f s)\n', rec.recording_id, rows(end), N, toc(t0));
end
h5create(tmp, '/mean_image', [Hc Wc], 'Datatype', 'single');
h5write(tmp, '/mean_image', single(sum_after(rr, cc) / N));
h5create(tmp, '/max_image', [Hc Wc], 'Datatype', 'single');
h5write(tmp, '/max_image', max_after);
h5create(tmp, '/frame_mean', [1 N], 'Datatype', 'single');
h5write(tmp, '/frame_mean', frame_mean);
h5writeatt(tmp, '/mov', 'source_raw_h5', rec.h5);
h5writeatt(tmp, '/mov', 'frame_table', rec.frames_csv);
h5writeatt(tmp, '/mov', 'rows', 'row k of this movie is row k of source_raw_h5 (same h5_row)');
h5writeatt(tmp, '/mov', 'crop_rect', double(crop));
h5writeatt(tmp, '/mov', 'crop_note', 'pixel (i, j) of /mov is raw pixel (crop_rect(1) + i - 1, crop_rect(3) + j - 1), after the shift');
h5writeatt(tmp, '/mov', 'method', ['raw values translated by NoRMCorre apply_shifts (FFT, border ', ...
           'copied, clamped to the frame range); shifts estimated on a filtered copy']);
delete_if_exists(rec.mc_h5);
movefile(tmp, rec.mc_h5);
rec.mc_bytes = dir(rec.mc_h5).bytes;
rec.mc_bytes_per_frame = rec.mc_bytes / N;
rec.n_uint16_clamped = n_clamped;
rec.mean_before = single(sum_before / N);
rec.mean_after = single(sum_after / N);        % whole frame (QC, template comparison)
rec.crop_rect = crop;
rec.shifts = [dx dy];
rec.corr_before = cb;
rec.corr_after = ca;

% shifts table: one row per h5 row, joined to the raw frame table
ft = rec.ft;
v = find(ft.valid == 1);
S = struct('recording_id', {ft.recording_id(v)}, 'isx_frame', ft.isx_frame(v), 'h5_row', ft.h5_row(v), ...
           'x_shift_px', dx, 'y_shift_px', dy, 'corr_before', cb, 'corr_after', ca);
fmt = struct('recording_id', '%s', 'isx_frame', '%d', 'h5_row', '%d', 'x_shift_px', '%.4f', ...
             'y_shift_px', '%.4f', 'corr_before', '%.5f', 'corr_after', '%.5f');
if strcmp(P.mode, 'piecewise')
    S.patch_dev_max_px = dev;
    fmt.patch_dev_max_px = '%.4f';
end
write_csv(rec.shifts_csv, S, fmt);

d = hypot(dx, dy);
rec.stats = struct('x_median', median(dx), 'y_median', median(dy), 'x_range', [min(dx) max(dx)], ...
    'y_range', [min(dy) max(dy)], 'abs_shift_median', median(d), 'abs_shift_p95', prctile(d, 95), ...
    'abs_shift_max', max(d), 'frame_to_frame_p95', prctile(hypot(diff(dx), diff(dy)), 95), ...
    'frame_to_frame_max', max([0; hypot(diff(dx), diff(dy))]), ...
    'corr_before_median', median(cb), 'corr_after_median', median(ca), ...
    'corr_after_p05', prctile(ca, 5), 'frac_corr_improved', mean(ca >= cb), ...
    'mean_image_corr_with_template', NaN);
if strcmp(P.mode, 'piecewise')
    rec.stats.patch_dev_max_p95 = prctile(dev, 95);
end
me = estimation_copy(rec.mean_after, P);
rec.stats.mean_image_corr_with_template = inner_corr(me, template, P.margin);
end


% =====================================================================================
function plot_recording_qc(rec, P)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1600 1000]);
lim = prctile(double(rec.mean_after(:)), [0.5 99.8]);
ax = axes(fig, 'Position', [0.03 0.42 0.45 0.53]);
imagesc(ax, rec.mean_before, lim + [0 eps]); axis(ax, 'image', 'off'); colormap(ax, gray);
title(ax, 'mean image, raw (before)', 'FontSize', 10);
ax = axes(fig, 'Position', [0.52 0.42 0.45 0.53]);
imagesc(ax, rec.mean_after, lim + [0 eps]); axis(ax, 'image', 'off'); colormap(ax, gray);
title(ax, 'mean image, motion corrected (after)', 'FontSize', 10);
t = (1:rec.N)';
ax = axes(fig, 'Position', [0.05 0.22 0.92 0.15]);
plot(ax, t, rec.shifts(:, 1), 'Color', [0 0.35 0.75]); hold(ax, 'on');
plot(ax, t, rec.shifts(:, 2), 'Color', [0.85 0.35 0.1]);
ylabel(ax, 'shift (px)'); legend(ax, {'x (columns)', 'y (rows)'}, 'Location', 'best');
xlim(ax, [1 max(2, rec.N)]); grid(ax, 'on');
ax = axes(fig, 'Position', [0.05 0.04 0.92 0.14]);
plot(ax, t, rec.corr_before, 'Color', [0.6 0.6 0.6]); hold(ax, 'on');
plot(ax, t, rec.corr_after, 'Color', [0 0.5 0.2]);
ylabel(ax, 'r with template'); xlabel(ax, 'h5 row (valid frame)');
legend(ax, {'before', 'after'}, 'Location', 'best'); xlim(ax, [1 max(2, rec.N)]); grid(ax, 'on');
s = rec.stats;
annotation(fig, 'textbox', [0 0.955 1 0.045], 'String', sprintf(['%s (%s): %d frames, |shift| median %.2f, ', ...
    'p95 %.2f, max %.2f px; r median %.3f -> %.3f'], rec.recording_id, P.mode, rec.N, s.abs_shift_median, ...
    s.abs_shift_p95, s.abs_shift_max, s.corr_before_median, s.corr_after_median), ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 12, 'Interpreter', 'none');
exportgraphics(fig, rec.qc_png, 'Resolution', 110);
close(fig);
end


function crop = choose_crop(cfg, T, H, W, files)
% [r0 r1 c0 c1]: whole frame, a given rectangle, or the lit lens area (+ margin) of the
% template's raw-space mean
c = cfg.crop_rect;
if isempty(c)
    crop = [1 H 1 W];
    return;
end
if isnumeric(c)
    crop = round(double(c(:)'));
else
    m = double(T.raw_mean);
    m = imgaussfilt(m, 8);
    g = (m - prctile(m(:), 1)) / max(prctile(m(:), 99.5) - prctile(m(:), 1), eps);
    g = min(max(g, 0), 1);
    bw = g > graythresh(g);
    L = bwlabel(bw);
    st = regionprops(L, 'Area');
    [~, k] = max([st.Area]);
    [ys, xs] = find(L == k);
    mg = cfg.crop_margin_px;
    crop = [min(ys) - mg, max(ys) + mg, min(xs) - mg, max(xs) + mg];
end
crop = [max(1, crop(1)), min(H, crop(2)), max(1, crop(3)), min(W, crop(4))];
if crop(2) - crop(1) < 63 || crop(4) - crop(3) < 63
    error('motion_correct_h5:crop', 'crop %s is too small', mat2str(crop));
end
try
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1000 700]);
    ax = axes(fig);
    imagesc(ax, T.raw_mean); axis(ax, 'image'); colormap(ax, gray); hold(ax, 'on');
    rectangle(ax, 'Position', [crop(3), crop(1), crop(4) - crop(3), crop(2) - crop(1)], ...
              'EdgeColor', [1 0.2 0.2], 'LineWidth', 2);
    title(ax, sprintf('%s: output crop rows %d-%d, cols %d-%d = %.0f%% of the %dx%d frame (red)', cfg.tag, ...
          crop, 100 * (crop(2) - crop(1) + 1) * (crop(4) - crop(3) + 1) / (H * W), H, W), ...
          'Interpreter', 'none', 'FontSize', 10);
    exportgraphics(fig, strrep(files.qc_png, '_mc_qc.png', '_mc_crop_qc.png'), 'Resolution', 100);
    close(fig);
catch
end
end


function plot_day_qc(png, R, T, P, tag)
n = numel(R);
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 520 * (n + 1) 520]);
w = 1 / (n + 1);
ax = axes(fig, 'Position', [0.005 0.05 w - 0.01 0.82]);
imagesc(ax, T.template); axis(ax, 'image', 'off'); colormap(ax, gray);
title(ax, sprintf('template (estimation copy, bin %d)', P.bin), 'FontSize', 9);
tn = mat2gray(T.template);
for r = 1:n
    me = estimation_copy(R{r}.mean_after, P);
    ax = axes(fig, 'Position', [r * w + 0.005 0.05 w - 0.01 0.82]);
    rgb = cat(3, mat2gray(me), tn, mat2gray(me));
    image(ax, rgb); axis(ax, 'image', 'off');
    title(ax, sprintf('%s after MC (magenta) vs template (green), r = %.3f', R{r}.recording_id, ...
          R{r}.stats.mean_image_corr_with_template), 'FontSize', 9, 'Interpreter', 'none');
end
annotation(fig, 'textbox', [0 0.9 1 0.09], 'String', sprintf('%s: one template for every recording', tag), ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 12, 'Interpreter', 'none');
exportgraphics(fig, png, 'Resolution', 110);
close(fig);
end


% =====================================================================================
function cfg = apply_defaults(cfg)
if ~isstruct(cfg) || ~isfield(cfg, 'inputs') || isempty(cfg.inputs)
    error('motion_correct_h5:cfg', 'cfg.inputs (raw h5 files) is required');
end
cfg.inputs = cellstr(cfg.inputs);
n = numel(cfg.inputs);
for r = 1:n
    if ~isfile(cfg.inputs{r})
        error('motion_correct_h5:cfg', 'raw h5 not found: %s', cfg.inputs{r});
    end
end
[d1, s1] = fileparts(cfg.inputs{1});
def = struct('frames_csv', {{}}, 'names', {{}}, 'tag', '', 'out_dir', d1, 'template_from', 1, ...
    'mode', 'rigid', 'max_shift_px', 40, 'us_fac', 50, 'filter_sigma_px', 7, 'median_px', 3, ...
    'estimate_bin', 2, 'grid_size_px', [128 128], 'overlap_px', [32 32], 'max_dev_px', [3 3], ...
    'template_blocks', 10, 'template_block_frames', 50, 'chunk_frames', [], 'output_class', 'uint16', ...
    'deflate', 1, 'crop_rect', [], 'crop_margin_px', 40, ...
    'workers', 8, 'use_gpu', 'auto', 'overwrite', false, ...
    'toolbox_dir', fullfile(getenv('USERPROFILE'), 'OneDrive', 'Desktop', 'cursor', 'pain_2plane_split'));
f = fieldnames(def);
for k = 1:numel(f)
    if ~isfield(cfg, f{k}) || (isempty(cfg.(f{k})) && ~strcmp(f{k}, 'chunk_frames'))
        cfg.(f{k}) = def.(f{k});
    end
end
if isempty(cfg.frames_csv)
    cfg.frames_csv = cellfun(@(p) [p(1:end - 3) '_frames.csv'], cfg.inputs, 'UniformOutput', false);
end
cfg.frames_csv = cellstr(cfg.frames_csv);
if isempty(cfg.names)
    cfg.names = cell(1, n);
    for r = 1:n
        [~, s] = fileparts(cfg.inputs{r});
        cfg.names{r} = regexprep(s, '_raw$', '');
    end
end
cfg.names = cellstr(cfg.names);
if isempty(cfg.tag)
    cfg.tag = regexprep(s1, '_raw$', '');
end
cfg.tag = char(cfg.tag);
cfg.out_dir = char(cfg.out_dir);
if ~isfolder(cfg.out_dir)
    mkdir(cfg.out_dir);
end
if numel(cfg.frames_csv) ~= n || numel(cfg.names) ~= n
    error('motion_correct_h5:cfg', 'frames_csv and names need one entry per input');
end
for r = 1:n
    if ~isfile(cfg.frames_csv{r})
        error('motion_correct_h5:cfg', 'frame table not found: %s', cfg.frames_csv{r});
    end
end
if cfg.template_from < 1 || cfg.template_from > n
    error('motion_correct_h5:cfg', 'template_from must index cfg.inputs');
end
if ~ismember(lower(cfg.mode), {'rigid', 'piecewise'})
    error('motion_correct_h5:cfg', 'mode must be rigid or piecewise');
end
if ~ismember(cfg.output_class, {'single', 'uint16'})
    error('motion_correct_h5:cfg', 'output_class must be single or uint16');
end
if cfg.estimate_bin < 1 || cfg.estimate_bin ~= round(cfg.estimate_bin)
    error('motion_correct_h5:cfg', 'estimate_bin must be a positive integer');
end
if ~isfile(fullfile(cfg.toolbox_dir, 'mini2p_toolbox_paths.m'))
    error('motion_correct_h5:cfg', 'mini2p_toolbox_paths.m not found in %s', cfg.toolbox_dir);
end
end


function p = param_record(cfg)
p = rmfield(cfg, {'toolbox_dir', 'overwrite', 'workers'});
end


function files = day_files(cfg)
files = struct('template_mat', fullfile(cfg.out_dir, [cfg.tag '_mc_template.mat']), ...
               'qc_png', fullfile(cfg.out_dir, [cfg.tag '_mc_qc.png']), ...
               'json', fullfile(cfg.out_dir, [cfg.tag '_mc.json']));
files.mc_h5 = cellfun(@(nm) fullfile(cfg.out_dir, [nm '_mc.h5']), cfg.names, 'UniformOutput', false);
files.shifts_csv = cellfun(@(nm) fullfile(cfg.out_dir, [nm '_mc_shifts.csv']), cfg.names, 'UniformOutput', false);
end


function prev = reusable(files, cfg)
prev = [];
if cfg.overwrite || ~isfile(files.json)
    return;
end
try
    js = jsondecode(fileread(files.json));
    same = strcmp(jsonencode(js.params), jsonencode(jsondecode(jsonencode(param_record(cfg)))));
    ok = strcmp(js.status, 'ok') && same && all(cellfun(@isfile, files.mc_h5)) && ...
         all(cellfun(@isfile, files.shifts_csv));
    recs = js.recordings;
    if isstruct(recs)
        recs = num2cell(recs);
    end
    for r = 1:numel(recs)
        ok = ok && dir(files.mc_h5{r}).bytes == recs{r}.mc_bytes;
    end
    if ok
        prev = js;
    end
catch
    prev = [];
end
end


function cleaner = start_pool(workers)
cleaner = [];
if workers <= 0
    return;
end
try
    if ~isempty(gcp('nocreate'))
        return;
    end
    try
        p = parpool('Threads', min(workers, maxNumCompThreads));
    catch
        p = parpool('Processes', workers);
    end
    cleaner = onCleanup(@() delete(p));
catch e
    fprintf(2, 'motion_correct_h5: no parallel pool (%s); NoRMCorre runs serially\n', e.message);
end
end


function g = resolve_gpu(spec)
if (ischar(spec) || isstring(spec)) && strcmpi(spec, 'auto')
    g = false;
    try
        g = gpuDeviceCount > 0;
    catch
    end
else
    g = logical(spec);
end
end


function b = free_ram_bytes()
b = 16e9;
try
    [~, s] = memory;
    b = s.PhysicalMemory.Available;
catch
end
end


function g = peak_ram_gb()
g = NaN;
try
    p = System.Diagnostics.Process.GetCurrentProcess();
    g = double(p.PeakWorkingSet64) / 2^30;
catch
end
end


function ft = read_frame_table(file)
opts = detectImportOptions(file, 'Delimiter', ',', 'TextType', 'char');
txt = intersect({'recording_id', 'assay', 'invalid_reason', 'checksum_sum', 'checksum_wsum'}, opts.VariableNames);
opts = setvartype(opts, txt, 'char');
opts = setvartype(opts, intersect({'isx_frame', 'valid', 'h5_row', 'tsc_us'}, opts.VariableNames), 'double');
tb = readtable(file, opts);
ft = struct('recording_id', {tb.recording_id}, 'isx_frame', tb.isx_frame, 'valid', tb.valid, ...
            'h5_row', tb.h5_row);
end


function write_csv(file, s, fmt)
cols = fieldnames(fmt);
n = numel(s.(cols{1}));
txt = strings(n, numel(cols));
for c = 1:numel(cols)
    v = s.(cols{c});
    if iscell(v)
        txt(:, c) = string(v);
    else
        x = compose(string(fmt.(cols{c})), double(v(:)));
        x(isnan(v(:))) = "";
        txt(:, c) = x;
    end
end
fid = fopen(file, 'w');
if fid < 0
    error('motion_correct_h5:io', 'cannot write %s', file);
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', strjoin(cols', ','));
if n > 0
    fprintf(fid, '%s\n', join(txt, ',', 2));
end
end


function write_json(file, s)
fid = fopen(file, 'w');
if fid < 0
    error('motion_correct_h5:io', 'cannot write %s', file);
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(s, 'PrettyPrint', true));
end


function delete_if_exists(f)
if isfile(f)
    delete(f);
end
end


% =====================================================================================
function info = run_selftest()
% Two synthetic uint16 recordings cut from one textured field (blobs, vessels, vignetting,
% static hot pixels) with known motion; 'post' sits 6 px / -4 px away
% and drifts more. Checks: both recordings land on ONE grid (estimated shift + true motion is
% the same constant for every frame of both), the output is the raw frame translated by
% the reported shift (an independent Fourier translation agrees; frame means kept), h5 rows
% pass through, re-use, uint16 output, piecewise mode.
fprintf('motion_correct_h5 selftest\n');
here = fileparts(mfilename('fullpath'));
root = fullfile(here, 'out', 'selftest_motion_correct_h5');
if isfolder(root)
    rmdir(root, 's');
end
mkdir(root);
rng(11);
H = 128;
W = 160;
m = 24;
[yy, xx] = ndgrid(1:H + 2 * m, 1:W + 2 * m);
field = 300 + 900 * exp(-((yy - (H / 2 + m)).^2 + (xx - (W / 2 + m)).^2) / (2 * 90^2));
for k = 1:70
    cy = rand * (H + 2 * m);
    cx = rand * (W + 2 * m);
    r = 3 + 4 * rand;
    field = field + 250 * rand * exp(-((yy - cy).^2 + (xx - cx).^2) / (2 * r^2));
end
for k = 1:4
    a = rand * pi;
    d = abs((xx - rand * W - m) * sin(a) - (yy - rand * H - m) * cos(a));
    field = field - 150 * exp(-d.^2 / (2 * 2.5^2));
end
hot = [5 7; 90 30; 64 150; 120 100];        % static sensor hot pixels
noise_sd = 15;
spec = {'T01_d1_pre', 120, [0 0], 1.0; 'T01_d1_post', 90, [6 -4], 3.0};
inputs = cell(1, 2);
truth = cell(1, 2);
for s = 1:2
    n = spec{s, 2};
    t = (0:n - 1)';
    mot = spec{s, 3} + spec{s, 4} * [sin(t / 9), cos(t / 13)];   % [dx dy] tissue motion, px
    mot(1:2:end, :) = round(mot(1:2:end, :));                      % odd frames integer
    M = zeros(H, W, n, 'uint16');
    for k = 1:n
        fr = subpixel_crop(field, m, mot(k, 1), mot(k, 2), H, W) + noise_sd * randn(H, W);
        for h = 1:size(hot, 1)
            fr(hot(h, 1), hot(h, 2)) = 8000;
        end
        M(:, :, k) = uint16(fr);
    end
    truth{s} = struct('motion', mot, 'raw', M);
    inputs{s} = fullfile(root, [spec{s, 1} '_raw.h5']);
    h5create(inputs{s}, '/mov', size(M), 'Datatype', 'uint16', 'ChunkSize', [64 80 10]);
    h5write(inputs{s}, '/mov', M);
    % frame table with one dropped frame (no h5 row) in the middle
    nr = n + 1;
    isx = (0:nr - 1)';
    valid = ones(nr, 1);
    valid(31) = 0;
    h5r = NaN(nr, 1);
    h5r(valid == 1) = 1:n;
    reason = repmat({''}, nr, 1);
    reason(31) = {'dropped'};
    ft = struct('recording_id', {repmat(spec(s, 1), nr, 1)}, 'isx_frame', isx, 'valid', valid, ...
                'invalid_reason', {reason}, 'h5_row', h5r);
    write_csv(fullfile(root, [spec{s, 1} '_raw_frames.csv']), ft, struct('recording_id', '%s', ...
              'isx_frame', '%d', 'valid', '%d', 'invalid_reason', '%s', 'h5_row', '%d'));
end

cfg = struct('inputs', {inputs}, 'tag', 'T01_d1', 'out_dir', root, 'max_shift_px', 16, ...
             'filter_sigma_px', 3, 'template_blocks', 3, 'template_block_frames', 20, ...
             'chunk_frames', 40, 'workers', 2, 'output_class', 'single', 'deflate', 0);
out = motion_correct_h5(cfg);
assert_true(strcmp(out.status, 'ok'), 'run ok');
c_all = [];
for s = 1:2
    sh = readtable(out.files.shifts_csv{s}, 'Delimiter', ',', 'TextType', 'char');
    n = spec{s, 2};
    assert_true(height(sh) == n && isequal(sh.h5_row', 1:n), sprintf('%s: one shift row per h5 row', spec{s, 1}));
    assert_true(~ismember(30, sh.isx_frame) && sh.isx_frame(31) == 31, 'dropped frame has no shift row; isx_frame joined');
    % NoRMCorre applies the opposite of the tissue motion: shift + motion = constant
    c_all = [c_all; [sh.x_shift_px, sh.y_shift_px] + truth{s}.motion]; %#ok<AGROW>
    hi = h5info(out.files.mc_h5{s}, '/mov');
    assert_true(isequal(hi.Dataspace.Size, [H W n]) && strcmp(hi.Datatype.Type, 'H5T_IEEE_F32LE'), ...
                'mc h5 same rows, single');
end
dev = c_all - median(c_all, 1);
fprintf('  registration: |error| median %.3f, max %.3f px over %d frames of both recordings\n', ...
        median(hypot(dev(:, 1), dev(:, 2))), max(hypot(dev(:, 1), dev(:, 2))), size(c_all, 1));
assert_true(max(hypot(dev(:, 1), dev(:, 2))) < 0.35 && median(hypot(dev(:, 1), dev(:, 2))) < 0.12, ...
            'pre and post registered to ONE template (shift + motion constant within 0.35 px)');
% The output is the raw frame translated by the reported shift and nothing else: an
% independent Fourier translation of the raw frame must give the same values inside.
s = 2;
A = h5read(out.files.mc_h5{s}, '/mov');
sh = readtable(out.files.shifts_csv{s}, 'Delimiter', ',', 'TextType', 'char');
d = [];
for k = 1:5:spec{s, 2}
    ref = fourier_shift(double(truth{s}.raw(:, :, k)), sh.y_shift_px(k), sh.x_shift_px(k));
    b = ceil(max(abs([sh.x_shift_px(k), sh.y_shift_px(k)]))) + 3;
    e = abs(double(A(b:H - b, b:W - b, k)) - ref(b:H - b, b:W - b));
    d = [d; e(:)]; %#ok<AGROW>
end
fprintf('  output vs independent Fourier translation of the raw frame: |diff| median %.4f, p99 %.3f counts\n', ...
        median(d), prctile(d, 99));
assert_true(median(d) < 0.05 && prctile(d, 99) < 1, 'output = raw frame translated by the reported shift');
rr = 20:H - 20;
cc = 20:W - 20;
fm = squeeze(mean(double(A(rr, cc, :)), [1 2]));
fr = squeeze(mean(double(truth{s}.raw(rr, cc, :)), [1 2]));
assert_true(max(abs(fm - fr) ./ fr) < 0.03, 'frame means kept (translation only)');
assert_true(isfile(fullfile(root, 'T01_d1_post_mc_qc.png')) && isfile(out.files.qc_png), 'QC PNGs');
fprintf('  output is the raw movie translated: ok\n');

% re-use, uint16 output, piecewise
out2 = motion_correct_h5(cfg);
assert_true(out2.reused, 're-use of an identical run');
cfg2 = cfg;
cfg2.inputs = inputs(1);
cfg2.tag = 'T01_pw';
cfg2.names = {'T01_pw'};
cfg2.mode = 'piecewise';
cfg2.grid_size_px = [64 80];
cfg2.overlap_px = [16 16];
cfg2.output_class = 'uint16';
cfg2.deflate = 1;
out3 = motion_correct_h5(cfg2);
sh = readtable(out3.files.shifts_csv{1}, 'Delimiter', ',', 'TextType', 'char');
c = [sh.x_shift_px, sh.y_shift_px] + truth{1}.motion;
dev = hypot(c(:, 1) - median(c(:, 1)), c(:, 2) - median(c(:, 2)));
hi = h5info(out3.files.mc_h5{1}, '/mov');
fprintf('  piecewise: |error| max %.3f px, patch deviation p95 %.3f px\n', max(dev), prctile(sh.patch_dev_max_px, 95));
assert_true(strcmp(hi.Datatype.Type, 'H5T_STD_U16LE'), 'uint16 output option');
assert_true(any(strcmp({hi.Filters.Name}, 'deflate')), 'uint16 output is gzip-compressed by default');
assert_true(max(dev) < 0.6, 'piecewise mode registers a rigid motion');
info = struct('selftest', 'ok');
fprintf('motion_correct_h5 selftest PASSED\n');
end


function out = fourier_shift(X, sy, sx)
% Content moved by +sy rows and +sx columns (the convention of NoRMCorre's shift_reconstruct).
[nr, nc] = size(X);
kr = ifftshift(-fix(nr / 2):ceil(nr / 2) - 1)';
kc = ifftshift(-fix(nc / 2):ceil(nc / 2) - 1);
out = real(ifft2(fft2(X) .* exp(-1i * 2 * pi * (sy * kr / nr + sx * kc / nc))));
end


function fr = subpixel_crop(field, m, dx, dy, H, W)
% Crop of field moved by (dx, dy): content appears at +dx columns, +dy rows.
ix = floor(dx);
iy = floor(dy);
fx = dx - ix;
fy = dy - iy;
r0 = m - iy;
c0 = m - ix;
A = field(r0 + (1:H), c0 + (1:W));
B = field(r0 + (1:H), c0 - 1 + (1:W));
C = field(r0 - 1 + (1:H), c0 + (1:W));
D = field(r0 - 1 + (1:H), c0 - 1 + (1:W));
fr = (1 - fx) * (1 - fy) * A + fx * (1 - fy) * B + (1 - fx) * fy * C + fx * fy * D;
end


function assert_true(cond, what)
if ~cond
    error('motion_correct_h5:selftest', 'selftest FAILED: %s', what);
end
end
