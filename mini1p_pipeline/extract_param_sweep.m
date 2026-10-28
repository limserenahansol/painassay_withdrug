function out = extract_param_sweep(cfg)
%EXTRACT_PARAM_SWEEP  Try many EXTRACT settings on one piece of a mouse-day movie and show,
%   side by side, which setting finds real somata (not background, vessels or fragments).
%   This is the "play around with parameters" step, made fast and repeatable: each run
%   takes about a minute instead of the hour a full-day run takes.
%
%   out = extract_param_sweep(cfg)
%   extract_param_sweep('selftest')
%
%   What it does
%     1. reads sweep_len_s of one recording (default: the first, from sweep_start_s = 180 s,
%        i.e. the first stimulus block after the 3-min baseline, when cells are active) from
%        the motion-corrected day movie and bins it bin_time x in time, exactly as the
%        full run bins it for detection (extract_1p detect_bin_time)
%     2. overview of the WHOLE field: mean image, local-correlation image (Cn) and
%        peak-to-noise image (PNR) of the spatially high-passed movie. Somata are bright
%        round spots in Cn x PNR; vessels and neuropil are not. -> <tag>_sweep_overview.png
%     3. crops crop_px x crop_px (default 400 px, ~320 um) where Cn x PNR is highest
%        ('auto'), or cfg.crop = [row0 col0 h w] / 'full'
%     4. runs extract_1p on that crop for EVERY combination of cfg.grid, then scores every
%        component with cell_qc_metrics (size, shape, pieces, SNR, skew, Ca-like transient)
%     5. writes
%          <tag>_sweep_summary.csv     one row per setting: parameters, n found, n cell /
%                                      check / not_cell, median SNR of the cells, runtime
%          <tag>_sweep_sheet.png       one panel per setting: crop's Cn x PNR image, every
%                                      component outlined (green cell, yellow check, red
%                                      not_cell), a circle of the tested radius for scale
%          run_<k>\                    extract_1p output of each setting + qc table
%          <tag>_sweep.json            everything, plus 'suggested' = most green outlines
%                                      with at most 25 % red. A suggestion: LOOK at the sheet.
%
%   Required: cfg.day_json (inscopix_day output) or cfg.h5 + cfg.frames_csv; cfg.out_dir
%   Optional (default)
%     tag            from the day JSON (mouse_d<day>), else 'sweep'
%     recording_id   ('')     '' = first recording in the frame table
%     sweep_start_s  (180)    clamped to the recording
%     sweep_len_s    (300)
%     bin_time       (4)      = extract_1p detect_bin_time
%     crop           ('auto') | 'full' | [row0 col0 h w] (1-based, full-resolution px)
%     crop_px        (400)
%     grid           struct of vectors; every combination is run:
%                      avg_cell_radius          ([10 13])     full-resolution px
%                      T_min_snr                ([5 7])
%                      cellfind_min_snr         (1)
%                      spatial_highpass_cutoff  ([5 8 12])   the setting that mattered on real data
%                    any other extract_1p field may be added as a vector
%     detect_downsample_space_by (2)   as in the full run on a full-resolution movie
%     use_gpu        ('auto')
%     keep_movie     (false)  keep the cropped sweep movie h5 after the sweep

if nargin == 1 && (ischar(cfg) || isstring(cfg)) && strcmpi(cfg, 'selftest')
    out = run_selftest();
    return;
end
t_all = tic;
cfg = defaults(cfg);
if ~isfolder(cfg.out_dir), mkdir(cfg.out_dir); end

% ---- 1. one piece of one recording, binned like the detection movie -----------------
ft = readtable(cfg.frames_csv, 'Delimiter', ',', 'TextType', 'char');
ft.recording_id = cellstr(string(ft.recording_id));
if isempty(cfg.recording_id), cfg.recording_id = ft.recording_id{1}; end
sel = strcmp(ft.recording_id, cfg.recording_id) & ft.valid == 1;
if ~any(sel)
    error('extract_param_sweep:recording', 'no valid frames of %s', cfg.recording_id);
end
isx = ft.isx_frame(sel);
h5r = ft.h5_row(sel);
fps = frame_rate(ft(sel, :));
t = isx / fps;
t0 = min(max(cfg.sweep_start_s, 0), max(t(end) - cfg.sweep_len_s, 0));
use = t >= t0 & t < t0 + cfg.sweep_len_s;
rows = h5r(use);
nb = floor(numel(rows) / cfg.bin_time);
if nb < 50
    error('extract_param_sweep:short', 'only %d binned frames in the sweep window', nb);
end
rows = rows(1:nb * cfg.bin_time);
hi = h5info(cfg.h5, '/mov');
H = hi.Dataspace.Size(1);
W = hi.Dataspace.Size(2);
fprintf('extract_param_sweep: %s %.0f-%.0f s, %d frames -> %d binned (%.2f Hz), %dx%d\n', ...
        cfg.recording_id, t0, t0 + numel(rows) / fps, numel(rows), nb, fps / cfg.bin_time, H, W);
M = read_binned(cfg.h5, rows, H, W, cfg.bin_time);
fps_b = fps / cfg.bin_time;

% ---- 2. overview of the whole field -------------------------------------------------
r_mid = median(cfg.grid.avg_cell_radius);
[Cn, PNR, mean_img] = seed_images(M, r_mid);
score = imgaussfilt(Cn .* PNR, r_mid);

% ---- 3. crop -------------------------------------------------------------------------
crop = choose_crop(cfg, score, H, W);
Mc = M(crop(1):crop(1) + crop(3) - 1, crop(2):crop(2) + crop(4) - 1, :);
clear M
seed = Cn .* PNR;                              % bright exactly at active somata
Cn_c = seed(crop(1):crop(1) + crop(3) - 1, crop(2):crop(2) + crop(4) - 1);
overview_png(fullfile(cfg.out_dir, [cfg.tag '_sweep_overview.png']), mean_img, Cn, PNR, crop, ...
             cfg, t0, nb, fps_b);

sub_h5 = fullfile(cfg.out_dir, [cfg.tag '_sweep_movie.h5']);
sub_csv = fullfile(cfg.out_dir, [cfg.tag '_sweep_frames.csv']);
write_sub_movie(sub_h5, sub_csv, Mc, fps_b);
clear Mc

% ---- 4. every setting ------------------------------------------------------------------
G = expand_grid(cfg.grid);
nR = numel(G);
fprintf('extract_param_sweep: %d settings on a %dx%d crop\n', nR, crop(3), crop(4));
rowsum = cell(nR, 1);
missed_rc = cell(nR, 1);
QC = cell(nR, 1);
S_all = cell(nR, 1);
for k = 1:nR
    rd = fullfile(cfg.out_dir, sprintf('run_%02d', k));
    if ~isfolder(rd), mkdir(rd); end
    ec = G(k);
    % grid fields that are not extract_1p options go to EXTRACT itself
    known = {'avg_cell_radius', 'spatial_highpass_cutoff', 'T_min_snr', 'cellfind_min_snr', ...
             'eccent_thresh', 'avg_event_tau_s', 'detect_downsample_space_by', 'trace_output_option'};
    other = setdiff(fieldnames(ec), known);
    if ~isempty(other)
        ov = struct();
        for j = 1:numel(other)
            ov.(other{j}) = ec.(other{j});
        end
        ec = rmfield(ec, other);
        ec.extract_overrides = ov;
    end
    ec.h5 = sub_h5;
    ec.frames_csv = sub_csv;
    ec.mouse_id = cfg.tag;
    ec.day = 0;
    ec.out_dir = rd;
    ec.detect_bin_time = 1;                      % already binned above
    ec.detect_downsample_space_by = cfg.detect_downsample_space_by;
    ec.use_gpu = cfg.use_gpu;
    ec.movie_mask = [];
    tk = tic;
    try
        prev = dir(fullfile(rd, '*_extract.mat'));
        if cfg.reuse_runs && ~isempty(prev) && isfile(fullfile(rd, 'setting.json')) && ...
                isequal(jsondecode(fileread(fullfile(rd, 'setting.json'))), jsondecode(jsonencode(G(k))))
            m = load(fullfile(prev(1).folder, prev(1).name), 'spatial_weights', 'temporal_raw');
        else
            r = evalc_extract(ec);
            m = load(r.files.mat, 'spatial_weights', 'temporal_raw');
            fid = fopen(fullfile(rd, 'setting.json'), 'w');
            fprintf(fid, '%s', jsonencode(G(k)));
            fclose(fid);
        end
        S = full(m.spatial_weights);
        Q = cell_qc_metrics(S, m.temporal_raw, G(k).avg_cell_radius, fps_b);
        % proof from the movie: does the movie at each component's own events show it?
        Q.event_image_corr = NaN(height(Q), 1);
        sft = struct('h5_row', (1:nb)');
        for c = 1:height(Q)
            [~, Q.event_image_corr(c)] = event_image(sub_h5, sft, m.temporal_raw(:, c), ...
                S(:, :, c), Q.row(c), Q.col(c), G(k).avg_cell_radius, fps_b);
            if strcmp(Q.verdict{c}, 'cell') && ~(Q.event_image_corr(c) >= 0.5)
                Q.verdict{c} = 'check';
                Q.reasons{c} = sprintf('movie at its events does not match (r %.2f)', ...
                                       Q.event_image_corr(c));
            end
        end
        status = 'ok';
        msg = '';
    catch err
        S = zeros(crop(3), crop(4), 0);
        Q = cell_qc_metrics(S, zeros(nb, 0), G(k).avg_cell_radius, fps_b);
        Q.event_image_corr = zeros(0, 1);
        status = 'failed';
        msg = err.message;
        fprintf(2, '  run %d failed: %s\n', k, msg);
    end
    writetable(Q, fullfile(rd, 'component_qc.csv'));
    QC{k} = Q;
    S_all{k} = S;
    s = G(k);
    s.run = k;
    s.status = status;
    s.n_found = height(Q);
    s.n_cell = nnz(strcmp(Q.verdict, 'cell'));
    s.n_check = nnz(strcmp(Q.verdict, 'check'));
    s.n_not_cell = nnz(strcmp(Q.verdict, 'not_cell'));
    s.frac_not_cell = s.n_not_cell / max(1, s.n_found);
    s.median_snr_cells = median(Q.snr(strcmp(Q.verdict, 'cell')), 'omitnan');
    [s.n_seeds, s.n_missed, missed_rc{k}] = missed_seeds(Cn_c, S, G(k).avg_cell_radius);
    s.runtime_s = toc(tk);
    s.error = msg;
    rowsum{k} = s;
    fprintf('  run %2d/%d  %s  found %3d  cell %3d  check %3d  not %3d  missed %d/%d  (%.0f s)\n', ...
            k, nR, setting_label(G(k)), s.n_found, s.n_cell, s.n_check, s.n_not_cell, s.n_missed, ...
            s.n_seeds, s.runtime_s);
end
summary = struct2table([rowsum{:}]);
writetable(summary, fullfile(cfg.out_dir, [cfg.tag '_sweep_summary.csv']));

% ---- 5. sheet + suggestion -----------------------------------------------------------
ok = strcmp(summary.status, 'ok') & summary.frac_not_cell <= 0.25 & summary.n_cell > 0;
sug = [];
if any(ok)
    cand = find(ok);
    [~, o] = sortrows([-summary.n_cell(cand), summary.n_not_cell(cand)]);
    sug = cand(o(1));
end
sheet_files = sheet_png(fullfile(cfg.out_dir, [cfg.tag '_sweep_sheet.png']), Cn_c, S_all, QC, G, ...
                        summary, sug, missed_rc);
save(fullfile(cfg.out_dir, [cfg.tag '_sweep_seed_image.mat']), 'Cn_c', 'crop');

if ~cfg.keep_movie
    delete(sub_h5);
end
out = struct('module', 'extract_param_sweep', 'tag', cfg.tag, 'recording_id', cfg.recording_id, ...
    'window_s', [t0, t0 + numel(rows) / fps], 'fps', fps, 'bin_time', cfg.bin_time, ...
    'crop_row_col_h_w', crop, 'grid', cfg.grid, 'summary', table2struct(summary), ...
    'suggested_run', sug, 'suggested', [], 'files', struct( ...
    'summary_csv', fullfile(cfg.out_dir, [cfg.tag '_sweep_summary.csv']), ...
    'sheet_png', {sheet_files}, ...
    'overview_png', fullfile(cfg.out_dir, [cfg.tag '_sweep_overview.png'])), ...
    'runtime_s', toc(t_all), 'note', ['suggested = most "cell" outlines with <= 25 % ', ...
    '"not_cell"; a suggestion only - judge the sheet and the per-run footprint PNGs']);
if ~isempty(sug)
    out.suggested = G(sug);
end
fid = fopen(fullfile(cfg.out_dir, [cfg.tag '_sweep.json']), 'w');
fprintf(fid, '%s', jsonencode(out, 'PrettyPrint', true));
fclose(fid);
if isempty(sug)
    fprintf('extract_param_sweep: no setting kept not_cell <= 25 %% - widen the grid\n');
else
    fprintf('extract_param_sweep: suggested run %d (%s): %d cell-like components\n', sug, ...
            setting_label(G(sug)), summary.n_cell(sug));
end
end


% =====================================================================================
function cfg = defaults(cfg)
if isfield(cfg, 'day_json') && ~isempty(cfg.day_json)
    js = jsondecode(fileread(cfg.day_json));
    jdir = fileparts(cfg.day_json);
    if ~isfield(cfg, 'h5'), cfg.h5 = relocate(js.files.h5, jdir); end
    if ~isfield(cfg, 'frames_csv'), cfg.frames_csv = relocate(js.files.frames_csv, jdir); end
    if ~isfield(cfg, 'tag'), cfg.tag = sprintf('%s_d%d', js.mouse_id, js.day); end
end
grid0 = struct('avg_cell_radius', [10 13], 'T_min_snr', [5 7], 'cellfind_min_snr', 1, ...
               'spatial_highpass_cutoff', [5 8 12]);   % 5941 test: the high-pass matters most
def = struct('tag', 'sweep', 'recording_id', '', 'sweep_start_s', 180, 'sweep_len_s', 300, ...
             'bin_time', 4, 'crop', 'auto', 'crop_px', 400, 'grid', grid0, ...
             'detect_downsample_space_by', 2, 'use_gpu', 'auto', 'keep_movie', false, ...
             'reuse_runs', true);
f = fieldnames(def);
for k = 1:numel(f)
    if ~isfield(cfg, f{k}) || (isempty(cfg.(f{k})) && ~strcmp(f{k}, 'recording_id'))
        cfg.(f{k}) = def.(f{k});
    end
end
g = fieldnames(grid0);
for k = 1:numel(g)
    if ~isfield(cfg.grid, g{k}), cfg.grid.(g{k}) = grid0.(g{k}); end
end
for k = {'h5', 'frames_csv', 'out_dir'}
    if ~isfield(cfg, k{1}) || isempty(cfg.(k{1}))
        error('extract_param_sweep:cfg', 'cfg.%s (or cfg.day_json) is required', k{1});
    end
end
end


function p = relocate(p, jdir)
if ~isfile(p)
    [~, nm, ext] = fileparts(p);
    p = fullfile(jdir, [nm ext]);
end
end


function fps = frame_rate(ft)
ok = isfinite(ft.tsc_us);
if nnz(ok) >= 2
    a = find(ok, 1); b = find(ok, 1, 'last');
    fps = (ft.isx_frame(b) - ft.isx_frame(a)) / ((ft.tsc_us(b) - ft.tsc_us(a)) / 1e6);
else
    error('extract_param_sweep:fps', 'no TSC timestamps in the frame table');
end
end


function M = read_binned(h5, rows, H, W, B)
nb = numel(rows) / B;
M = zeros(H, W, nb, 'single');
blk = 50 * B;
k = 0;
i = 1;
while i <= numel(rows)
    j = min(i + blk - 1, numel(rows));
    r = rows(i:j);
    if all(diff(r) == 1)
        X = single(h5read(h5, '/mov', [1 1 r(1)], [H W numel(r)]));
    else
        X = zeros(H, W, numel(r), 'single');
        for q = 1:numel(r)
            X(:, :, q) = single(h5read(h5, '/mov', [1 1 r(q)], [H W 1]));
        end
    end
    n = numel(r) / B;
    M(:, :, k + 1:k + n) = squeeze(mean(reshape(X, H, W, B, n), 3));
    k = k + n;
    i = j + 1;
end
end


function [Cn, PNR, mean_img] = seed_images(M, radius)
% CNMF-E style seed images on a 2x spatially binned copy (for speed), back to full size.
[H, W, n] = size(M);
mean_img = mean(M, 3);
h2 = floor(H / 2); w2 = floor(W / 2);
X = M(1:2 * h2, 1:2 * w2, :);
X = squeeze(mean(mean(reshape(X, 2, h2, 2, w2, n), 1), 3));
X = reshape(X, h2, w2, n);
X = X - mean(X, 3);
sig = max(1, radius / 2);
for k = 1:n
    f = X(:, :, k);
    X(:, :, k) = imgaussfilt(f, sig / 2) - imgaussfilt(f, 3 * sig);
end
sd = std(X, 0, 3) + eps;
Z = X ./ sd;
Cn2 = zeros(h2, w2, 'single');
cnt = 0;
for dy = -1:1
    for dx = -1:1
        if dy == 0 && dx == 0, continue; end
        Cn2 = Cn2 + mean(Z .* circshift(Z, [dy dx 0]), 3);
        cnt = cnt + 1;
    end
end
Cn2 = Cn2 / cnt;
noise = 1.4826 * median(abs(diff(X, 1, 3)), 3) / sqrt(2) + eps;
PNR2 = max(movmean(X, 3, 3), [], 3) ./ noise;
Cn = imresize(Cn2, [H W], 'bilinear');
PNR = imresize(PNR2, [H W], 'bilinear');
Cn = max(Cn, 0);
end


function crop = choose_crop(cfg, score, H, W)
if ischar(cfg.crop) || isstring(cfg.crop)
    if strcmpi(cfg.crop, 'full')
        crop = [1 1 H W];
        return;
    end
    s = min([cfg.crop_px, H, W]);
    m = 20;                                           % stay off the frame edge
    box = conv2(score, ones(s, 1), 'valid');
    box = conv2(box, ones(1, s), 'valid');            % box(i,j) = sum over the s x s window
    box(1:min(m, end), :) = -inf; box(max(1, end - m + 1):end, :) = -inf;
    box(:, 1:min(m, end)) = -inf; box(:, max(1, end - m + 1):end) = -inf;
    if all(isinf(box(:))), box = conv2(conv2(score, ones(s, 1), 'valid'), ones(1, s), 'valid'); end
    [~, i] = max(box(:));
    [r0, c0] = ind2sub(size(box), i);
    crop = [r0 c0 s s];
else
    crop = double(cfg.crop(:)');
    if numel(crop) ~= 4 || crop(1) < 1 || crop(2) < 1 || crop(1) + crop(3) - 1 > H || crop(2) + crop(4) - 1 > W
        error('extract_param_sweep:crop', 'crop must be [row0 col0 h w] inside %dx%d', H, W);
    end
end
end


function write_sub_movie(h5, csv, M, fps_b)
if isfile(h5), delete(h5); end
[H, W, n] = size(M);
h5create(h5, '/mov', [H W n], 'Datatype', 'single', 'ChunkSize', [H W min(n, 50)]);
h5write(h5, '/mov', M);
h5create(h5, '/mean_image', [H W], 'Datatype', 'single');
h5write(h5, '/mean_image', mean(M, 3));
fid = fopen(csv, 'w');
fprintf(fid, 'recording_id,assay,isx_frame,valid,invalid_reason,h5_row,tsc_us\n');
for k = 1:n
    fprintf(fid, 'sweep,pre,%d,1,,%d,%.3f\n', k - 1, k, (k - 1) / fps_b * 1e6);
end
fclose(fid);
end


function G = expand_grid(grid)
f = fieldnames(grid);
v = cellfun(@(x) num2cell(x(:)'), struct2cell(grid), 'UniformOutput', false);
n = cellfun(@numel, v);
G = struct([]);
idx = cell(1, numel(f));
for k = 1:prod(n)
    [idx{:}] = ind2sub(n(:)', k);
    s = struct();
    for j = 1:numel(f)
        s.(f{j}) = v{j}{idx{j}};
    end
    G = [G, s]; %#ok<AGROW>
end
end


function r = evalc_extract(ec)
% extract_1p prints EXTRACT's full log; keep it in the run folder instead of the console.
txt = evalc('r = extract_1p(ec);');
fid = fopen(fullfile(ec.out_dir, 'extract_log.txt'), 'w');
fprintf(fid, '%s', txt);
fclose(fid);
end


function s = setting_label(g)
s = sprintf('r %g  Tsnr %g  cf %g  hp %g', g.avg_cell_radius, g.T_min_snr, g.cellfind_min_snr, ...
            g.spatial_highpass_cutoff);
f = setdiff(fieldnames(g), {'avg_cell_radius', 'T_min_snr', 'cellfind_min_snr', 'spatial_highpass_cutoff'});
for k = 1:numel(f)
    s = sprintf('%s  %s %g', s, f{k}, g.(f{k}));
end
end


function overview_png(png, mean_img, Cn, PNR, crop, cfg, t0, nb, fps_b)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1500 520]);
ims = {mean_img, Cn, Cn .* PNR};
ttl = {'mean image', 'local correlation (Cn)', 'Cn x PNR (somata = bright round spots)'};
for k = 1:3
    subplot(1, 3, k);
    imagesc(ims{k}); axis image off; colormap(gca, gray);
    v = prctile(ims{k}(:), [1 99.7]);
    if v(2) > v(1), caxis(v); end
    hold on;
    rectangle('Position', [crop(2) crop(1) crop(4) crop(3)], 'EdgeColor', [1 0.8 0], 'LineWidth', 1.5);
    title(ttl{k}, 'FontWeight', 'normal');
end
sgtitle(sprintf('%s  %s  %.0f s from %.0f s, %d frames at %.2f Hz  |  yellow box = sweep crop', ...
        strrep(cfg.tag, '_', '\_'), strrep(cfg.recording_id, '_', '\_'), nb / fps_b, t0, nb, fps_b), ...
        'FontSize', 11);
exportgraphics(fig, png, 'Resolution', 130);
close(fig);
end


function files = sheet_png(png, bg, S_all, QC, G, summary, sug, missed_rc)
% Pages of at most 9 settings (3 x 3), composed as pixels (no axes), so every panel keeps
% the crop's exact 1:1 geometry.
nR = numel(G);
per = 9;
P = 380;                                   % panel size, px
TH = 44;                                   % title band, px
col = struct('cell', [40 190 60], 'check', [255 190 0], 'not_cell', [230 40 40]);
v = prctile(bg(:), [1 99.7]);
g = uint8(255 * min(max((double(bg) - v(1)) / max(v(2) - v(1), eps), 0), 1));
sc = P / max(size(bg));
files = {};
for p0 = 1:per:nR
    ks = p0:min(p0 + per - 1, nR);
    tiles = cell(1, numel(ks));
    for t = 1:numel(ks)
        k = ks(t);
        im = repmat(imresize(g, sc, 'bilinear'), 1, 1, 3);
        S = S_all{k};
        Q = QC{k};
        for c = 1:size(S, 3)
            f = S(:, :, c);
            bw = imresize(f >= 0.3 * max(f(:)), sc, 'nearest');
            e = bwperim(bw);
            e = imdilate(e, ones(2));
            rgb = col.(Q.verdict{c});
            for ch = 1:3
                a = im(:, :, ch);
                a(e) = rgb(ch);
                im(:, :, ch) = a;
            end
        end
        mr = missed_rc{k};
        if ~isempty(mr)
            im = insertMarker(im, [mr(:, 2), mr(:, 1)] * sc, 'plus', 'Color', 'cyan', 'Size', 7);
        end
        r = G(k).avg_cell_radius * sc;
        im = insertShape(im, 'circle', [2 * r + 4, 2 * r + 4, r], 'Color', 'cyan', 'LineWidth', 2);
        band = 255 * ones(TH, size(im, 2), 3, 'uint8');
        tc = 'black';
        if isequal(k, sug), tc = [0 120 30]; end
        band = insertText(band, [2 1], sprintf('#%d  %s', k, setting_label(G(k))), 'FontSize', 11, ...
                          'BoxOpacity', 0, 'TextColor', tc);
        band = insertText(band, [2 21], sprintf('%d found: %d cell, %d check, %d not | missed %d of %d', ...
                          summary.n_found(k), summary.n_cell(k), summary.n_check(k), ...
                          summary.n_not_cell(k), summary.n_missed(k), summary.n_seeds(k)), ...
                          'FontSize', 11, 'BoxOpacity', 0, 'TextColor', tc);
        tiles{t} = [band; im];
    end
    while numel(tiles) < per && numel(tiles) > 3 && mod(numel(tiles), 3)
        tiles{end + 1} = 255 * ones(size(tiles{1}), 'uint8'); %#ok<AGROW>
    end
    nc = min(3, numel(tiles));
    rows = {};
    gap = 255 * ones(size(tiles{1}, 1), 10, 3, 'uint8');
    for r0 = 1:nc:numel(tiles)
        rr = tiles(r0:min(r0 + nc - 1, numel(tiles)));
        while numel(rr) < nc, rr{end + 1} = 255 * ones(size(tiles{1}), 'uint8'); end %#ok<AGROW>
        line = rr{1};
        for q = 2:numel(rr), line = [line, gap, rr{q}]; end %#ok<AGROW>
        rows{end + 1} = line; %#ok<AGROW>
        rows{end + 1} = 255 * ones(10, size(line, 2), 3, 'uint8'); %#ok<AGROW>
    end
    sheet = vertcat(rows{:});
    head = 255 * ones(48, size(sheet, 2), 3, 'uint8');
    head = insertText(head, [4 2], 'EXTRACT settings on the same crop (background: Cn x PNR, bright = active soma)', ...
                      'FontSize', 13, 'BoxOpacity', 0);
    head = insertText(head, [4 24], ['green cell-like   yellow check   red not a cell   ' ...
                      'cyan + bright spot no component covers   cyan circle tested radius   green title = suggested'], ...
                      'FontSize', 11, 'BoxOpacity', 0);
    sheet = [head; sheet];
    if p0 == 1
        f = png;
    else
        f = strrep(png, '.png', sprintf('_p%d.png', ceil(p0 / per)));
    end
    imwrite(sheet, f);
    files{end + 1} = f; %#ok<AGROW>
end
end


function [n_seed, n_missed, rc] = missed_seeds(seed, S, radius)
% Bright, compact spots of the Cn x PNR image (local maxima > median + 6 robust SD of the
% smoothed image, >= 1.5 radii apart) that no component's footprint (>= 0.3 x its peak)
% covers. A recall proxy: a real soma that EXTRACT missed shows up here.
sm = imgaussfilt(double(seed), max(1, radius / 3));
mu = median(sm(:));
sd = 1.4826 * median(abs(sm(:) - mu));
pk = imregionalmax(sm) & sm > mu + 6 * sd;
m = max(20, round(2 * radius));               % the frame / crop edge is noise, not somata
pk([1:m, end - m + 1:end], :) = false;
pk(:, [1:m, end - m + 1:end]) = false;
[r, c] = find(pk);
v = sm(pk);
[~, o] = sort(v, 'descend');
r = r(o); c = c(o);
keep = true(numel(r), 1);
for i = 1:numel(r)
    if ~keep(i), continue; end
    d = hypot(r - r(i), c - c(i));
    keep(d < 1.5 * radius & (1:numel(r))' > i) = false;
end
r = r(keep); c = c(keep);
cov = false(size(seed));
for k = 1:size(S, 3)
    im = S(:, :, k);
    cov = cov | im >= 0.3 * max(im(:));
end
cov = imdilate(cov, strel('disk', max(1, round(radius / 3))));
hit = cov(sub2ind(size(cov), r, c));
n_seed = numel(r);
n_missed = nnz(~hit);
rc = [r(~hit), c(~hit)];
end


% =====================================================================================
function out = run_selftest()
% A synthetic full-resolution-like day: 20 Gaussian somata, a vessel and a smooth
% background on a 260 x 300 field. The sweep must run every setting, find most somata
% as 'cell' at the right radius, and write the sheet, summary and JSON.
root = fullfile(tempdir, 'extract_param_sweep_selftest');
if isfolder(root), rmdir(root, 's'); end
mkdir(root);
rng(3);
H = 260; W = 300; fps = 20; n = 2400; r = 6;
[xx, yy] = meshgrid(1:W, 1:H);
nc = 20;
cen = [40 + rand(nc, 1) * (H - 80), 40 + rand(nc, 1) * (W - 80)];
M = 800 + 300 * exp(-((xx - W / 2) .^ 2 + (yy - H / 2) .^ 2) / (2 * 150 ^ 2));
M = repmat(single(M), 1, 1, n);
for c = 1:nc
    fp = exp(-((xx - cen(c, 2)) .^ 2 + (yy - cen(c, 1)) .^ 2) / (2 * (0.6 * r) ^ 2));
    sp = zeros(n, 1);
    sp(randperm(n, 25)) = 1;
    tr = filter(1, [1 -exp(-1 / (0.6 * fps))], sp) * 400;
    M = M + single(fp) .* reshape(single(tr), 1, 1, n);
end
vessel = exp(-((yy - 0.4 * xx - 30) .^ 2) / (2 * 3 ^ 2));
M = M - single(150 * vessel);
M = M + single(randn(size(M)) * 25);
h5 = fullfile(root, 'S01_d1_mc.h5');
h5create(h5, '/mov', [H W n], 'Datatype', 'single', 'ChunkSize', [H W 50]);
h5write(h5, '/mov', M);
csv = fullfile(root, 'S01_d1_mc_frames.csv');
fid = fopen(csv, 'w');
fprintf(fid, 'recording_id,assay,isx_frame,valid,invalid_reason,h5_row,tsc_us\n');
for k = 1:n
    fprintf(fid, 'S01_d1_pre,pre,%d,1,,%d,%.1f\n', k - 1, k, (k - 1) / fps * 1e6);
end
fclose(fid);
cfg = struct('h5', h5, 'frames_csv', csv, 'out_dir', fullfile(root, 'sweep'), 'tag', 'S01_d1', ...
             'sweep_start_s', 0, 'sweep_len_s', 120, 'crop', 'full', 'use_gpu', false, ...
             'detect_downsample_space_by', 1, ...
             'grid', struct('avg_cell_radius', [3 6], 'T_min_snr', [5 10]));
o = extract_param_sweep(cfg);
S = struct2table(o.summary);
assert_true(height(S) == 4 && all(strcmp(S.status, 'ok')), 'all 4 settings ran');
assert_true(all(isfile(o.files.sheet_png)) && isfile(o.files.summary_csv) && isfile(o.files.overview_png), ...
            'sheet, summary and overview written');
best = max(S.n_cell(S.avg_cell_radius == 6));
assert_true(best >= 12, sprintf('radius 6: %d of %d somata cell-like', best, nc));
assert_true(~isempty(o.suggested_run), 'a setting is suggested');
fprintf('extract_param_sweep selftest PASSED (best %d of %d somata cell-like)\n', best, nc);
out = true;
end


function assert_true(c, msg)
if ~c, error('extract_param_sweep:selftest', 'FAILED: %s', msg); end
end
