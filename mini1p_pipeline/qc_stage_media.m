function out = qc_stage_media(cfg)
%QC_STAGE_MEDIA  Show the movie at every processing stage, so each step can be checked by
%   eye before cells are trusted: raw vs motion-corrected, and the filtered movie EXTRACT
%   actually detects cells on.
%
%   out = qc_stage_media(cfg)
%
%   cfg.raw_h5      lossless copy (<rec>_raw.h5:/mov), optional (no raw panel without it)
%   cfg.mc_h5       motion-corrected movie (<rec>_mc.h5:/mov)
%   cfg.frames_csv  frame table of the recording (raw frame table: h5_row, tsc_us)
%   cfg.out_dir, cfg.tag
%   optional (default)
%     window_s        ([60 90])  piece shown in the videos (s from the recording start)
%     avg_cell_radius (13)       full-resolution px, as given to EXTRACT
%     spatial_highpass_cutoff (5)
%     bin_time        (4)        EXTRACT detection binning
%     downsample_space_by (2)    EXTRACT detection binning
%
%   Writes
%     <tag>_qc_1_motion.mp4    raw | motion-corrected, both high-passed so vessels and
%                              somata are visible, real time. Raw jitters, MC must not.
%     <tag>_qc_2_filtered.mp4  EXTRACT's own preprocessing (dF/F with per-pixel F0, spatial
%                              high-pass, baseline correction) of the binned movie, played
%                              at 4x: this is what cells are found on. Somata = round
%                              flashes; vessels, edges and neuropil haze are not cells.
%     <tag>_qc_stages.png      mean raw, mean MC, temporal std, max of the filtered movie,
%                              local correlation (Cn), Cn x PNR
%     <tag>_qc_motion_frames.png  the most-shifted frame before and after MC (magenta)
%                              over the template (green): white = aligned

def = struct('raw_h5', '', 'window_s', [60 90], 'avg_cell_radius', 13, ...
             'spatial_highpass_cutoff', 5, 'bin_time', 4, 'downsample_space_by', 2);
f = fieldnames(def);
for k = 1:numel(f)
    if ~isfield(cfg, f{k}) || isempty(cfg.(f{k})), cfg.(f{k}) = def.(f{k}); end
end
if ~isfolder(cfg.out_dir), mkdir(cfg.out_dir); end
P = @(s) fullfile(cfg.out_dir, [cfg.tag s]);

ft = readtable(cfg.frames_csv, 'Delimiter', ',', 'TextType', 'char');
v = ft.valid == 1;
fps = (ft.isx_frame(find(v, 1, 'last')) - ft.isx_frame(find(v, 1))) / ...
      ((ft.tsc_us(find(v, 1, 'last')) - ft.tsc_us(find(v, 1))) / 1e6);
t = ft.isx_frame(v) / fps;
h5r = ft.h5_row(v);
hi = h5info(cfg.mc_h5, '/mov');
H = hi.Dataspace.Size(1); W = hi.Dataspace.Size(2); N = hi.Dataspace.Size(3);
w = cfg.window_s;
w(2) = min(w(2), t(end));
w(1) = min(w(1), max(0, w(2) - 10));
rows = h5r(t >= w(1) & t < w(2));
rows = rows(1:floor(numel(rows) / cfg.bin_time) * cfg.bin_time);
fprintf('qc_stage_media %s: window %.0f-%.0f s (%d frames)\n', cfg.tag, w(1), w(2), numel(rows));

M = single(h5read(cfg.mc_h5, '/mov', [1 1 rows(1)], [H W numel(rows)]));
R = [];
if ~isempty(cfg.raw_h5) && isfile(cfg.raw_h5)
    c0 = [1 1];                              % the mc movie may be cropped (motion_correct_h5 crop_rect)
    try
        cr = h5readatt(cfg.mc_h5, '/mov', 'crop_rect'); c0 = [cr(1) cr(3)];
    catch
    end
    R = single(h5read(cfg.raw_h5, '/mov', [c0 rows(1)], [H W numel(rows)]));
end
r = cfg.avg_cell_radius;

% ---- 1. motion video ---------------------------------------------------------------
hp = @(X) hp_stack(X, r);
Mh = hp(M(1:2:end, 1:2:end, :));
lim = prctile(Mh(:), [0.5 99.8]);
vw = VideoWriter(P('_qc_1_motion.mp4'), 'MPEG-4');
vw.FrameRate = round(fps);
vw.Quality = 90;
open(vw);
if ~isempty(R), Rh = hp(R(1:2:end, 1:2:end, :)); end
for k = 1:size(Mh, 3)
    a = to8(Mh(:, :, k), lim);
    if ~isempty(R)
        b = to8(Rh(:, :, k), lim);
        fr = [label(b, 'raw (high-passed)'), 255 * ones(size(a, 1), 6, 3, 'uint8'), ...
              label(a, 'motion-corrected (high-passed)')];
    else
        fr = label(a, 'motion-corrected (high-passed)');
    end
    fr = insertText(fr, [10 size(fr, 1) - 30], sprintf('%.2f s', t(find(h5r == rows(k), 1))), ...
                    'FontSize', 16, 'BoxOpacity', 0.4, 'TextColor', 'white');
    writeVideo(vw, fr);
end
close(vw);
clear Rh Mh

% ---- 2. the movie EXTRACT detects on ------------------------------------------------
d = cfg.downsample_space_by;
B = cfg.bin_time;
nb = numel(rows) / B;
X = squeeze(mean(reshape(M, H, W, B, nb), 3));
if d > 1
    X = imresize(X, 1 / d, 'box');
end
ec = get_defaults(struct());
ec.avg_cell_radius = r / d;
ec.spatial_highpass_cutoff = cfg.spatial_highpass_cutoff;
ec.remove_background = 1;
ec.use_gpu = 0;
ec.avg_event_tau = max(1, round(0.5 * fps / B));
ec.preprocess = 1;
ec.skip_dff = 0;
[Xp, ~] = preprocess_movie(X, ec);
% For the eye only: a light Gaussian (sigma = radius / 3 on the binned grid) and a 3-frame
% running mean. Single binned frames are dominated by shot noise; EXTRACT integrates over
% space and time itself, the display has to as well.
Xd = movmean(Xp, 3, 3);
for k = 1:size(Xd, 3)
    Xd(:, :, k) = imgaussfilt(Xd(:, :, k), max(0.5, r / d / 3));
end
[hh, ww, ~] = size(Xd);
ctr = Xd(round(hh * 0.2):round(hh * 0.8), round(ww * 0.2):round(ww * 0.8), :);
lim = prctile(ctr(:), [2 99.95]);
vw = VideoWriter(P('_qc_2_filtered.mp4'), 'MPEG-4');
vw.FrameRate = round(fps);
vw.Quality = 90;
open(vw);
for k = 1:size(Xp, 3)
    fr = label(to8(imresize(Xd(:, :, k), d, 'nearest'), lim), ...
               sprintf(['what EXTRACT sees: dF/F, high-pass, binned %dx time %dx space ' ...
                        '(4x speed; display lightly smoothed)'], B, d));
    writeVideo(vw, fr);
end
close(vw);

% ---- 3. stage images ----------------------------------------------------------------
mean_mc = mean(M, 3);
sd_mc = std(M, 0, 3);
mx = imresize(max(Xp, [], 3), [H W]);
[Cn, PNR] = seed_images(Xp);
Cn = imresize(Cn, [H W]); PNR = imresize(PNR, [H W]);
ims = {mean(R, 3), mean_mc, sd_mc, mx, Cn, Cn .* PNR};
ttl = {'mean, raw', 'mean, motion-corrected', 'temporal std (MC)', ...
       'max of the filtered movie', 'local correlation Cn', 'Cn x PNR (somata bright)'};
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [20 20 1500 700]);
for k = 1:6
    subplot(2, 3, k);
    if isempty(ims{k}), axis off; title([ttl{k} ' (no raw)']); continue; end
    imagesc(ims{k}); axis image off; colormap(gca, gray);
    q = prctile(ims{k}(:), [1 99.7]);
    if q(2) > q(1), clim(q); end
    title(ttl{k}, 'FontWeight', 'normal');
end
sgtitle(sprintf('%s  %.0f-%.0f s', strrep(cfg.tag, '_', '\_'), w(1), w(2)));
exportgraphics(fig, P('_qc_stages.png'), 'Resolution', 120);
close(fig);

% ---- 4. worst frame before / after ----------------------------------------------------
shf = regexprep(cfg.mc_h5, '_mc\.h5$', '_mc_shifts.csv');
if isfile(shf) && ~isempty(R)
    sh = readtable(shf, 'TextType', 'char');
    in = ismember(sh.h5_row, rows);
    [~, j] = max(hypot(sh.x_shift_px(in), sh.y_shift_px(in)));
    rr = sh.h5_row(in);
    k = find(rows == rr(j), 1);
    tmpl = hp_stack(mean_mc, r);
    a = hp_stack(R(:, :, k), r); b = hp_stack(M(:, :, k), r);
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [20 20 1400 480]);
    subplot(1, 2, 1); imshow(imfuse(norm01(a), norm01(tmpl), 'falsecolor', 'ColorChannels', [1 2 1]));
    title(sprintf('before MC, frame h5 row %d (shift %.1f px)', rows(k), ...
          hypot(sh.x_shift_px(sh.h5_row == rows(k)), sh.y_shift_px(sh.h5_row == rows(k)))), 'FontWeight', 'normal');
    subplot(1, 2, 2); imshow(imfuse(norm01(b), norm01(tmpl), 'falsecolor', 'ColorChannels', [1 2 1]));
    title('after MC (magenta = frame, green = mean; white = aligned)', 'FontWeight', 'normal');
    exportgraphics(fig, P('_qc_motion_frames.png'), 'Resolution', 110);
    close(fig);
end
save(P('_qc_seed_images.mat'), 'Cn', 'PNR', 'mean_mc', 'sd_mc', 'mx', '-v7');
% zoom on the most cell-like 300 x 300 px, circles of 8 / 12 / 16 px radius for sizing
sc = imgaussfilt(Cn .* PNR, r);
[~, i] = max(sc(:));
[r0, c0] = ind2sub(size(sc), i);
hw = 150;
rr = max(1, r0 - hw):min(H, r0 + hw); cc = max(1, c0 - hw):min(W, c0 + hw);
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [20 20 1300 460]);
zz = {mean_mc(rr, cc), Cn(rr, cc), Cn(rr, cc) .* PNR(rr, cc)};
zt = {'mean (MC)', 'Cn', 'Cn x PNR'};
for k = 1:3
    subplot(1, 3, k);
    imagesc(zz{k}); axis image off; colormap(gca, gray); hold on;
    q = prctile(zz{k}(:), [1 99.8]);
    if q(2) > q(1), clim(q); end
    th = linspace(0, 2 * pi, 60);
    for j = 1:3
        rad = [8 12 16];
        plot(25 + 40 * (j - 1) + rad(j) * cos(th), 25 + rad(j) * sin(th), 'c-', 'LineWidth', 1);
    end
    title(sprintf('%s  (rows %d-%d, cols %d-%d)', zt{k}, rr(1), rr(end), cc(1), cc(end)), ...
          'FontWeight', 'normal');
end
sgtitle('zoom on the most active area; cyan circles = radius 8, 12, 16 px (full resolution)');
exportgraphics(fig, P('_qc_zoom_sizes.png'), 'Resolution', 120);
close(fig);
out = struct('motion_mp4', P('_qc_1_motion.mp4'), 'filtered_mp4', P('_qc_2_filtered.mp4'), ...
             'stages_png', P('_qc_stages.png'), 'motion_frames_png', P('_qc_motion_frames.png'));
fprintf('qc_stage_media %s: done -> %s\n', cfg.tag, cfg.out_dir);
end


function Y = hp_stack(X, r)
Y = zeros(size(X), 'single');
for k = 1:size(X, 3)
    f = X(:, :, k);
    Y(:, :, k) = imgaussfilt(f, max(1, r / 4)) - imgaussfilt(f, 2 * r);
end
end


function a = to8(x, lim)
a = uint8(255 * min(max((x - lim(1)) / max(lim(2) - lim(1), eps), 0), 1));
a = repmat(a, 1, 1, 3);
end


function a = norm01(x)
q = prctile(x(:), [1 99.5]);
a = min(max((x - q(1)) / max(q(2) - q(1), eps), 0), 1);
end


function fr = label(a, txt)
fr = insertText(a, [10 8], txt, 'FontSize', 16, 'BoxColor', 'black', 'BoxOpacity', 0.5, ...
                'TextColor', 'white');
end


function [Cn, PNR] = seed_images(X)
X = X - mean(X, 3);
sd = std(X, 0, 3) + eps;
Z = X ./ sd;
Cn = zeros(size(X, 1), size(X, 2), 'single');
for dy = -1:1
    for dx = -1:1
        if dy == 0 && dx == 0, continue; end
        Cn = Cn + mean(Z .* circshift(Z, [dy dx 0]), 3) / 8;
    end
end
noise = 1.4826 * median(abs(diff(X, 1, 3)), 3) / sqrt(2) + eps;
PNR = max(movmean(X, 3, 3), [], 3) ./ noise;
Cn = max(Cn, 0);
end
