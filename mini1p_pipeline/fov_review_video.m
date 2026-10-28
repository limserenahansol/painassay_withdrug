function out = fov_review_video(extract_mat, review_csv, out_mp4, opts)
%FOV_REVIEW_VIDEO  The whole field of view as a video with every EXTRACT component outlined
%   and numbered, to decide cell / not cell by watching each one flash among its neighbours.
%
%   out = fov_review_video(extract_mat, review_csv, out_mp4, opts)
%
%   LEFT  motion-corrected movie (fixed contrast)
%   RIGHT dF/F (F0 = per-pixel mean, as EXTRACT) + spatial high-pass (EXTRACT's cutoff) +
%         light display smoothing - the view EXTRACT finds cells in
%   Outlines at 0.3 x each footprint's peak, numbered as in the review csv; green cell-like,
%   yellow check, red not a cell. Time (s) on every frame.
%
%   opts: bin_time (4) frames averaged per video frame; video_fps (30) -> 4x real time with
%         bin 4 at 30 Hz; window_s ([] = whole recording); scale (0.744: two panels fit 1920
%         px, the MPEG-4 limit); movie_h5 (from the EXTRACT config)

if nargin < 4, opts = struct(); end
B = getopt(opts, 'bin_time', 4);
vfps = getopt(opts, 'video_fps', 30);
win = getopt(opts, 'window_s', []);
sc = getopt(opts, 'scale', 0.744);
m = load(extract_mat);
movie = getopt(opts, 'movie_h5', m.config.extract_1p.h5);
R = readtable(review_csv, 'TextType', 'char');
S = full(m.spatial_weights);
ft = m.frame_table;
fps = double(m.fps);
radius = double(m.config.extract_1p.avg_cell_radius);
hp = double(m.config.extract.spatial_highpass_cutoff);
hi = h5info(movie, '/mov');
H = hi.Dataspace.Size(1); W = hi.Dataspace.Size(2);
v = find(ft.valid == 1);
t = (v - v(1)) / fps;
if ~isempty(win), v = v(t >= win(1) & t < win(2)); t = (v - v(1)) / fps + win(1); end
h5r = ft.h5_row(v);
nb = floor(numel(h5r) / B);
h5r = h5r(1:nb * B);
Hs = round(H * sc); Ws = round(W * sc);

% ---- pass 1: binned movie at display size, per-pixel mean (F0) ----------------------------
X = zeros(Hs, Ws, nb, 'single');
blk = 50 * B;
k = 0;
for i0 = 1:blk:numel(h5r)
    r = h5r(i0:min(i0 + blk - 1, numel(h5r)));
    A = single(h5read(movie, '/mov', [1 1 r(1)], [H W numel(r)]));
    n = numel(r) / B;
    A = squeeze(mean(reshape(A, H, W, B, n), 3));
    for j = 1:n
        X(:, :, k + j) = imresize(A(:, :, j), [Hs Ws], 'box');
    end
    k = k + n;
end
F0 = mean(X, 3);
rawlim = prctile(F0(:), [0.5 99.9]) .* [0.9 1.15];

% ---- overlay: outlines + numbers, drawn once --------------------------------------------
ov = zeros(Hs, Ws, 3, 'uint8');
for c = 1:size(S, 3)
    f = imresize(S(:, :, c), [Hs Ws], 'bilinear');
    e = bwperim(f >= 0.3 * max(f(:)));
    rgb = vcolor(R.verdict{c});
    for ch = 1:3
        q = ov(:, :, ch);
        q(e) = rgb(ch);
        ov(:, :, ch) = q;
    end
    ov = insertText(ov, [R.col(c) * sc + 6, R.row(c) * sc - 18], sprintf('%d', c), 'FontSize', 12, ...
                    'BoxOpacity', 0, 'TextColor', rgb);
end
omask = any(ov > 0, 3);

% ---- display scaling of the dF/F + high-pass panel (from a sample of frames) --------------
sig_hp = max(1, hp * radius * sc / 2.5);           % ~EXTRACT's spatial high-pass scale
smp = round(linspace(1, nb, min(nb, 200)));
vals = [];
for j = smp
    d = dff_hp(X(:, :, j), F0, sig_hp, radius * sc);
    c0 = d(round(Hs * 0.15):round(Hs * 0.85), round(Ws * 0.15):round(Ws * 0.85));
    vals = [vals; c0(:)]; %#ok<AGROW>
end
dlim = prctile(vals, [3 99.9]);

% ---- write --------------------------------------------------------------------------------
vw = VideoWriter(out_mp4, 'MPEG-4');
vw.FrameRate = vfps;
vw.Quality = 92;
open(vw);
prev = [];
for j = 1:nb
    d = dff_hp(X(:, :, j), F0, sig_hp, radius * sc);
    if isempty(prev), dd = d; else, dd = (d + prev) / 2; end   % 2-frame display mean
    prev = d;
    a = gray8(X(:, :, j), rawlim);
    b = uint8(255 * ind2rgb(round(1 + 255 * min(max((dd - dlim(1)) / diff(dlim), 0), 1)), hot(256)));
    a(repmat(omask, 1, 1, 3)) = ov(repmat(omask, 1, 1, 3));
    b(repmat(omask, 1, 1, 3)) = ov(repmat(omask, 1, 1, 3));
    fr = [a, 255 * ones(Hs, 6, 3, 'uint8'), b];
    head = 255 * ones(30, size(fr, 2), 3, 'uint8');
    tt = t(1) + ((j - 1) * B + B / 2) / fps;
    head = insertText(head, [6 4], sprintf(['t = %6.1f s   (%.0fx real time)      LEFT motion-corrected movie' ...
        '      RIGHT dF/F + high-pass (what EXTRACT sees)      green cell-like  yellow check  red not a cell'], ...
        tt, vfps * B / fps), 'FontSize', 13, 'BoxOpacity', 0, 'TextColor', 'black');
    writeVideo(vw, [head; fr]);
end
close(vw);
out = struct('file', out_mp4, 'n_frames', nb, 'speed', vfps * B / fps);
fprintf('fov_review_video: %d frames (%.0fx real time) -> %s\n', nb, out.speed, out_mp4);
end


function d = dff_hp(x, F0, sig, r)
d = (x - F0) ./ max(F0, 1);
d = d - imgaussfilt(d, sig);
d = imgaussfilt(d, max(0.8, r / 5));
end


function a = gray8(x, lim)
a = uint8(255 * min(max((x - lim(1)) / max(lim(2) - lim(1), eps), 0), 1));
a = repmat(a, 1, 1, 3);
end


function v = getopt(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end


function c = vcolor(v)
switch v
    case 'cell', c = [40 220 60];
    case 'check', c = [255 200 0];
    otherwise, c = [240 50 50];
end
end
