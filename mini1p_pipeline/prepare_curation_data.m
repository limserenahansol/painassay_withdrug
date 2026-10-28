function out = prepare_curation_data(extract_mat, review_csv, projections_mat, out_mat, opts)
%PREPARE_CURATION_DATA  Everything cell_curation_gui needs, in one file that loads fast:
%   a 1-2 min movie window (dF/F + high-pass and raw, binned 4x, half resolution, uint8),
%   the whole-movie projections, every component's outline and centroid, the traces and the
%   review table.
%
%   out = prepare_curation_data(extract_mat, review_csv, projections_mat, out_mat, opts)
%   opts: window_s ([] = the 120 s with the most activity), win_len_s (120), bin_time (4),
%         scale (0.5), movie_h5 (from the EXTRACT config)

if nargin < 5, opts = struct(); end
L = getopt(opts, 'win_len_s', 120);
B = getopt(opts, 'bin_time', 4);
sc = getopt(opts, 'scale', 0.5);
m = load(extract_mat);
movie = getopt(opts, 'movie_h5', m.config.extract_1p.h5);
R = readtable(review_csv, 'TextType', 'char');
P = load(projections_mat);
S = full(m.spatial_weights);
T = double(m.temporal_raw);
ft = m.frame_table;
fps = double(m.fps);
radius = double(m.config.extract_1p.avg_cell_radius);
C = size(S, 3);
v = find(ft.valid == 1);
t = (v - v(1)) / fps;

% ---- window: the L seconds where the cells are most active, unless given -------------------
win = getopt(opts, 'window_s', []);
if isempty(win)
    Z = (T(v, :) - median(T(v, :), 1, 'omitnan')) ./ (1.4826 * mad(T(v, :), 1, 1) + eps);
    act = sum(max(Z, 0), 2, 'omitnan');
    k = min(max(1, round(L * fps)), numel(act));   % a movie shorter than L: all of it
    s = movsum(act, [0 k - 1]);
    s(numel(s) - k + 2:end) = -inf;
    [~, i0] = max(s);
    win = [t(i0), t(i0) + L];
end
sel = v(t >= win(1) & t < win(2));
h5r = ft.h5_row(sel);
nb = floor(numel(h5r) / B);
h5r = h5r(1:nb * B);
hi = h5info(movie, '/mov');
H = hi.Dataspace.Size(1); W = hi.Dataspace.Size(2);
Hs = round(H * sc); Ws = round(W * sc);
% at 0.5x bin exactly 2 x 2 (an odd height such as 5937's 781 rows made imresize slip one row of
% the sensor's alternating odd/even row gain halfway down -> a false full-width line in the
% display, 2026-10-21); the last row / column of an odd size is dropped
exact2 = abs(sc - 0.5) < 1e-9;
if exact2, Hs = floor(H / 2); Ws = floor(W / 2); end
X = zeros(Hs, Ws, nb, 'single');
k = 0;
for i0 = 1:50 * B:numel(h5r)
    rr = h5r(i0:min(i0 + 50 * B - 1, numel(h5r)));
    A = single(h5read(movie, '/mov', [1 1 rr(1)], [H W numel(rr)]));
    n = numel(rr) / B;
    A = squeeze(mean(reshape(A, H, W, B, n), 3));
    for j = 1:n, X(:, :, k + j) = shrink(A(:, :, j), Hs, Ws, exact2); end
    k = k + n;
end
F0 = shrink(single(P.mean_img), Hs, Ws, exact2);
rawlim = prctile(F0(:), [0.5 99.9]) .* [0.9 1.15];
movie_raw = uint8(255 * min(max((X - rawlim(1)) / diff(rawlim), 0), 1));
D = zeros(size(X), 'single');
rs = radius * sc;
for j = 1:nb
    d = (X(:, :, j) - F0) ./ max(F0, 1);
    d = imgaussfilt(d, max(0.8, rs / 4)) - imgaussfilt(d, 3 * rs);
    D(:, :, j) = d;
end
D = movmean(D, 2, 3);
ctr = D(round(Hs * 0.15):round(Hs * 0.85), round(Ws * 0.15):round(Ws * 0.85), :);
dlim = prctile(ctr(:), [3 99.9]);
movie_dff = uint8(255 * min(max((D - dlim(1)) / diff(dlim), 0), 1));
movie_t = win(1) + ((0:nb - 1) * B + B / 2) / fps;
clear X D ctr

% ---- projections at display size ------------------------------------------------------
proj = struct();
proj.max_dff = imresize(P.max_dff, [Hs Ws]);
proj.mean = imresize(P.mean_img, [Hs Ws]);
proj.max_bgsub = imresize(P.max_bgsub, [Hs Ws]);
proj.cn_pnr = imresize(P.Cn .* P.PNR, [Hs Ws]);
proj.cn = imresize(P.Cn, [Hs Ws]);

% ---- outlines, centroids --------------------------------------------------------------
outline = cell(C, 1);
for c = 1:C
    f = S(:, :, c);
    Bd = bwboundaries(f >= 0.3 * max(f(:)));
    for b = 1:numel(Bd), Bd{b} = Bd{b} * sc; end
    outline{c} = Bd;
end
cen = [R.row(:), R.col(:)] * sc;

tt = (0:size(T, 1) - 1)' / fps;              % trace time (frame-table rows)
Tb = T;                                       % full rate traces
save(out_mat, 'movie_raw', 'movie_dff', 'movie_t', 'win', 'proj', 'outline', 'cen', 'Tb', 'tt', ...
     'fps', 'sc', 'radius', 'review_csv', 'extract_mat', 'dlim', 'rawlim', '-v7.3');
out = struct('file', out_mat, 'window_s', win, 'n_frames', nb, 'n_cells', C);
fprintf('prepare_curation_data: %d cells, window %.0f-%.0f s (%d frames) -> %s\n', C, win(1), win(2), nb, out_mat);
end


function B = shrink(A, Hs, Ws, exact2)
if exact2
    A = A(1:2 * Hs, 1:2 * Ws);
    B = squeeze(mean(mean(reshape(A, 2, Hs, 2, Ws), 1), 3));
else
    B = imresize(A, [Hs Ws], 'box');
end
end


function v = getopt(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end
