function out = fov_projection_images(mc_h5, frames_csv, out_mat, opts)
%FOV_PROJECTION_IMAGES  Whole-field summary images of a motion-corrected 1p movie, for cell
%   review: mean, max (background removed), max of dF/F + high-pass, local correlation (Cn)
%   and peak-to-noise (PNR), all at full resolution, over the whole movie binned in time.
%
%   out = fov_projection_images(mc_h5, frames_csv, out_mat, opts)
%   opts: bin_time (4), radius (10, px), window_s ([] = all), recording_id ('' = all rows)
%
%   Saves out_mat with mean_img, max_bgsub, max_dff, std_dff, Cn, PNR (single, H x W) and
%   the settings. Background = Gaussian blur of 3 x radius; dF/F uses the per-pixel mean.

if nargin < 4, opts = struct(); end
B = getopt(opts, 'bin_time', 4);
r = getopt(opts, 'radius', 10);
win = getopt(opts, 'window_s', []);
ft = readtable(frames_csv, 'TextType', 'char');
rid = getopt(opts, 'recording_id', '');
if ~isempty(rid)
    ft = ft(strcmp(cellstr(string(ft.recording_id)), rid), :);
end
v = ft.valid == 1;
fps = (ft.isx_frame(find(v, 1, 'last')) - ft.isx_frame(find(v, 1))) / ...
      ((ft.tsc_us(find(v, 1, 'last')) - ft.tsc_us(find(v, 1))) / 1e6);
rows = ft.h5_row(v);
t = ft.isx_frame(v) / fps;
if ~isempty(win), rows = rows(t >= win(1) & t < win(2)); end
nb = floor(numel(rows) / B);
rows = rows(1:nb * B);
hi = h5info(mc_h5, '/mov');
H = hi.Dataspace.Size(1); W = hi.Dataspace.Size(2);
X = zeros(H, W, nb, 'single');
k = 0;
for i0 = 1:50 * B:numel(rows)
    rr = rows(i0:min(i0 + 50 * B - 1, numel(rows)));
    A = single(h5read(mc_h5, '/mov', [1 1 rr(1)], [H W numel(rr)]));
    n = numel(rr) / B;
    X(:, :, k + 1:k + n) = squeeze(mean(reshape(A, H, W, B, n), 3));
    k = k + n;
end
mean_img = mean(X, 3);
max_bgsub = -inf(H, W, 'single');
max_dff = -inf(H, W, 'single');
s1 = zeros(H, W, 'single'); s2 = s1;
D = zeros(H, W, nb, 'single');
for j = 1:nb
    f = X(:, :, j);
    max_bgsub = max(max_bgsub, f - imgaussfilt(f, 3 * r));
    d = (f - mean_img) ./ max(mean_img, 1);
    d = imgaussfilt(d, max(1, r / 4)) - imgaussfilt(d, 3 * r);
    D(:, :, j) = d;
    max_dff = max(max_dff, d);
    s1 = s1 + d; s2 = s2 + d .^ 2;
end
std_dff = sqrt(max(s2 / nb - (s1 / nb) .^ 2, 0));
clear X
% Cn / PNR on the dF/F + high-pass movie, 2x binned for speed, back to full size
h2 = floor(H / 2); w2 = floor(W / 2);
Z = D(1:2 * h2, 1:2 * w2, :);
Z = squeeze(mean(mean(reshape(Z, 2, h2, 2, w2, nb), 1), 3));
clear D
Z = Z - mean(Z, 3);
sd = std(Z, 0, 3) + eps;
noise = 1.4826 * median(abs(diff(Z, 1, 3)), 3) / sqrt(2) + eps;
PNR = imresize(max(movmean(Z, 3, 3), [], 3) ./ noise, [H W]);
Z = Z ./ sd;
Cn = zeros(h2, w2, 'single');
for dy = -1:1
    for dx = -1:1
        if dy == 0 && dx == 0, continue; end
        Cn = Cn + mean(Z .* circshift(Z, [dy dx 0]), 3) / 8;
    end
end
Cn = imresize(max(Cn, 0), [H W]);
save(out_mat, 'mean_img', 'max_bgsub', 'max_dff', 'std_dff', 'Cn', 'PNR', 'B', 'r', 'fps', '-v7');
out = struct('file', out_mat);
fprintf('fov_projection_images: %d binned frames -> %s\n', nb, out_mat);
end


function v = getopt(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end
