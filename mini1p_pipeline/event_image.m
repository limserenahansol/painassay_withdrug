function [img, r] = event_image(h5, ft, x, fpimg, row, col, radius, fps)
%EVENT_IMAGE  What the movie itself shows at one component's own largest transients.
%   [img, r] = event_image(h5, ft, x, fpimg, row, col, radius, fps)
%   h5 movie (h5:/mov), ft frame table (h5_row per trace row), x the trace (one row per
%   frame-table row), fpimg the footprint, row/col its centroid. img = mean over the (up to
%   10) largest events of [movie at peak +-1 frame] - [movie 2.0-0.7 s before], spatially
%   high-passed, in a window around the component; r = its correlation with the footprint.
%   A soma gives a round spot inside its outline (r high); a vessel, an edge or background
%   noise does not. Used by extract_cell_review and extract_param_sweep.
img = []; r = NaN;
half = max(25, round(4 * radius));
if ~isfinite(row), return; end
x = x(:);
ok = isfinite(x);
if nnz(ok) < 50, return; end
x(~ok) = median(x(ok));
xm = movmean(medfilt1(x, 3), max(1, round(0.2 * fps)));   % as cell_qc_metrics: no noise spikes
xm = xm - median(xm);
noise = 1.4826 * median(abs(diff(xm) - median(diff(xm)))) / sqrt(2) + eps;
ws = warning('off', 'signal:findpeaks:largeMin');
[pk, loc] = findpeaks(xm, 'MinPeakHeight', 4 * noise, 'MinPeakDistance', max(1, round(3 * fps)));
warning(ws);
if isempty(pk), return; end
[~, o] = sort(pk, 'descend');
loc = loc(o(1:min(10, end)));
b0 = round(2.0 * fps); b1 = round(0.7 * fps);
hi = h5info(h5, '/mov');
H = hi.Dataspace.Size(1); W = hi.Dataspace.Size(2);
r0 = round(row); c0 = round(col);
rr = max(1, r0 - half):min(H, r0 + half);
cl = max(1, c0 - half):min(W, c0 + half);
acc = zeros(numel(rr), numel(cl));
n = 0;
for k = 1:numel(loc)
    p = loc(k);
    if p - b0 < 1 || p + 1 > numel(x), continue; end
    hr = ft.h5_row(p - b0:p + 1);
    if any(~isfinite(hr)) || any(diff(hr) ~= 1), continue; end   % dropped frame in between
    blk = double(h5read(h5, '/mov', [rr(1) cl(1) hr(1)], [numel(rr) numel(cl) numel(hr)]));
    peak = mean(blk(:, :, end - 2:end), 3);
    base = mean(blk(:, :, 1:b0 - b1 + 1), 3);
    acc = acc + (peak - base);
    n = n + 1;
end
if n == 0, return; end
d = acc / n;
d = imgaussfilt(d, max(1, radius / 4)) - imgaussfilt(d, 2 * radius);
img = nan(2 * half + 1);
img(rr - r0 + half + 1, cl - c0 + half + 1) = d;
img(~isfinite(img)) = 0;
f = fpimg(rr, cl);
r = corr(d(:), double(f(:)));
end
