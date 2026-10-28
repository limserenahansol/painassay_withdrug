function out = cell_review_clips(extract_mat, review_csv, out_dir, opts)
%CELL_REVIEW_CLIPS  A video and a filmstrip per EXTRACT component, to decide keep / drop by
%   watching the movie itself: does a soma light up inside the outline when the trace says
%   it fires?
%
%   out = cell_review_clips(extract_mat, review_csv, out_dir, opts)
%
%   extract_mat : <tag>_extract.mat of extract_1p (its config names the movie h5)
%   review_csv  : <tag>_cell_review.csv of extract_cell_review (ids, centroids, verdicts)
%   out_dir     : default <folder of review_csv>\clips
%   opts        : n_events (4), pre_s (2), post_s (4), movie_h5 (from the EXTRACT config)
%
%   Writes, per component k
%     clips\cell_<k>.mp4   its n_events largest events one after another, real time (fps of
%                          the movie): LEFT the motion-corrected movie (fixed contrast),
%                          RIGHT the local dF/F (each pixel relative to its own 2.0-0.7 s
%                          before the event), both with the footprint outline (green
%                          cell-like, yellow check, red not a cell); UNDER them the trace
%                          of that window with a moving cursor
%   and <tag>_filmstrips_NN.png: per component its largest event as 9 local-dF/F frames,
%   -1 s to +3 s every 0.5 s, outline drawn, 6 components per page.

if nargin < 3 || isempty(out_dir), out_dir = fullfile(fileparts(review_csv), 'clips'); end
if nargin < 4, opts = struct(); end
n_ev = getopt(opts, 'n_events', 4);
pre_s = getopt(opts, 'pre_s', 2);
post_s = getopt(opts, 'post_s', 4);
if ~isfolder(out_dir), mkdir(out_dir); end
m = load(extract_mat);
movie = getopt(opts, 'movie_h5', m.config.extract_1p.h5);
[~, stem] = fileparts(extract_mat);
tag = regexprep(stem, '_extract$', '');
R = readtable(review_csv, 'TextType', 'char');
S = full(m.spatial_weights);
T = double(m.temporal_raw);
ft = m.frame_table;
fps = double(m.fps);
radius = double(m.config.extract_1p.avg_cell_radius);
hi = h5info(movie, '/mov');
H = hi.Dataspace.Size(1); W = hi.Dataspace.Size(2);
half = max(30, round(4 * radius));
P = 300;                                   % panel px
pre = round(pre_s * fps); post = round(post_s * fps);
b0 = round(2.0 * fps); b1 = round(0.7 * fps);
strips = {};
fprintf('cell_review_clips %s: %d components, movie %s\n', tag, size(S, 3), movie);
for c = 1:size(S, 3)
    col = vcolor(R.verdict{c});
    r0 = round(R.row(c)); c0 = round(R.col(c));
    rr = max(1, r0 - half):min(H, r0 + half);
    cl = max(1, c0 - half):min(W, c0 + half);
    fp = S(rr, cl, c);
    edge = bwperim(imresize(fp >= 0.3 * max(S(:, :, c), [], 'all'), [P P], 'nearest'));
    edge = imdilate(edge, ones(2));
    % events on the 0.2-s smoothed trace (as cell_qc_metrics)
    x = T(:, c);
    ok = isfinite(x);
    x(~ok) = median(x(ok));
    xm = movmean(medfilt1(x, 3), max(1, round(0.2 * fps)));
    xm = xm - median(xm);
    noise = 1.4826 * median(abs(diff(xm) - median(diff(xm)))) / sqrt(2) + eps;
    ws = warning('off', 'signal:findpeaks:largeMin');
    [pk, loc] = findpeaks(xm, 'MinPeakHeight', 3 * noise, 'MinPeakDistance', round(6 * fps));
    warning(ws);
    [~, o] = sort(pk, 'descend');
    loc = loc(o);
    loc = loc(loc > pre & loc + post <= numel(x));
    keep = false(size(loc));
    for i = 1:numel(loc)
        hr = ft.h5_row(loc(i) - pre:loc(i) + post);
        keep(i) = all(isfinite(hr)) && all(diff(hr) == 1);
    end
    loc = loc(keep);
    loc = loc(1:min(n_ev, end));
    if isempty(loc)
        fprintf('  #%d: no event to show\n', c);
        continue;
    end
    % read every event block once
    blocks = cell(numel(loc), 1);
    for i = 1:numel(loc)
        hr = ft.h5_row(loc(i) - pre:loc(i) + post);
        blocks{i} = double(h5read(movie, '/mov', [rr(1) cl(1) hr(1)], [numel(rr) numel(cl) numel(hr)]));
    end
    allraw = cat(3, blocks{:});
    rlim = prctile(allraw(:), [1 99.8]);
    D = cell(numel(loc), 1);
    pk_vals = [];
    for i = 1:numel(loc)
        base = mean(blocks{i}(:, :, pre - b0 + 1:pre - b1 + 1), 3);
        d = (blocks{i} - base) ./ max(base, 1);
        d = movmean(d, 3, 3);                % display only: 0.1 s and sigma 1.5 px, far
        for k = 1:size(d, 3), d(:, :, k) = imgaussfilt(d(:, :, k), 1.5); end   % shorter than a transient
        D{i} = d;
        pk_vals = [pk_vals; reshape(d(:, :, pre:min(end, pre + round(fps))), [], 1)]; %#ok<AGROW>
    end
    dlim = [-0.5, 1] * max(prctile(pk_vals, 99.5), 0.02);
    % ---- video ----
    vw = VideoWriter(fullfile(out_dir, sprintf('cell_%02d.mp4', c)), 'MPEG-4');
    vw.FrameRate = round(fps);
    vw.Quality = 90;
    open(vw);
    tw = (-pre:post) / fps;
    for i = 1:numel(loc)
        xs = x(loc(i) - pre:loc(i) + post);
        for k = 1:numel(tw)
            a = to_rgb(imresize(blocks{i}(:, :, k), [P P], 'bilinear'), rlim, gray(256));
            b = to_rgb(imresize(D{i}(:, :, k), [P P], 'bilinear'), dlim, hot(256));
            a = paint(a, edge, col); b = paint(b, edge, col);
            tr = trace_strip(xs, k, 2 * P + 8, 110, tw);
            top = [a, 255 * ones(P, 8, 3, 'uint8'), b];
            fr = [top; tr];
            head = 255 * ones(34, size(fr, 2), 3, 'uint8');
            head = insertText(head, [4 6], sprintf(['#%d  %s   event %d/%d at %.1f s   t = %+.2f s   ' ...
                '(left: movie   right: local dF/F)'], c, strrep(R.verdict{c}, '_', ' '), i, numel(loc), ...
                (loc(i) - 1) / fps, tw(k)), 'FontSize', 13, 'BoxOpacity', 0, 'TextColor', 'black');
            writeVideo(vw, [head; fr]);
        end
    end
    close(vw);
    % ---- filmstrip of the largest event ----
    ts = -1:0.5:3;
    row = {};
    for t0 = ts
        k = min(max(round(t0 * fps) + pre + 1, 1), size(D{1}, 3));
        f = to_rgb(imresize(D{1}(:, :, k), [150 150], 'bilinear'), dlim, hot(256));
        e2 = imdilate(bwperim(imresize(fp >= 0.3 * max(S(:, :, c), [], 'all'), [150 150], 'nearest')), ones(1));
        f = paint(f, e2, col);
        f = insertText(f, [2 2], sprintf('%+.1f s', t0), 'FontSize', 11, 'BoxOpacity', 0.4, 'TextColor', 'white');
        row{end + 1} = f; %#ok<AGROW>
        row{end + 1} = 255 * ones(150, 4, 3, 'uint8'); %#ok<AGROW>
    end
    strip = [row{:}];
    lab = 255 * ones(24, size(strip, 2), 3, 'uint8');
    lab = insertText(lab, [2 2], sprintf('#%d  %s   largest event at %.1f s   (local dF/F, same colour scale as its video)', ...
          c, strrep(R.verdict{c}, '_', ' '), (loc(1) - 1) / fps), 'FontSize', 12, 'BoxOpacity', 0, 'TextColor', 'black');
    strips{end + 1} = [lab; strip; 255 * ones(10, size(strip, 2), 3, 'uint8')]; %#ok<AGROW>
    fprintf('  #%d: %d events -> cell_%02d.mp4\n', c, numel(loc), c);
end
pages = {};
for p0 = 1:6:numel(strips)
    pg = vertcat(strips{p0:min(p0 + 5, end)});
    f = fullfile(fileparts(out_dir), sprintf('%s_filmstrips_%02d.png', tag, ceil(p0 / 6)));
    imwrite(pg, f);
    pages{end + 1} = f; %#ok<AGROW>
end
out = struct('clips_dir', out_dir, 'filmstrips', {pages});
fprintf('cell_review_clips %s: done -> %s\n', tag, out_dir);
end


function v = getopt(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end


function c = vcolor(v)
switch v
    case 'cell', c = [40 190 60];
    case 'check', c = [255 190 0];
    otherwise, c = [230 40 40];
end
end


function a = to_rgb(x, lim, cmap)
i = round(1 + 255 * min(max((x - lim(1)) / max(lim(2) - lim(1), eps), 0), 1));
a = uint8(255 * ind2rgb(i, cmap));
end


function a = paint(a, e, rgb)
for ch = 1:3
    q = a(:, :, ch);
    q(e) = rgb(ch);
    a(:, :, ch) = q;
end
end


function img = trace_strip(xs, k, Wd, Ht, tw)
img = 255 * ones(Ht, Wd, 3, 'uint8');
lo = min(xs); hi_ = max(xs);
if hi_ <= lo, hi_ = lo + eps; end
px = round(linspace(6, Wd - 6, numel(xs)));
py = round(Ht - 10 - (xs - lo) / (hi_ - lo) * (Ht - 30));
pts = reshape([px(:)'; py(:)'], 1, []);
img = insertShape(img, 'Line', pts, 'Color', 'black', 'LineWidth', 1);
z = px(find(tw >= 0, 1));
img = insertShape(img, 'Line', [z 4 z Ht - 4], 'Color', [150 150 150], 'LineWidth', 1);
img = insertShape(img, 'Line', [px(k) 4 px(k) Ht - 4], 'Color', 'red', 'LineWidth', 2);
img = insertText(img, [6 2], 'trace (dF/F, least squares); grey = event peak, red = now', ...
                 'FontSize', 11, 'BoxOpacity', 0, 'TextColor', 'black');
end
