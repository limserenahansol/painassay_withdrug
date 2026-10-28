function out = extract_cell_review(extract_mat, out_dir, opts)
%EXTRACT_CELL_REVIEW  Cell-by-cell QC of a full-day EXTRACT result, as in the mini2p
%   pipeline: numbered map of every component, one row per component with its footprint
%   and its whole-day trace, and a keep / drop table pre-filled with a suggestion.
%   YOU decide; apply_cell_keep then writes the kept-cells file the analysis uses.
%
%   out = extract_cell_review(extract_mat, out_dir)
%   out = extract_cell_review(extract_mat, out_dir, opts)
%   extract_cell_review('selftest')
%
%   extract_mat : <mouse>_d<day>_extract.mat of extract_1p
%   out_dir     : review folder (default <folder of extract_mat>\cell_review)
%   opts        : qc (struct of cell_qc_metrics thresholds), trace ('raw' | 'extract'),
%                 rows_per_page (8)
%
%   Writes
%     <tag>_cells_map.png        every component outlined and numbered on the EXTRACT
%                                summary image (green cell-like, yellow check, red not)
%     <tag>_cells_page_NN.png    rows_per_page components per page: zoom on the summary
%                                image with the outline | footprint | whole-day trace
%                                (pre | post) with the average of its largest transients
%     <tag>_cell_review.csv      id, centroid, every QC number, verdict, reasons,
%                                keep (1 / 0, pre-filled: cell 1, check 1, not_cell 0),
%                                note.  EDIT keep (and note) here.
%   An existing <tag>_cell_review.csv is never overwritten (it holds your decisions): the
%   new table goes to <tag>_cell_review_NEW.csv and a warning says so.

if nargin == 1 && strcmpi(extract_mat, 'selftest')
    out = run_selftest();
    return;
end
if nargin < 2 || isempty(out_dir)
    out_dir = fullfile(fileparts(extract_mat), 'cell_review');
end
if nargin < 3, opts = struct(); end
qc_opts = getopt(opts, 'qc', struct());
trace_kind = getopt(opts, 'trace', 'raw');
per_page = getopt(opts, 'rows_per_page', 8);
if ~isfolder(out_dir), mkdir(out_dir); end

m = load(extract_mat);
[~, stem] = fileparts(extract_mat);
tag = regexprep(stem, '_extract$', '');
S = full(m.spatial_weights);
[H, W, C] = size(S);
if strcmp(trace_kind, 'extract'), T = double(m.temporal_extract); else, T = double(m.temporal_raw); end
fps = double(m.fps);
radius = double(m.config.extract_1p.avg_cell_radius);
bg = background_image(m, H, W);

Q = cell_qc_metrics(S, T, radius, fps, qc_opts);
% PROOF from the movie itself: the average of the movie at this component's own largest
% transients (peak minus the 2 s before). A soma lights up as a round spot inside its
% outline; a vessel, an edge or background does not. event_image_corr = how well that
% activity image matches the footprint (1 = identical).
movie = getopt(opts, 'movie_h5', '');
if isempty(movie) && isfield(m, 'config') && isfield(m.config, 'extract_1p') && ...
        isfield(m.config.extract_1p, 'h5')
    movie = m.config.extract_1p.h5;
end
EI = cell(C, 1);
Q.event_image_corr = NaN(C, 1);
if ~isempty(movie) && isfile(movie)
    fprintf('extract_cell_review: event-triggered images from %s\n', movie);
    for c = 1:C
        [EI{c}, Q.event_image_corr(c)] = event_image(movie, m.frame_table, T(:, c), ...
            S(:, :, c), Q.row(c), Q.col(c), radius, fps);
        if strcmp(Q.verdict{c}, 'cell') && ~(Q.event_image_corr(c) >= 0.5)
            Q.verdict{c} = 'check';
            Q.reasons{c} = sprintf('movie at its events does not match the footprint (r %.2f)', ...
                                   Q.event_image_corr(c));
        elseif strcmp(Q.verdict{c}, 'check') && Q.event_image_corr(c) < 0.3
            Q.verdict{c} = 'not_cell';
            Q.reasons{c} = sprintf('%s; movie at its events shows nothing (r %.2f)', Q.reasons{c}, ...
                                   Q.event_image_corr(c));
        end
        % Learned from the user's review of 5941 SBI (2026-10-20, 91 candidates: every
        % cell/check kept except #10 and #11; these two rules remove exactly those two and
        % none of the 47 kept): weak movie proof AND a bad shape, or weak proof AND only
        % flash-like events (shorter than a GCaMP transient).
        r_ev = Q.event_image_corr(c);
        bad_shape = Q.area_ratio(c) > 2.5 || Q.n_pieces(c) > 1 || Q.axis_ratio(c) > 2.5;
        flash = ~Q.ca_like(c) && Q.decay_half_s(c) <= 0.10;
        if ~strcmp(Q.verdict{c}, 'not_cell') && ((r_ev < 0.35 && bad_shape) || (r_ev < 0.45 && flash))
            Q.verdict{c} = 'not_cell';
            Q.reasons{c} = sprintf('%s; weak movie proof (r %.2f) with %s', Q.reasons{c}, r_ev, ...
                                   tern(bad_shape && r_ev < 0.35, 'a bad shape', 'flash-like events only'));
        end
    end
else
    fprintf('extract_cell_review: movie not found, no event-triggered images\n');
end
keep = double(~strcmp(Q.verdict, 'not_cell'));
note = repmat({''}, C, 1);
R = [Q, table(keep, note)];
csv = fullfile(out_dir, [tag '_cell_review.csv']);
if isfile(csv)
    csv_new = fullfile(out_dir, [tag '_cell_review_NEW.csv']);
    warning('extract_cell_review:exists', ['%s exists and holds your decisions; it is NOT ' ...
            'overwritten. The new table is %s'], csv, csv_new);
    writetable(R, csv_new);
else
    writetable(R, csv);
end

% recording boundaries for the trace panels
rid = cellstr(string(m.frame_table.recording_id));
[ids, first] = unique(rid, 'stable');

map_png = fullfile(out_dir, [tag '_cells_map.png']);
plot_map(map_png, bg, S, Q, tag);
pages = {};
for p = 1:ceil(C / per_page)
    cc = (p - 1) * per_page + 1:min(p * per_page, C);
    pages{end + 1} = fullfile(out_dir, sprintf('%s_cells_page_%02d.png', tag, p)); %#ok<AGROW>
    plot_page(pages{end}, bg, S, T, Q, cc, radius, fps, ids, first, tag, EI);
end
n_v = @(v) nnz(strcmp(Q.verdict, v));
fprintf(['extract_cell_review %s: %d components -> %d cell-like, %d check, %d not_cell ' ...
         '(suggestion). Edit keep in %s, then run apply_cell_keep.\n'], tag, C, n_v('cell'), ...
         n_v('check'), n_v('not_cell'), csv);
out = struct('tag', tag, 'csv', csv, 'map_png', map_png, 'pages', {pages}, ...
             'n_components', C, 'n_cell', n_v('cell'), 'n_check', n_v('check'), ...
             'n_not_cell', n_v('not_cell'));
end


function v = getopt(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end


function bg = background_image(m, H, W)
% EXTRACT's summary image (max of its preprocessed movie) shows somata far better than
% the raw mean of a 1p movie; it can be on EXTRACT's binned grid, so resize.
bg = [];
if isfield(m, 'extract_info') && isfield(m.extract_info, 'summary_image')
    bg = double(m.extract_info.summary_image);
end
if isempty(bg) || all(~isfinite(bg(:)))
    bg = double(m.max_image);
end
if ~isequal(size(bg), [H W])
    bg = imresize(bg, [H W], 'bilinear');
end
bg(~isfinite(bg)) = 0;
end


function c = vcolor(v)
switch v
    case 'cell', c = [0.15 0.75 0.25];
    case 'check', c = [1 0.75 0];
    otherwise, c = [0.9 0.15 0.15];
end
end


function plot_map(png, bg, S, Q, tag)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [20 20 1300 900]);
imagesc(bg); axis image off; colormap(gray); hold on;
v = prctile(bg(:), [1 99.7]);
if v(2) > v(1), caxis(v); end
for c = 1:size(S, 3)
    im = S(:, :, c);
    B = bwboundaries(im >= 0.3 * max(im(:)));
    col = vcolor(Q.verdict{c});
    for b = 1:numel(B)
        plot(B{b}(:, 2), B{b}(:, 1), '-', 'Color', col, 'LineWidth', 1);
    end
    if isfinite(Q.col(c))
        text(Q.col(c) + 4, Q.row(c) - 4, sprintf('%d', c), 'Color', col, 'FontSize', 8, ...
             'FontWeight', 'bold');
    end
end
title(sprintf('%s: %d components  (green cell-like %d, yellow check %d, red not a cell %d - suggestion)', ...
      strrep(tag, '_', '\_'), size(S, 3), nnz(strcmp(Q.verdict, 'cell')), ...
      nnz(strcmp(Q.verdict, 'check')), nnz(strcmp(Q.verdict, 'not_cell'))), 'FontWeight', 'normal');
exportgraphics(fig, png, 'Resolution', 140);
close(fig);
end


function plot_page(png, bg, S, T, Q, cc, radius, fps, ids, first, tag, EI)
n = numel(cc);
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [20 20 1800 210 * n]);
half = max(25, round(4 * radius));
[H, W] = size(bg);
nT = size(T, 1);
step = max(1, round(fps / 5));                % plot at ~5 Hz (mean of each bin)
nb = floor(nT / step);
tt = ((0:nb - 1) * step + step / 2) / fps / 60;
for i = 1:n
    c = cc(i);
    col = vcolor(Q.verdict{c});
    r0 = round(Q.row(c)); c0 = round(Q.col(c));
    if ~isfinite(r0), r0 = round(H / 2); c0 = round(W / 2); end
    rr = max(1, r0 - half):min(H, r0 + half);
    cl = max(1, c0 - half):min(W, c0 + half);
    im = S(:, :, c);
    % zoom on the background with the outline
    subplot(n, 7, (i - 1) * 7 + 1);
    z = bg(rr, cl);
    imagesc(z); axis image off; colormap(gca, gray); hold on;
    v = prctile(z(:), [1 99.5]);
    if v(2) > v(1), caxis(v); end
    B = bwboundaries(im(rr, cl) >= 0.3 * max(im(:)));
    for b = 1:numel(B)
        plot(B{b}(:, 2), B{b}(:, 1), '-', 'Color', col, 'LineWidth', 1.2);
    end
    title(sprintf('#%d  %s', c, strrep(Q.verdict{c}, '_', ' ')), 'Color', col, 'FontSize', 9);
    % footprint
    subplot(n, 7, (i - 1) * 7 + 2);
    imagesc(im(rr, cl)); axis image off; colormap(gca, parula);
    title(sprintf('size %.2fx  axis %.1f', Q.area_ratio(c), Q.axis_ratio(c)), 'FontSize', 8, ...
          'FontWeight', 'normal');
    % activity image: the movie at this component's own events
    subplot(n, 7, (i - 1) * 7 + 3);
    if ~isempty(EI{c})
        e = EI{c};
        imagesc(e); axis image off; colormap(gca, gray); hold on;
        q = prctile(e(:), [1 99.7]);
        if q(2) > q(1), clim(q); end
        hw = (size(e, 1) - 1) / 2;
        cr = round(Q.row(c)) - hw:round(Q.row(c)) + hw;
        cc2 = round(Q.col(c)) - hw:round(Q.col(c)) + hw;
        okr = cr >= 1 & cr <= H; okc = cc2 >= 1 & cc2 <= W;
        fp = zeros(size(e));
        fp(okr, okc) = im(cr(okr), cc2(okc));
        B2 = bwboundaries(fp >= 0.3 * max(fp(:)));
        for b = 1:numel(B2)
            plot(B2{b}(:, 2), B2{b}(:, 1), '-', 'Color', col, 'LineWidth', 1);
        end
        title(sprintf('movie at its events  r %.2f', Q.event_image_corr(c)), 'FontSize', 8, ...
              'FontWeight', 'normal');
    else
        axis off; title('no movie / no events', 'FontSize', 8, 'FontWeight', 'normal');
    end
    % whole-day trace
    subplot(n, 7, (i - 1) * 7 + (4:6));
    x = T(1:nb * step, c);
    x = mean(reshape(x, step, nb), 1, 'omitnan');
    plot(tt, x, 'k-', 'LineWidth', 0.5); hold on;
    for k = 2:numel(first)
        xline((first(k) - 1) / fps / 60, 'b-', strrep(ids{k}, '_', '\_'), 'FontSize', 7, ...
              'LabelVerticalAlignment', 'top');
    end
    yl = prctile(x, [0.2 99.9]);
    if all(isfinite(yl)) && yl(2) > yl(1), ylim(yl + [-0.1 0.1] * diff(yl)); end
    xlim([0 max(tt(end), eps)]);
    ylabel('dF/F'); box off;
    if i == n, xlabel('time in the day movie (min)'); end
    title(sprintf('SNR %.1f   skew %.2f   %d events   %s', Q.snr(c), Q.skew(c), Q.n_events(c), ...
          Q.reasons{c}), 'FontSize', 8, 'FontWeight', 'normal', 'Interpreter', 'none');
    % average of the largest transients
    subplot(n, 7, (i - 1) * 7 + 7);
    [a, ta] = top_event_average(T(:, c), fps);
    if ~isempty(a)
        plot(ta, a, '-', 'Color', col, 'LineWidth', 1.4); hold on; xline(0, 'k:');
        xlim([ta(1) ta(end)]);
    end
    box off;
    title(sprintf('rise %.2f s  decay %.2f s', Q.rise_s(c), Q.decay_half_s(c)), 'FontSize', 8, ...
          'FontWeight', 'normal');
    if i == n, xlabel('s from peak'); end
end
sgtitle(sprintf('%s  components %d-%d  (verdict = suggestion; decide in the review csv)', ...
        strrep(tag, '_', '\_'), cc(1), cc(end)), 'FontSize', 10);
exportgraphics(fig, png, 'Resolution', 100);
close(fig);
end


function [a, ta] = top_event_average(x, fps)
a = []; ta = [];
x = x(:);
ok = isfinite(x);
if nnz(ok) < 20, return; end
x(~ok) = median(x(ok));
xm = medfilt1(x, 3) - median(x);
noise = 1.4826 * median(abs(diff(x) - median(diff(x)))) / sqrt(2) + eps;
ws = warning('off', 'signal:findpeaks:largeMin');
[pk, loc] = findpeaks(xm, 'MinPeakHeight', 4 * noise, 'MinPeakDistance', max(1, round(2 * fps)));
warning(ws);
if isempty(pk), return; end
[~, o] = sort(pk, 'descend');
loc = loc(o(1:min(10, end)));
pre = round(2 * fps); post = round(5 * fps);
loc = loc(loc > pre & loc + post <= numel(xm));
if isempty(loc), return; end
A = zeros(numel(loc), pre + post + 1);
for i = 1:numel(loc), A(i, :) = xm(loc(i) - pre:loc(i) + post); end
a = mean(A, 1);
ta = (-pre:post) / fps;
end


% =====================================================================================
function out = run_selftest()
root = fullfile(tempdir, 'extract_cell_review_selftest');
if isfolder(root), rmdir(root, 's'); end
mkdir(root);
rng(5);
H = 120; W = 140; fps = 20; n = 3000; C = 6; r = 5;
[xx, yy] = meshgrid(1:W, 1:H);
S = zeros(H, W, C, 'single');
cen = [30 30; 30 90; 80 40; 80 100; 60 70; 100 120];
for c = 1:5
    S(:, :, c) = exp(-((xx - cen(c, 2)) .^ 2 + (yy - cen(c, 1)) .^ 2) / (2 * (0.6 * r) ^ 2));
end
S(:, :, 6) = exp(-((yy - 0.3 * xx - 10) .^ 2) / (2 * 2 ^ 2)) .* (xx > 20 & xx < 130);   % a vessel
T = 0.01 * randn(n, C);
for c = 1:5
    sp = zeros(n, 1); sp(randperm(n, 20)) = 1;
    T(:, c) = T(:, c) + filter(1, [1 -exp(-1 / (0.6 * fps))], sp) * 0.3;
end
T(:, 6) = T(:, 6) - 0.05 * (sin((1:n)' / 200) > 0.9);                          % dips only
ft = struct('recording_id', {[repmat({'Z_d1_pre'}, n / 2, 1); repmat({'Z_d1_post'}, n / 2, 1)]}, ...
            'isx_frame', [(0:n / 2 - 1)'; (0:n / 2 - 1)'], 'valid', ones(n, 1));
m = struct('spatial_weights', S, 'temporal_raw', T, 'temporal_extract', max(T, 0), 'fps', fps, ...
           'frame_table', ft, 'max_image', max(S, [], 3), ...
           'extract_info', struct('summary_image', max(S, [], 3)), ...
           'config', struct('extract_1p', struct('avg_cell_radius', r)));
f = fullfile(root, 'Z_d1_extract.mat');
save(f, '-struct', 'm');
o = extract_cell_review(f, fullfile(root, 'rev'));
R = readtable(o.csv, 'TextType', 'char');
assert_true(height(R) == C && all(ismember({'keep', 'note', 'verdict'}, R.Properties.VariableNames)), 'review table');
assert_true(all(strcmp(R.verdict(1:5), 'cell')), sprintf('5 somata cell-like (%s)', strjoin(R.verdict(1:5)', ',')));
assert_true(strcmp(R.verdict{6}, 'not_cell') && R.keep(6) == 0, 'vessel / dipping trace is not_cell, keep 0');
assert_true(isfile(o.map_png) && isfile(o.pages{1}), 'map and page written');
% decisions are never overwritten
R.keep(2) = 0; writetable(R, o.csv);
o2 = extract_cell_review(f, fullfile(root, 'rev'));
R2 = readtable(o2.csv, 'TextType', 'char');
assert_true(R2.keep(2) == 0 && isfile(fullfile(root, 'rev', 'Z_d1_cell_review_NEW.csv')), 'review csv kept');
k = apply_cell_keep(f, o.csv);
K = load(k.file);
assert_true(isequal(K.kept_ids(:)', [1 3 4 5]) && size(K.spatial_weights, 3) == 4 && ...
            size(K.temporal_raw, 2) == 4, 'apply_cell_keep keeps the keep = 1 rows, ids kept');
assert_true(isfile(f), 'EXTRACT file untouched');
fprintf('extract_cell_review selftest PASSED\n');
out = true;
end


function assert_true(c, msg)
if ~c, error('extract_cell_review:selftest', 'FAILED: %s', msg); end
end


function out = tern(c, a, b)
if c, out = a; else, out = b; end
end
