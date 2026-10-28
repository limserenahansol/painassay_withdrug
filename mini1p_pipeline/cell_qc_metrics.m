function Q = cell_qc_metrics(S, T, radius, fps, opts)
%CELL_QC_METRICS  Per-component QC of an EXTRACT result: is it a soma, and does its trace
%   look like calcium? Used by extract_param_sweep (to compare settings) and by
%   extract_cell_review (to pre-fill the keep / drop table). The verdict is a SUGGESTION
%   for the human review, never the final decision.
%
%   Q = cell_qc_metrics(S, T, radius, fps)
%   S      H x W x C footprints (any numeric; sparse ndSparse is fine after full())
%   T      n x C traces, dF/F (NaN allowed at dropped frames)
%   radius expected soma radius in px of S's grid (EXTRACT avg_cell_radius)
%   fps    frame rate of T (Hz)
%   opts   thresholds (defaults below), any field may be overridden
%
%   Q is a table, one row per component:
%     id, row, col               centroid (px)
%     area_px, area_ratio        footprint above 0.3 x its peak; ratio to pi * radius^2
%     axis_ratio                 major / minor axis of that region (vessels, streaks >> 1)
%     n_pieces                   separate regions above 0.3 x peak (fragments, two cells)
%     edge_px                    centroid distance to the nearest FOV edge
%     snr                        peak of the 3-frame median-filtered trace / noise std,
%                                noise = 1.4826 x MAD(diff(T)) / sqrt(2) (EXTRACT's style)
%     skew                       skewness of the trace (calcium transients: clearly > 0)
%     neg_ratio                  -p0.5 / p99.5: > 1 means it dips more than it rises
%     n_events                   peaks > 4 noise std, >= 2 s apart
%     rise_s, decay_half_s       of the average of the (up to 10) largest events: time from
%                                half height to the peak, and from the peak back to half
%     ca_like                    decay_half_s >= rise_s and decay_half_s >= 0.15 s (GCaMP:
%                                fast rise, slow fall; noise and artefacts are symmetric)
%     max_S_corr, partner        most similar other footprint (duplicates)
%     verdict                    'cell' | 'check' | 'not_cell'. not_cell = wrong size, wrong
%                                shape, fragments or no signal; a mostly decreasing trace is
%                                'check' (inhibited cells), never not_cell by itself
%     reasons                    why (semicolon list)

if nargin < 5, opts = struct(); end
P = struct('thr', 0.3, 'area_min', 0.2, 'area_max', 5, 'area_cell_min', 0.35, ...
           'area_cell_max', 3, 'axis_max', 3, 'axis_cell_max', 2, 'snr_min', 3, ...
           'snr_cell_min', 5, 'skew_min', 0, 'skew_cell_min', 0.5, 'neg_max', 1, ...
           'dup_corr', 0.8, 'event_sd', 4, 'event_gap_s', 2, ...
           'piece_min_frac', 0.05, 'pieces_max', 2);
% piece_min_frac: a region counts as a piece if >= this fraction of the biggest one (short
% movies give noisy footprints with many specks above 0.3 x peak - 2026-10-21 loose preview
% uses 0.25); pieces_max: more pieces than this = not_cell (default 2, loose 4)
f = fieldnames(opts);
for k = 1:numel(f), P.(f{k}) = opts.(f{k}); end

S = double(full(S));
[H, W, C] = size(S);
if C == 0
    Q = empty_table();
    return;
end
T = double(T);
if size(T, 2) ~= C
    error('cell_qc_metrics:size', 'T has %d columns, S has %d components', size(T, 2), C);
end
area_exp = pi * radius ^ 2;

row = NaN(C, 1); col = row; area = row; axr = row; npc = row; edge = row;
for c = 1:C
    im = S(:, :, c);
    m = max(im(:));
    if ~(m > 0), continue; end
    bw = im >= P.thr * m;
    cc = bwconncomp(bw, 8);
    sz = cellfun(@numel, cc.PixelIdxList);
    [big, j] = max(sz);
    npc(c) = nnz(sz >= max(3, P.piece_min_frac * big));
    one = false(H, W);
    one(cc.PixelIdxList{j}) = true;
    st = regionprops(one, im, 'WeightedCentroid', 'MajorAxisLength', 'MinorAxisLength', 'Area');
    col(c) = st.WeightedCentroid(1);
    row(c) = st.WeightedCentroid(2);
    area(c) = st.Area;
    axr(c) = st.MajorAxisLength / max(st.MinorAxisLength, 1);
    edge(c) = min([row(c) - 1, H - row(c), col(c) - 1, W - col(c)]);
end

snr = NaN(C, 1); skw = snr; negr = snr; nev = snr; rise = snr; dech = snr;
gap = max(1, round(P.event_gap_s * fps));
for c = 1:C
    x = T(:, c);
    x = x(isfinite(x));
    if numel(x) < 10, continue; end
    % Transients are found and measured on a 0.2-s running mean: GCaMP events last longer
    % than that, single-frame shot-noise spikes do not (at 30 Hz the largest raw peaks of a
    % least-squares trace are noise spikes, which looked like 0.07-s "events" on 5941).
    w = max(1, round(0.2 * fps));
    xm = movmean(medfilt1(x, 3), w);
    noise = 1.4826 * median(abs(diff(xm) - median(diff(xm)))) / sqrt(2);
    noise = max(noise, eps);
    snr(c) = max(xm) / noise;
    skw(c) = skewness(x);
    p = prctile(x, [0.5 99.5]);
    negr(c) = -p(1) / max(p(2), eps);
    xs = xm - median(xm);
    ws = warning('off', 'signal:findpeaks:largeMin');
    [pk, loc] = findpeaks(xs, 'MinPeakHeight', P.event_sd * noise, 'MinPeakDistance', gap);
    warning(ws);
    nev(c) = numel(pk);
    [rise(c), dech(c)] = event_shape(xm - median(xm), pk, loc, fps);
end
ca_like = dech >= rise & dech >= 0.15;

% most similar other footprint
Sn = reshape(S, H * W, C);
Sn = Sn ./ max(sqrt(sum(Sn .^ 2, 1)), eps);
G = Sn' * Sn;
G(1:C + 1:end) = -inf;
[mxc, partner] = max(G, [], 1);
mxc = mxc(:); partner = partner(:);
if C == 1, mxc = 0; partner = 0; end
D = hypot(row - row', col - col');
D(1:C + 1:end) = inf;
[dmin, near] = min(D, [], 2);

ar = area / area_exp;
verdict = repmat({'check'}, C, 1);
reasons = repmat({''}, C, 1);
for c = 1:C
    bad = {};
    if ~(ar(c) >= P.area_min), bad{end + 1} = sprintf('too small (%.2fx)', ar(c)); end %#ok<AGROW>
    if ar(c) > P.area_max, bad{end + 1} = sprintf('too large (%.1fx)', ar(c)); end %#ok<AGROW>
    if axr(c) > P.axis_max, bad{end + 1} = sprintf('elongated (%.1f)', axr(c)); end %#ok<AGROW>
    if npc(c) > P.pieces_max, bad{end + 1} = sprintf('%d pieces', npc(c)); end %#ok<AGROW>
    if ~(snr(c) >= P.snr_min), bad{end + 1} = sprintf('low SNR (%.1f)', snr(c)); end %#ok<AGROW>
    % A trace that mostly DIPS is not rejected here: on the 5941 test the brightest soma was
    % tonically active and then suppressed when the mouse moved (skew -1.0). That is the
    % inhibited-cell case; it goes to 'check' and the movie-at-its-events test decides.
    weak = {};
    if ar(c) < P.area_cell_min || ar(c) > P.area_cell_max, weak{end + 1} = sprintf('size %.2fx', ar(c)); end %#ok<AGROW>
    if axr(c) > P.axis_cell_max, weak{end + 1} = sprintf('axis %.1f', axr(c)); end %#ok<AGROW>
    if npc(c) > 1, weak{end + 1} = sprintf('%d pieces', npc(c)); end %#ok<AGROW>
    if snr(c) < P.snr_cell_min, weak{end + 1} = sprintf('SNR %.1f', snr(c)); end %#ok<AGROW>
    if ~(skw(c) >= P.skew_min) || negr(c) > P.neg_max
        weak{end + 1} = sprintf('mostly decreases (skew %.2f) - inhibited cell?', skw(c)); %#ok<AGROW>
    elseif skw(c) < P.skew_cell_min
        weak{end + 1} = sprintf('skew %.2f', skw(c)); %#ok<AGROW>
    end
    if ~ca_like(c), weak{end + 1} = 'no Ca-like transient'; end %#ok<AGROW>
    if mxc(c) > P.dup_corr, weak{end + 1} = sprintf('overlaps #%d (r %.2f)', partner(c), mxc(c)); end %#ok<AGROW>
    if C > 1 && dmin(c) < radius
        weak{end + 1} = sprintf('centre %.0f px from #%d: same soma split?', dmin(c), near(c)); %#ok<AGROW>
    end
    if ~isempty(bad)
        verdict{c} = 'not_cell';
        reasons{c} = strjoin(bad, '; ');
    elseif isempty(weak)
        verdict{c} = 'cell';
    else
        reasons{c} = strjoin(weak, '; ');
    end
end

Q = table((1:C)', row, col, area, ar, axr, npc, edge, snr, skw, negr, nev, rise, dech, ...
          ca_like, mxc, partner, verdict, reasons, 'VariableNames', ...
          {'id', 'row', 'col', 'area_px', 'area_ratio', 'axis_ratio', 'n_pieces', 'edge_px', ...
           'snr', 'skew', 'neg_ratio', 'n_events', 'rise_s', 'decay_half_s', 'ca_like', ...
           'max_S_corr', 'partner', 'verdict', 'reasons'});
end


function [rise, dech] = event_shape(xs, pk, loc, fps)
% Average of the largest events (unsmoothed trace), aligned on the peak: time from half
% height up to the peak, and from the peak down to half height.
rise = NaN; dech = NaN;
if isempty(pk), return; end
[~, o] = sort(pk, 'descend');
loc = loc(o(1:min(10, numel(o))));
pre = round(2 * fps); post = round(5 * fps);
loc = loc(loc > pre & loc + post <= numel(xs));
if isempty(loc), return; end
A = zeros(numel(loc), pre + post + 1);
for i = 1:numel(loc)
    A(i, :) = xs(loc(i) - pre:loc(i) + post);
end
a = mean(A, 1);
a = a - median(a(1:max(1, round(pre / 2))));
p = a(pre + 1);
if ~(p > 0), return; end
before = a(1:pre + 1);
ir = find(before <= 0.5 * p, 1, 'last');
if ~isempty(ir), rise = (pre + 1 - ir) / fps; else, rise = pre / fps; end
after = a(pre + 1:end);
ih = find(after <= 0.5 * p, 1, 'first');
if ~isempty(ih), dech = (ih - 1) / fps; else, dech = post / fps; end
end


function Q = empty_table()
Q = table(zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
          zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
          zeros(0, 1), zeros(0, 1), false(0, 1), zeros(0, 1), zeros(0, 1), cell(0, 1), cell(0, 1), ...
          'VariableNames', {'id', 'row', 'col', 'area_px', 'area_ratio', 'axis_ratio', 'n_pieces', ...
          'edge_px', 'snr', 'skew', 'neg_ratio', 'n_events', 'rise_s', 'decay_half_s', 'ca_like', ...
          'max_S_corr', 'partner', 'verdict', 'reasons'});
end
