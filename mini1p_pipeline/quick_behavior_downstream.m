function out = quick_behavior_downstream(analysis_dir, features_csv, opts)
%QUICK_BEHAVIOR_DOWNSTREAM  Fast first look at calcium vs AUTOMATIC behaviour (no manual
%   scoring needed): movement onsets and rearing from behavior_auto.py, and how every cell
%   follows the mouse's movement. For pipeline checks; the real analysis uses the manual
%   scoring through merge_mini1p_scoring + classify_pain_neurons.
%
%   out = quick_behavior_downstream(analysis_dir, features_csv)
%
%   analysis_dir : folder with final_neuron_behavior.mat (merge_mini1p_scoring)
%   features_csv : <tag>_features.csv of behavior_auto.py (cam_frame, isx_frame, motion, rearing)
%   opts         : still_s (2) quiet time before a movement onset; motion_q (0.75) motion
%                  threshold quantile of the side camera; n_shuffle (500)
%
%   Adds movement_onset_vector and rearing (as escape_vector, the mini2p "escape / rearing"
%   key) to final_neuron_behavior.mat (a copy, final_neuron_behavior_auto.mat, is made first
%   and the original is kept as final_neuron_behavior_scored.mat if it existed), runs the
%   mini2p classify_pain_neurons with the movement onsets as an extra event, and writes
%     quick_overview.png        z-scored traces of every cell (heatmap) over the session
%                               with side + bottom motion energy and rearing bouts under it
%     quick_motion_corr.csv     per cell: Pearson r with the side-camera motion energy (1-s
%                               bins), circular-shift null p, sign
%     quick_motion_corr.png     the r of every cell against its null band

if nargin < 3, opts = struct(); end
still_s = getopt(opts, 'still_s', 2);
mq = getopt(opts, 'motion_q', 0.75);
nsh = getopt(opts, 'n_shuffle', 500);

f = fullfile(analysis_dir, 'final_neuron_behavior.mat');
M = load(f);
F = readtable(features_csv);
n = size(M.deltaF_over_F, 1);
fps = M.metadata.neuron_frame_rate_hz;
% camera frame n = isx_frame n = row n + 1
rowof = F.isx_frame + 1;
ok = rowof >= 1 & rowof <= n;
side = nan(n, 1); bot = nan(n, 1); rear = false(n, 1);
side(rowof(ok)) = F.side_motion(ok);
bot(rowof(ok)) = F.bottom_motion(ok);
rear(rowof(ok)) = F.rearing(ok) > 0;

% movement onsets: side motion crosses its q-quantile after >= still_s below it
thr = quantile(side(isfinite(side)), mq);
mov = side > thr;
mov(~isfinite(side)) = false;
q = round(still_s * fps);
on = find(diff([false; mov]) == 1);
on = on(arrayfun(@(i) i > q && ~any(mov(i - q:i - 1)), on));
movement_onset_vector = false(n, 1);
movement_onset_vector(on) = true;

M.movement_onset_vector = movement_onset_vector;
M.escape_vector = rear;                       % mini2p: escape / rearing
M.side_motion = side;
M.bottom_motion = bot;
M.metadata.auto_behavior = struct('features_csv', features_csv, 'still_s', still_s, ...
    'motion_threshold', thr, 'n_movement_onsets', numel(on), 'n_rearing_frames', nnz(rear), ...
    'note', 'AUTOMATIC behaviour (behavior_auto.py), not manual scoring');
copyfile(f, fullfile(analysis_dir, 'final_neuron_behavior_scored.mat'));
save(fullfile(analysis_dir, 'final_neuron_behavior_auto.mat'), '-struct', 'M', '-v7.3');
save(f, '-struct', 'M', '-v7.3');

% mini2p classification with the movement onsets added
o = struct('pre_sec', 3, 'post_sec', 5, 'f0_start_sec', -3, 'f0_end_sec', -1, 'save_fig', true, ...
           'extra_events', {{'movement_onset_vector', 'Movement onset', [0.1 0.5 0.2]}});
classify_pain_neurons(analysis_dir, o);

% motion correlation, 1-s bins, circular-shift null
Z = double(M.traces_z);
C = size(Z, 2);
b = round(fps);
nb = floor(n / b);
bin = @(x) squeeze(mean(reshape(x(1:nb * b, :), b, nb, []), 1, 'omitnan'));
Zb = bin(Z);
if C == 1, Zb = Zb(:); end
sb = bin(side);
sb = sb(:);
okb = isfinite(sb) & all(isfinite(Zb), 2);
Zb = Zb(okb, :); sb = sb(okb);
r = corr(Zb, sb);
rng(1);
null = zeros(nsh, C);
L = numel(sb);
for k = 1:nsh
    sh = randi([round(0.1 * L), round(0.9 * L)]);
    null(k, :) = corr(Zb, circshift(sb, sh));
end
p = (1 + sum(abs(null) >= abs(r(:))', 1))' / (nsh + 1);
T = table(M.orig_id(:), r(:), p, sign(r(:)) .* (p < 0.05), 'VariableNames', ...
          {'cell_id', 'r_motion', 'p_shift', 'motion_sign'});
writetable(T, fullfile(analysis_dir, 'quick_motion_corr.csv'));

% figures
t = (0:n - 1)' / fps;
[~, o2] = sort(r, 'descend');
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [20 20 1400 900]);
ax1 = subplot(5, 1, 1:3);
imagesc(t, 1:C, Z(:, o2)'); clim([-2 4]); colormap(ax1, parula);
ylabel('cell (sorted by r with motion)'); title(sprintf(['%s: %d cells, z-scored ' ...
      '(AUTOMATIC behaviour; dashed = movement onsets)'], strrep(M.metadata.recording_id, '_', '\_'), C));
hold on;
for i = on(:)', xline(t(i), 'w--', 'LineWidth', 0.5); end
colorbar;
ax2 = subplot(5, 1, 4);
plot(t, side, 'Color', [0.85 0.4 0.1]); hold on; plot(t, bot, 'Color', [0.2 0.4 0.8]);
yline(thr, 'k:'); legend({'side motion', 'bottom motion', 'onset threshold'}, 'Box', 'off');
ylabel('motion energy'); box off;
ax3 = subplot(5, 1, 5);
area(t, double(rear), 'FaceColor', [0.8 0.2 0.2], 'EdgeColor', 'none'); ylim([0 1.2]);
ylabel('rearing'); xlabel('time (s)'); box off;
linkaxes([ax1 ax2 ax3], 'x');
exportgraphics(fig, fullfile(analysis_dir, 'quick_overview.png'), 'Resolution', 110);
close(fig);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [20 20 900 420]);
[rs, o3] = sort(r);
lo = prctile(null, 2.5, 1); hi = prctile(null, 97.5, 1);
fill([1:C, C:-1:1], [lo(o3), fliplr(hi(o3))], [0.85 0.85 0.85], 'EdgeColor', 'none'); hold on;
sig = p(o3) < 0.05;
plot(find(~sig), rs(~sig), 'o', 'Color', [0.5 0.5 0.5]);
plot(find(sig & rs > 0), rs(sig & rs > 0), 'o', 'MarkerFaceColor', [0.85 0.2 0.2], 'MarkerEdgeColor', 'none');
plot(find(sig & rs < 0), rs(sig & rs < 0), 'o', 'MarkerFaceColor', [0.2 0.4 0.85], 'MarkerEdgeColor', 'none');
yline(0, 'k:');
xlabel('cell (sorted)'); ylabel('r with side-camera motion (1-s bins)');
title(sprintf('motion-correlated: %d up, %d down of %d (circular-shift null, p < 0.05; grey = 95%% null)', ...
      nnz(sig & rs > 0), nnz(sig & rs < 0), C), 'FontWeight', 'normal');
box off;
exportgraphics(fig, fullfile(analysis_dir, 'quick_motion_corr.png'), 'Resolution', 110);
close(fig);
out = struct('n_cells', C, 'n_movement_onsets', numel(on), 'n_motion_up', nnz(sig & rs > 0), ...
             'n_motion_down', nnz(sig & rs < 0), 'table', T);
fprintf('quick_behavior_downstream: %d cells, %d movement onsets, motion-correlated %d up / %d down\n', ...
        C, numel(on), out.n_motion_up, out.n_motion_down);
end


function v = getopt(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end
