function out = seed_footprints(projections_mat, extract_mat, out_dir, opts)
%SEED_FOOTPRINTS  Candidate cells from the whole-movie correlation image, to give EXTRACT as
%   starting footprints (S_init) when its own cell finding misses somata (5941 test: the
%   dim, slightly larger cells in the bright middle of the lens). Deliberately loose: the
%   user deletes the non-cells in cell_curation_gui.
%
%   out = seed_footprints(projections_mat, extract_mat, out_dir, opts)
%
%   projections_mat : fov_projection_images output (Cn, PNR, mean_img)
%   extract_mat     : an earlier extract_1p result whose components are kept as seeds too
%                     ('' = none)
%   opts: radius (13 px), k_sd (3: spot threshold, median + k robust SD of the smoothed
%         Cn x PNR), min_sep (1.5 radii), lens_frac (0.15 of the mean image range),
%         grow_frac (0.4 of the spot peak)
%
%   Writes <out_dir>\seeds.mat (S_init, H*W x C, full resolution; seed table) and
%   seeds.png (Cn x PNR with the earlier components in green and the new seeds in cyan).

if nargin < 4, opts = struct(); end
r = getopt(opts, 'radius', 13);
k_sd = getopt(opts, 'k_sd', 3);
sep = getopt(opts, 'min_sep', 1.5) * r;
lens_frac = getopt(opts, 'lens_frac', 0.15);
gfrac = getopt(opts, 'grow_frac', 0.4);
if ~isfolder(out_dir), mkdir(out_dir); end
P = load(projections_mat);
seed = double(P.Cn .* P.PNR);
[H, W] = size(seed);
sm = imgaussfilt(seed, max(1, r / 3));
mu = median(sm(:));
sd = 1.4826 * median(abs(sm(:) - mu));
% inside the lens: the smoothed mean image above lens_frac of its range
mi = imgaussfilt(double(P.mean_img), 20);
q = prctile(mi(:), [1 99]);
lens = mi > q(1) + lens_frac * (q(2) - q(1));
lens = imerode(lens, strel('disk', round(r)));
m = max(20, round(2 * r));
lens([1:m, end - m + 1:end], :) = false;
lens(:, [1:m, end - m + 1:end]) = false;
pk = imregionalmax(sm) & sm > mu + k_sd * sd & lens;
[pr, pc] = find(pk);
v = sm(pk);
[v, o] = sort(v, 'descend');
pr = pr(o); pc = pc(o);
keep = true(numel(pr), 1);
for i = 1:numel(pr)
    if ~keep(i), continue; end
    d = hypot(pr - pr(i), pc - pc(i));
    keep(d < sep & (1:numel(pr))' > i) = false;
end
pr = pr(keep); pc = pc(keep); v = v(keep);

% earlier components
S0 = zeros(H * W, 0);
cov = false(H, W);
if ~isempty(extract_mat)
    e = load(extract_mat, 'spatial_weights');
    S = full(e.spatial_weights);
    S0 = reshape(S, H * W, []);
    for c = 1:size(S, 3)
        f = S(:, :, c);
        cov = cov | f >= 0.3 * max(f(:));
    end
end
cov = imdilate(cov, strel('disk', round(r / 2)));
new = ~cov(sub2ind([H W], pr, pc));
pr = pr(new); pc = pc(new); v = v(new);

% forced seeds (opts.extra_rc, N x 2 [row col]: cells you marked on a screenshot), unless an
% earlier component's core or another seed already sits within 0.6 radius
xr = getopt(opts, 'extra_rc', zeros(0, 2));
n_forced = 0;
for i = 1:size(xr, 1)
    r0 = round(xr(i, 1)); c0 = round(xr(i, 2));
    if r0 < 1 || r0 > H || c0 < 1 || c0 > W, continue; end
    near0 = hypot(pr - r0, pc - c0) < 0.6 * r;
    if any(near0)
        if ~getopt(opts, 'extra_replace', false), continue; end
        pr(near0) = []; pc(near0) = []; v(near0) = [];     % your mark replaces the automatic seed there
    end
    pr(end + 1, 1) = r0; pc(end + 1, 1) = c0; v(end + 1, 1) = sm(r0, c0); %#ok<AGROW>
    n_forced = n_forced + 1;
end
if size(xr, 1), fprintf('seed_footprints: %d of %d marked positions added as seeds\n', n_forced, size(xr, 1)); end

% a footprint per new seed: the connected part of the spot above grow_frac of its peak,
% within 2 radii, weighted by the image; a Gaussian if that is too small
[xx, yy] = meshgrid(1:W, 1:H);
Snew = zeros(H * W, numel(pr));
for i = 1:numel(pr)
    near = hypot(yy - pr(i), xx - pc(i)) <= 2 * r;
    bw = near & sm >= gfrac * sm(pr(i), pc(i));
    bw = bwselect(bw, pc(i), pr(i), 8);
    if nnz(bw) < 0.3 * pi * r ^ 2 || i > numel(pr) - n_forced     % forced seeds: a Gaussian soma
        g = exp(-((yy - pr(i)) .^ 2 + (xx - pc(i)) .^ 2) / (2 * (0.6 * r) ^ 2));
        g(g < 0.05) = 0;
        f = g;
    else
        f = sm .* bw;
        f = imgaussfilt(f, 1);
    end
    Snew(:, i) = f(:) / max(f(:));
end
S_init = [S0, Snew];
src = [repmat({'extract'}, size(S0, 2), 1); repmat({'seed'}, numel(pr), 1)];
seeds = table(pr, pc, v, 'VariableNames', {'row', 'col', 'seed_value'});
save(fullfile(out_dir, 'seeds.mat'), 'S_init', 'src', 'seeds', 'r', 'k_sd', '-v7.3');

% picture
a = seed / prctile(seed(:), 99.7);
im = repmat(uint8(255 * min(max(a, 0), 1)), 1, 1, 3);
for c = 1:size(S0, 2)
    f = reshape(S0(:, c), H, W);
    e2 = imdilate(bwperim(f >= 0.3 * max(f(:))), ones(2));
    im(:, :, 2) = max(im(:, :, 2), uint8(255 * e2));
end
im = insertMarker(im, [pc pr], 'plus', 'Color', 'cyan', 'Size', 9);
im = insertText(im, [10 10], sprintf('Cn x PNR: green = %d earlier EXTRACT components, cyan + = %d new seeds (k = %g SD)', ...
                size(S0, 2), numel(pr), k_sd), 'FontSize', 18, 'BoxOpacity', 0.5);
imwrite(im, fullfile(out_dir, 'seeds.png'));
out = struct('file', fullfile(out_dir, 'seeds.mat'), 'n_extract', size(S0, 2), 'n_seed', numel(pr));
fprintf('seed_footprints: %d earlier + %d new seeds -> %s\n', size(S0, 2), numel(pr), out.file);
end


function v = getopt(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end
