function out = extract_1p(cfg)
%EXTRACT_1P  EXTRACT with one-photon settings on one mouse-day's <tag>_mc.h5:/mov.
%
%   out = extract_1p(cfg)        run on the output of inscopix_day
%   extract_1p('selftest')       synthetic test, errors (non-zero exit under -batch) on failure
%
%   What it does
%     1. reads the frame table written by inscopix_day (<tag>_mc_frames.csv); the frame
%        rate comes from its raw TSC timestamps (else from the IDPS timing period in the
%        JSON), never from a constant
%     2. runs EXTRACT on <tag>_mc.h5:/mov with its own preprocessing ON (per-pixel F0,
%        spatial high-pass, background removal) so traces come out as dF/F; cells are
%        found on a movie binned detect_bin_time x in time and traces are re-estimated at
%        full rate
%     3. computes unconstrained least-squares traces on the final footprints, because
%        EXTRACT's robust/non-negative traces clip decreases (pin-prick-inhibited cells)
%     4. re-expands both trace sets to the full raw isx_frame index of every recording,
%        NaN at trimmed and dropped frames
%     5. saves <tag>_extract.mat, <tag>_frame_table.csv (CONTRACTS.md "extract outputs"),
%        <tag>_extract.json and two QC PNGs
%
%   Inputs: the day movie of inscopix_day. In its default mode 'lossless_mc' that is the
%   FULL-RESOLUTION movie (1280 x 800 nVista3), raw values motion-corrected and nothing
%   else; EXTRACT does every other step itself (preprocess = 1: per-pixel F0 and dF/F,
%   spatial high-pass, background removal; detection on a movie it bins itself with
%   downsample_space_by / downsample_time_by). In mode 'idps_preprocess' the movie is
%   already 2x binned by IDPS. The defaults below follow the mode the day JSON names.
%
%   Required: cfg.day_json = the <tag>_inscopix.json of inscopix_day,
%             or all of cfg.h5, cfg.frames_csv, cfg.mouse_id, cfg.day
%   Optional (default)
%     out_dir                  (folder of the h5)
%     avg_cell_radius          ([])    px of the INPUT movie; [] = 13 for a full-resolution
%                                      movie (the 2024 PCA-ICA used a 13 px diameter after
%                                      2x binning), 7 for the IDPS 2x-binned movie
%     detect_downsample_space_by ([]) EXTRACT downsample_space_by; [] = 2 for a full-
%                                      resolution movie, 1 for the IDPS-binned one. EXTRACT
%                                      re-expands the footprints to the input grid.
%     spatial_highpass_cutoff  (5)     EXTRACT units (x cell radius); 3-10 is the usual range
%     detect_prebin            (false) DETECTION on a movie this function bins itself
%                                      (detect_bin_time x in time, 2 x 2 in space, written next
%                                      to the outputs) so that EXTRACT fits in ONE partition:
%                                      EXTRACT's partition seams put false candidates along a
%                                      line and can split cells. Equivalent to EXTRACT's own
%                                      downsample_space_by 2 / downsample_time_by; footprints
%                                      are scaled back up 2x (bilinear, max 1) and the
%                                      least-squares traces are full rate and full resolution as
%                                      always. temporal_extract is then the binned-rate EXTRACT
%                                      trace, repeated per frame (the analysis uses temporal_raw).
%     detect_bin_time          (4)     EXTRACT downsample_time_by for DETECTION only (traces
%                                      are re-estimated on every frame). 4 = the mini2p pipeline,
%                                      EXTRACT's own template (downsampletime_pipeline default)
%                                      and Corder 2019 / Parker 2018 (20 -> 5 Hz); 30 Hz -> 7.5 Hz
%     avg_event_tau_s          (0.5)   indicator decay; config.avg_event_tau = round(fps * this)
%     use_gpu                  ('auto') true if a GPU is found
%     num_partitions           ([])    [] = planned from RAM and GPU memory (never fewer than
%                                      EXTRACT's own ~250 px blocks), or [nx ny]
%     ram_fraction             (0.6)   share of the free RAM one partition may use
%     gpu_fraction             (0.7)   share of the free GPU memory one partition may use
%     movie_mask               ([])    [] none | 'auto' (lens disk from the mean image) |
%                                      path to .png/.tif/.mat holding a logical H x W | logical
%                                      EXTRACT 1.x applies movie_mask to the movie AFTER its
%                                      own downsample_space_by and then fails (size mismatch,
%                                      checked 2026-10-19). So with detect_downsample_space_by
%                                      > 1 the mask is applied to the result instead: a
%                                      component is dropped when less than mask_min_inside
%                                      of its footprint weight lies inside the mask.
%     mask_min_inside          (0.5)
%     mask_auto_frac           (0.15)  'auto' keeps the smoothed mean image above this
%                                      fraction of its dark-to-bright range
%     trace_output_option      ('raw') EXTRACT's robust traces without the non-negativity clamp
%     T_min_snr                (7)     set explicitly: the local default (7) and the README (10) differ
%     cellfind_min_snr         (1)
%     eccent_thresh            (6)     removes elongated blood-vessel components in 1p
%     extract_overrides        (struct()) any other EXTRACT config field, applied last
%     ls_use_gpu               ('auto') the least-squares movie filter on the GPU (250-frame
%                                      blocks, CPU fallback on any error); false = CPU only
%     ls_block_frames          (1000)
%     fallback                 (true)  on GPU / memory errors retry with 2x partitions,
%                                      then 2x partitions on the CPU
%     toolbox_dir   (fullfile(getenv('USERPROFILE'), 'OneDrive', 'Desktop', 'cursor',
%                   'pain_2plane_split'))   folder of mini2p_toolbox_paths.m
%
%   .mat contents: spatial_weights (H x W x C), temporal_raw and temporal_extract
%   (T x C, dF/F, T = every raw frame of every recording, NaN where not imaged),
%   frame_table (struct of T-row columns: recording_id, isx_frame, assay, valid,
%   invalid_reason, h5_row, extract_row, tsc_us), config, fps, plus QC images and
%   cell_qc (per-footprint centroid, area, axis ratio: streaks along vessels or motion
%   edges have axis_ratio >> 1 and are curation candidates, EXTRACT keeps them).
%   extract_row = row of temporal_* for this frame (1-based), empty/NaN if the frame
%   is trimmed or dropped (the row exists but holds NaN).

if nargin == 1 && (ischar(cfg) || isstring(cfg)) && strcmpi(cfg, 'selftest')
    out = run_selftest();
    return;
end

t_all = tic;
cfg = apply_defaults(cfg);
addpath(cfg.toolbox_dir);
tb = mini2p_toolbox_paths();    % strips ActSort, which shadows ~20 EXTRACT helpers

tag = sprintf('%s_d%d', cfg.mouse_id, cfg.day);
files = struct( ...
    'mat', fullfile(cfg.out_dir, [tag '_extract.mat']), ...
    'frame_table_csv', fullfile(cfg.out_dir, [tag '_frame_table.csv']), ...
    'json', fullfile(cfg.out_dir, [tag '_extract.json']), ...
    'png_footprints', fullfile(cfg.out_dir, [tag '_extract_footprints.png']), ...
    'png_traces', fullfile(cfg.out_dir, [tag '_extract_traces.png']));
run_log = struct('module', 'extract_1p', 'status', 'running', ...
    'created', char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss')), ...
    'matlab_version', version, 'mouse_id', cfg.mouse_id, 'day', cfg.day, ...
    'h5', cfg.h5, 'frames_csv', cfg.frames_csv, 'files', files, 'toolboxes', tb);
run_log.warnings = {};
run_log.runtime_s = struct();
try
    [out, run_log] = run_extract_day(cfg, files, run_log);
    run_log.status = 'ok';
catch err
    run_log.status = 'failed';
    run_log.error = err.message;
    run_log.error_report = getReport(err, 'extended', 'hyperlinks', 'off');
    run_log.runtime_s.total = toc(t_all);
    write_json(files.json, run_log);
    rethrow(err);
end
run_log.runtime_s.total = toc(t_all);
out.log = run_log;
write_json(files.json, run_log);
fprintf('extract_1p: %s, %d cells, %.1f s -> %s\n', tag, run_log.n_cells, run_log.runtime_s.total, files.mat);
end


% =====================================================================================
function [out, run_log] = run_extract_day(cfg, files, run_log)
% ---- frame table, movie size, frame rate -----------------------------------------
ft = read_frame_table(cfg.frames_csv);
hi = h5info(cfg.h5, '/mov');
sz = hi.Dataspace.Size;
H = sz(1);
W = sz(2);
n_h5 = sz(3);
valid_rows = find(ft.valid == 1);
if numel(valid_rows) ~= n_h5 || ~isequal(ft.h5_row(valid_rows), (1:n_h5)')
    error('extract_1p:frames', ['frame table and h5 disagree: %d valid rows, h5 has %d frames, ', ...
          'h5_row must be 1..n in table order'], numel(valid_rows), n_h5);
end
if any(~isnan(ft.h5_row(ft.valid ~= 1)))
    error('extract_1p:frames', 'invalid frames must not have an h5_row');
end
run_log.movie_check = check_movie(cfg.h5, ft);
[fps, fps_info] = frame_rate(ft, cfg.timing_period_s);
run_log.fps = fps;
run_log.fps_info = fps_info;
if fps_info.max_rel_diff > 1e-3
    run_log.warnings{end+1} = sprintf('recordings differ in frame rate by %.3g (relative)', fps_info.max_rel_diff);
end
dst = cfg.detect_bin_time;
run_log.downsample_time_by = dst;
run_log.detection_hz = fps / dst;
run_log.movie_size = [H W n_h5];

mean_image = read_or_compute_mean(cfg.h5, H, W, n_h5, cfg.ls_block_frames);
max_image = [];
try
    max_image = h5read(cfg.h5, '/max_image');
catch
end

% ---- movie mask -------------------------------------------------------------
[mask, mask_info] = resolve_mask(cfg.movie_mask, mean_image, cfg.avg_cell_radius, cfg.mask_auto_frac);
dss = cfg.detect_downsample_space_by;
mask_in_extract = ~isempty(mask) && dss == 1;
mask_info.applied = 'none';
if ~isempty(mask)
    if mask_in_extract
        mask_info.applied = 'EXTRACT movie_mask';
    else
        mask_info.applied = sprintf(['after EXTRACT: components with < %.0f%% of their footprint ', ...
            'weight inside the mask are dropped (EXTRACT cannot combine movie_mask with ', ...
            'downsample_space_by = %d)'], 100 * cfg.mask_min_inside, dss);
    end
end
run_log.movie_mask = mask_info;
run_log.input_full_resolution = cfg.input_full_resolution;
run_log.input_grid = cfg.input_grid;
run_log.avg_cell_radius = cfg.avg_cell_radius;
run_log.downsample_space_by = dss;

% ---- EXTRACT -------------------------------------------------------------
use_gpu = resolve_gpu(cfg.use_gpu);
pb = cfg.detect_prebin;
mov_extract = [cfg.h5 ':/mov'];
He = H; We = W;
if pb
    % detection movie binned here (time x dst, space 2 x 2) -> EXTRACT in one partition
    [bh5, pbinfo] = binned_detection_movie(cfg.h5, H, W, n_h5, dst, 2, ...
        regexprep(cfg.h5, '\.h5$', sprintf('_detect_binned_t%d_s2.h5', dst)), cfg.ls_block_frames);   % next to the movie: pass 1 and 2 share it
    mov_extract = [bh5 ':/mov'];
    He = pbinfo.H; We = pbinfo.W;
    run_log.detect_prebin = rmfield(pbinfo, 'mean');
    fprintf('extract_1p: detection on a %dx%dx%d movie binned here (time %d, space 2)\n', He, We, pbinfo.n, dst);
end
config = struct();
config.avg_cell_radius = cfg.avg_cell_radius / (1 + pb);
config.preprocess = 1;               % 1p: EXTRACT's own F0, high-pass and background removal
config.skip_dff = 0;
config.remove_background = 1;        % local get_defaults has false; it only acts when preprocess = 1
config.spatial_highpass_cutoff = cfg.spatial_highpass_cutoff;
config.downsample_space_by = dss;    % EXTRACT bins for detection and re-expands the footprints
config.reestimate_S_if_downsampled = 0;
config.downsample_time_by = dst;
config.reestimate_T_if_downsampled = 1;
config.avg_event_tau = max(1, round(cfg.avg_event_tau_s * fps));
if pb
    config.downsample_space_by = 1;  % already binned
    config.downsample_time_by = 1;
    config.reestimate_T_if_downsampled = 0;
    config.avg_event_tau = max(1, round(cfg.avg_event_tau_s * fps / dst));
end
config.trace_output_option = cfg.trace_output_option;
config.use_gpu = use_gpu;
config.verbose = 2;
config.thresholds = struct('T_min_snr', cfg.T_min_snr, 'eccent_thresh', cfg.eccent_thresh);
config.cellfind_min_snr = cfg.cellfind_min_snr;
if mask_in_extract
    config.movie_mask = mask;
end
if pb && ~isempty(mask)
    config.movie_mask = imresize(pad_even(double(mask)), [He We], 'nearest') > 0.5;
    mask_in_extract = true;          % the binned movie is masked inside EXTRACT
    mask_info.applied = 'EXTRACT movie_mask (binned detection movie)';
    run_log.movie_mask = mask_info;
end
if ~isempty(cfg.num_partitions)
    parts = cfg.num_partitions(:)';
    run_log.partition_plan = struct('source', 'cfg.num_partitions', 'partitions', parts);
else
    if pb
        % the point of the binned movie is ONE partition (no seams): if the GPU cannot hold it,
        % run on the CPU instead of splitting (a real memory error still falls back to more)
        [parts, plan] = plan_partitions(He, We, pbinfo.n, config.avg_cell_radius, 1, use_gpu, ...
                                        cfg.ram_fraction, cfg.gpu_fraction);
        if any(parts > 1)
            if use_gpu
                p_cpu = plan_partitions(He, We, pbinfo.n, config.avg_cell_radius, 1, false, ...
                                        cfg.ram_fraction, cfg.gpu_fraction);
                if all(p_cpu == 1)
                    use_gpu = false;
                    config.use_gpu = false;
                end
            end
            plan.note = sprintf('binned detection movie: forced 1 x 1 (planner said %d x %d); GPU %d', parts, use_gpu);
            parts = [1 1];
        end
    else
        [parts, plan] = plan_partitions(H, W, n_h5, cfg.avg_cell_radius, dss, use_gpu, ...
                                        cfg.ram_fraction, cfg.gpu_fraction);
    end
    run_log.partition_plan = plan;
end
config.num_partitions_x = parts(1);
config.num_partitions_y = parts(2);
fprintf('extract_1p: %dx%dx%d movie, radius %g px, downsample_space_by %d, time %d, partitions %dx%d\n', ...
        H, W, n_h5, cfg.avg_cell_radius, dss, dst, parts(1), parts(2));
ov = fieldnames(cfg.extract_overrides);
for k = 1:numel(ov)
    if strcmp(ov{k}, 'thresholds')
        th = fieldnames(cfg.extract_overrides.thresholds);
        for j = 1:numel(th)
            config.thresholds.(th{j}) = cfg.extract_overrides.thresholds.(th{j});
        end
    else
        config.(ov{k}) = cfg.extract_overrides.(ov{k});
    end
end
if pb && isfield(config, 'S_init') && size(config.S_init, 1) == H * W
    % starting footprints are full resolution: bin them like the movie (2 x 2), max 1
    K = size(config.S_init, 2);
    Sb = zeros(He * We, K);
    for k = 1:K
        x = imresize(pad_even(reshape(full(double(config.S_init(:, k))), H, W)), [He We], 'box');
        Sb(:, k) = x(:) / max(max(x(:)), eps);
    end
    config.S_init = Sb;
end

t = tic;
[output, attempts] = run_with_fallback(mov_extract, config, He, We, cfg.fallback);
run_log.runtime_s.extract = toc(t);
run_log.extract_attempts = attempts;
used = attempts{end};
if numel(attempts) > 1
    run_log.warnings{end+1} = sprintf('EXTRACT needed %d attempts; final: use_gpu=%d partitions=%s', ...
        numel(attempts), used.use_gpu, mat2str(used.partitions));
end

S = full(single(output.spatial_weights));
if pb
    % footprints of the binned movie -> full resolution (2x bilinear, cropped, max 1)
    if ndims(S) == 2 && size(S, 1) == He * We, S = reshape(S, He, We, []); end
    if ndims(S) == 2 && size(S, 1) == He && size(S, 2) == We, S = reshape(S, He, We, 1); end
    Su = zeros(H, W, size(S, 3), 'single');
    for k = 1:size(S, 3)
        x = imresize(S(:, :, k), 2, 'bilinear');
        x = x(1:H, 1:W);
        x(x < 0) = 0;
        Su(:, :, k) = x / max(max(x(:)), eps);
    end
    S = Su;
    if size(output.temporal_weights, 2) > 0
        Tb = single(output.temporal_weights);
        Tr = repelem(Tb, dst, 1);
        if size(Tr, 1) < n_h5, Tr = [Tr; repmat(Tr(end, :), n_h5 - size(Tr, 1), 1)]; end
        output.temporal_weights = Tr(1:n_h5, :);
    end
end
if ndims(S) == 2 && size(S, 1) == H && size(S, 2) == W
    S = reshape(S, H, W, 1);
end
C = size(output.temporal_weights, 2);
if C == 0
    S = zeros(H, W, 0, 'single');
    run_log.warnings{end+1} = 'EXTRACT found no cells';
end
T_ext = single(output.temporal_weights);
if C == 0
    T_ext = zeros(n_h5, 0, 'single');    % EXTRACT returns 0 x 0 when it finds nothing
end
if size(T_ext, 1) ~= n_h5 && C > 0
    error('extract_1p:extract', 'EXTRACT returned %d trace samples, movie has %d frames', size(T_ext, 1), n_h5);
end
if size(S, 1) ~= H || size(S, 2) ~= W
    error('extract_1p:extract', 'EXTRACT footprints are %dx%d, the movie is %dx%d', size(S, 1), size(S, 2), H, W);
end
run_log.n_cells_extract = C;
if ~isempty(mask) && ~mask_in_extract && C > 0
    w = reshape(S, H * W, C);
    inside = (double(mask(:))' * double(w)) ./ max(sum(double(w), 1), eps);
    keep = inside >= cfg.mask_min_inside;
    run_log.movie_mask.n_dropped_outside = nnz(~keep);
    run_log.movie_mask.inside_fraction_dropped = inside(~keep);
    S = S(:, :, keep);
    T_ext = T_ext(:, keep);
    C = nnz(keep);
end
run_log.n_cells = C;

% ---- unconstrained least-squares traces on the final footprints ---------------------
t = tic;
% The LS movie must be the one EXTRACT fitted: masked only if EXTRACT masked it.
ls_mask = [];
if mask_in_extract
    ls_mask = mask;
end
% The spatial filter of the LS movie is the slow part (~25 s per 1000 frames on the CPU): with
% ls_use_gpu ('auto') it runs on the GPU in 250-frame blocks (~3 GB; 3.3x faster, max |diff|
% 1e-3 counts), and any GPU error falls back to the CPU, so the result never depends on the GPU.
T_ls = [];
lsg = cfg.ls_use_gpu;
if ischar(lsg) || isstring(lsg), lsg = resolve_gpu('auto'); end
if lsg
    try
        [T_ls, ls_info] = least_squares_traces(cfg.h5, S, mean_image, ls_mask, cfg.avg_cell_radius, ...
            config.spatial_highpass_cutoff, true, 250);
        ls_info.device = 'gpu';
    catch e
        fprintf(2, 'extract_1p: least squares on the GPU failed (%s) - CPU\n', e.message);
        T_ls = [];
    end
end
if isempty(T_ls)
    [T_ls, ls_info] = least_squares_traces(cfg.h5, S, mean_image, ls_mask, cfg.avg_cell_radius, ...
        config.spatial_highpass_cutoff, false, cfg.ls_block_frames);
    ls_info.device = 'cpu';
end
run_log.runtime_s.least_squares = toc(t);
if C > 0
    ls_info.corr_with_extract = diag(corr(double(T_ls), double(T_ext)))';
    ls_info.corr_median = median(ls_info.corr_with_extract, 'omitnan');
    ls_info.frac_negative_ls = mean(T_ls < 0, 1);
    ls_info.frac_negative_extract = mean(T_ext < 0, 1);
    ls_info.min_ls_median = median(min(T_ls, [], 1));
    ls_info.min_extract_median = median(min(T_ext, [], 1));
end
% The LS dF/F divides by our own mean image; it must be EXTRACT's F_per_pixel. With
% downsample_space_by > 1 EXTRACT's F_per_pixel is the binned mean spline-interpolated back
% to full resolution, so only the median difference is expected to be small.
fp = double(output.info.F_per_pixel);
mi_cmp = double(mean_image);
if pb, mi_cmp = double(pbinfo.mean); end    % EXTRACT saw the binned movie
if ~isequal(size(fp), size(mi_cmp)), fp = imresize(fp, size(mi_cmp), 'bilinear'); end
ok = fp > 0;
rel = abs(mi_cmp(ok) - fp(ok)) ./ fp(ok);
ls_info.F_per_pixel_max_rel_diff = max(rel);
ls_info.F_per_pixel_median_rel_diff = median(rel);
if (dss == 1 && ls_info.F_per_pixel_max_rel_diff > 1e-3) || ls_info.F_per_pixel_median_rel_diff > 0.02
    run_log.warnings{end+1} = sprintf('mean image differs from EXTRACT F_per_pixel (median %.3g, max %.3g)', ...
        ls_info.F_per_pixel_median_rel_diff, ls_info.F_per_pixel_max_rel_diff);
end
run_log.least_squares = rmfield(ls_info, intersect(fieldnames(ls_info), {'F_per_cell'}));

% ---- footprint shape, for curation -------------------------------------------
cell_qc = footprint_metrics(S, cfg.avg_cell_radius);
run_log.footprints = struct('area_px_median', median(cell_qc.area_px), ...
    'expected_area_px', pi * cfg.avg_cell_radius^2, ...
    'axis_ratio_median', median(cell_qc.axis_ratio), 'n_axis_ratio_gt3', nnz(cell_qc.axis_ratio > 3));

% ---- re-expand to the full raw index -----------------------------------------
n_full = numel(ft.isx_frame);
temporal_raw = NaN(n_full, C, 'single');
temporal_extract = NaN(n_full, C, 'single');
temporal_raw(valid_rows, :) = T_ls(ft.h5_row(valid_rows), :);
temporal_extract(valid_rows, :) = T_ext(ft.h5_row(valid_rows), :);
ft.extract_row = NaN(n_full, 1);
ft.extract_row(valid_rows) = valid_rows;
frame_table = ft;
write_csv(files.frame_table_csv, frame_table, frame_table_formats());

% ---- save --------------------------------------------------------------------
ex_cfg = output.config;
if isfield(ex_cfg, 'movie_mask')
    ex_cfg.movie_mask = logical(ex_cfg.movie_mask);
end
config_out = struct('extract', ex_cfg, 'extract_1p', cfg, 'attempts', {attempts}, ...
                    'inscopix_day', cfg.inscopix_day_params);
info_small = struct('summary_image', single(output.info.summary_image), ...
                    'max_image', single(output.info.max_image), ...
                    'F_per_pixel', single(output.info.F_per_pixel), ...
                    'runtime_s', output.info.runtime, 'version', output.info.version);
if isfield(output.info, 'cellcheck')
    info_small.cellcheck = output.info.cellcheck;
end
vars = struct('spatial_weights', S, 'temporal_raw', temporal_raw, 'temporal_extract', temporal_extract, ...
    'frame_table', frame_table, 'config', config_out, 'fps', fps, 'mean_image', mean_image, ...
    'max_image', max_image, 'movie_mask', mask, 'F_per_cell', ls_info.F_per_cell, ...
    'extract_info', info_small, 'cell_qc', cell_qc, 'mouse_id', cfg.mouse_id, 'day', cfg.day, ...
    'recording_ids', {unique(frame_table.recording_id, 'stable')'}, ...
    'trace_units', 'dF/F (EXTRACT F_per_pixel = per-pixel mean of the day movie)');
fmt = '-v7';
w = whos('vars');
if w.bytes > 1.9e9
    fmt = '-v7.3';     % v7 caps a variable at 2 GB; v7.3 needs h5py on the Python side
    run_log.warnings{end+1} = 'saved as -v7.3 (too large for -v7): read with h5py, not scipy.io';
end
save(files.mat, '-struct', 'vars', fmt);
run_log.mat_format = fmt;

% ---- QC figures ---------------------------------------------------------------
try
    bg = max_image;
    if isempty(bg)
        bg = info_small.max_image;
    end
    plot_footprints(files.png_footprints, bg, info_small.max_image, S, mask, ...
                    sprintf('%s_d%d', cfg.mouse_id, cfg.day), cfg.avg_cell_radius);
    plot_traces(files.png_traces, T_ext, T_ls, ft, valid_rows, fps);
catch err
    run_log.warnings{end+1} = ['QC figure failed: ' err.message];
end

out = struct('files', files, 'n_cells', C, 'fps', fps, 'downsample_time_by', dst, ...
             'downsample_space_by', dss, 'partitions', parts, 'avg_cell_radius', cfg.avg_cell_radius);
end


% =====================================================================================
function cfg = apply_defaults(cfg)
if ~isstruct(cfg)
    error('extract_1p:cfg', 'cfg must be a struct (or ''selftest'')');
end
cfg.inscopix_day_params = struct();
cfg.timing_period_s = struct('recording_id', {}, 'period_s', {});
cfg.input_mode = '';
if isfield(cfg, 'day_json') && ~isempty(cfg.day_json)
    js = jsondecode(fileread(cfg.day_json));
    jdir = fileparts(cfg.day_json);
    cfg.mouse_id = js.mouse_id;
    cfg.day = js.day;
    cfg.input_mode = 'idps_preprocess';
    if isfield(js, 'mode')
        cfg.input_mode = js.mode;
    end
    % Prefer the recorded paths, but survive a moved folder.
    cfg.h5 = first_file(js.files.h5, jdir);
    cfg.frames_csv = first_file(js.files.frames_csv, jdir);
    cfg.inscopix_day_params = js.params;
    recs = js.recordings;
    if isstruct(recs)
        recs = num2cell(recs);
    end
    for r = 1:numel(recs)
        cfg.timing_period_s(r) = struct('recording_id', recs{r}.recording_id, ...
                                        'period_s', recs{r}.period_s_timing);
    end
end
need = {'h5', 'frames_csv', 'mouse_id', 'day'};
for k = 1:numel(need)
    if ~isfield(cfg, need{k}) || isempty(cfg.(need{k}))
        error('extract_1p:cfg', 'cfg.%s is required (or give cfg.day_json)', need{k});
    end
end
cfg.h5 = char(cfg.h5);
cfg.frames_csv = char(cfg.frames_csv);
cfg.mouse_id = char(cfg.mouse_id);
if ~isfile(cfg.h5)
    error('extract_1p:cfg', 'h5 not found: %s', cfg.h5);
end
if ~isfile(cfg.frames_csv)
    error('extract_1p:cfg', 'frame table not found: %s', cfg.frames_csv);
end
def = struct( ...
    'out_dir', fileparts(cfg.h5), ...
    'avg_cell_radius', [], ...
    'detect_downsample_space_by', [], ...
    'spatial_highpass_cutoff', 5, ...
    'detect_bin_time', 4, 'detect_prebin', false, ...
    'avg_event_tau_s', 0.5, ...
    'use_gpu', 'auto', ...
    'num_partitions', [], ...
    'ram_fraction', 0.6, ...
    'gpu_fraction', 0.7, ...
    'movie_mask', [], ...
    'mask_min_inside', 0.5, ...
    'mask_auto_frac', 0.15, ...
    'trace_output_option', 'raw', ...
    'T_min_snr', 7, ...
    'cellfind_min_snr', 1, ...
    'eccent_thresh', 6, ...
    'extract_overrides', struct(), ...
    'ls_block_frames', 1000, 'ls_use_gpu', 'auto', ...
    'fallback', true, ...
    'toolbox_dir', fullfile(getenv('USERPROFILE'), 'OneDrive', 'Desktop', 'cursor', 'pain_2plane_split'));
f = fieldnames(def);
optional_empty = {'num_partitions', 'movie_mask', 'avg_cell_radius', 'detect_downsample_space_by'};
for k = 1:numel(f)
    if ~isfield(cfg, f{k}) || (isempty(cfg.(f{k})) && ~ismember(f{k}, optional_empty))
        cfg.(f{k}) = def.(f{k});
    end
end
% Is the input the full sensor grid or IDPS's 2x-binned one? From the day JSON's mode
% (a JSON without 'mode' was written by the IDPS path), else from the frame width.
hi = h5info(cfg.h5, '/mov');
if ~isempty(cfg.input_mode)
    full_res = strcmp(cfg.input_mode, 'lossless_mc');
    cfg.input_grid = 'from day JSON mode';
else
    full_res = hi.Dataspace.Size(2) >= 1000;
    cfg.input_grid = 'from frame width (no day JSON)';
    % a lens-cropped motion-corrected movie (motion_correct_h5 crop_rect) is narrower than
    % 1000 px but still full-resolution pixels: decide from the crop attribute, not the width
    try
        h5readatt(cfg.h5, '/mov', 'crop_rect');
        full_res = true;
        cfg.input_grid = 'full resolution (lens-cropped movie, /mov crop_rect)';
    catch
    end
end
cfg.input_full_resolution = full_res;
if isempty(cfg.avg_cell_radius)
    cfg.avg_cell_radius = 7;
    if full_res
        cfg.avg_cell_radius = 13;
    end
end
if isempty(cfg.detect_downsample_space_by)
    cfg.detect_downsample_space_by = 1;
    if full_res
        cfg.detect_downsample_space_by = 2;
    end
end
dss = cfg.detect_downsample_space_by;
if dss < 1 || dss ~= round(dss)
    error('extract_1p:cfg', 'detect_downsample_space_by must be a positive integer');
end
if ~isscalar(cfg.detect_bin_time) || cfg.detect_bin_time < 1 || cfg.detect_bin_time ~= round(cfg.detect_bin_time)
    error('extract_1p:cfg', 'detect_bin_time must be a positive integer');
end
if ~isempty(cfg.num_partitions) && numel(cfg.num_partitions) ~= 2
    error('extract_1p:cfg', 'num_partitions must be [nx ny]');
end
if ~isfolder(cfg.out_dir)
    mkdir(cfg.out_dir);
end
if ~isfile(fullfile(cfg.toolbox_dir, 'mini2p_toolbox_paths.m'))
    error('extract_1p:cfg', 'mini2p_toolbox_paths.m not found in %s', cfg.toolbox_dir);
end
end


function f = first_file(p, jdir)
f = p;
if ~isfile(f)
    [~, nm, ext] = fileparts(p);
    f = fullfile(jdir, [nm ext]);
end
end


function ft = read_frame_table(file)
opts = detectImportOptions(file, 'Delimiter', ',', 'TextType', 'char');
text_cols = intersect({'recording_id', 'assay', 'invalid_reason'}, opts.VariableNames);
opts = setvartype(opts, text_cols, 'char');
num_cols = setdiff(opts.VariableNames, text_cols);
opts = setvartype(opts, num_cols, 'double');
tb = readtable(file, opts);
need = {'recording_id', 'isx_frame', 'valid', 'h5_row'};
miss = setdiff(need, tb.Properties.VariableNames);
if ~isempty(miss)
    error('extract_1p:frames', 'frame table %s lacks column(s): %s', file, strjoin(miss, ', '));
end
ft = struct();
ft.recording_id = tb.recording_id;
ft.isx_frame = tb.isx_frame;
if ismember('assay', tb.Properties.VariableNames)
    ft.assay = tb.assay;
else
    ft.assay = repmat({''}, height(tb), 1);
end
ft.valid = tb.valid;
if ismember('invalid_reason', tb.Properties.VariableNames)
    r = tb.invalid_reason;
    r(cellfun(@isempty, r)) = {''};
    ft.invalid_reason = r;
else
    ft.invalid_reason = repmat({''}, height(tb), 1);
end
ft.h5_row = tb.h5_row;
if ismember('tsc_us', tb.Properties.VariableNames)
    ft.tsc_us = tb.tsc_us;
else
    ft.tsc_us = NaN(height(tb), 1);
end
% The master index of a recording must be 0..n-1 with no gaps (CONTRACTS.md section 1).
[ids, ~, g] = unique(ft.recording_id, 'stable');
for k = 1:numel(ids)
    fr = ft.isx_frame(g == k);
    if ~isequal(fr(:)', 0:numel(fr) - 1)
        error('extract_1p:frames', '%s: isx_frame is not 0..n-1 in order', ids{k});
    end
end
end


function [fps, info] = frame_rate(ft, timing)
% From the hardware TSC timestamps when present: (last - first) / frames between, per
% recording, the same convention IDPS uses for its average period.
[ids, ~, g] = unique(ft.recording_id, 'stable');
per = NaN(1, numel(ids));
src = cell(1, numel(ids));
for k = 1:numel(ids)
    sel = find(g == k & ft.valid == 1 & isfinite(ft.tsc_us));
    if numel(sel) >= 2
        per(k) = (ft.tsc_us(sel(end)) - ft.tsc_us(sel(1))) / (ft.isx_frame(sel(end)) - ft.isx_frame(sel(1))) / 1e6;
        src{k} = 'tsc';
    else
        j = find(strcmp({timing.recording_id}, ids{k}), 1);
        if ~isempty(j)
            per(k) = timing(j).period_s;
            src{k} = 'idps_timing_period';
        end
    end
end
if any(isnan(per))
    error('extract_1p:fps', ['no frame period for %s: the frame table has no TSC timestamps and no ', ...
          'IDPS timing period was given (pass cfg.day_json)'], strjoin(ids(isnan(per)), ', '));
end
fps = 1 / mean(per);
info = struct('recording_id', {ids'}, 'period_s', per, 'source', {src}, ...
              'max_rel_diff', (max(per) - min(per)) / mean(per));
end


function m = read_or_compute_mean(h5, H, W, n, block)
try
    m = h5read(h5, '/mean_image');
    if isequal(size(m), [H W])
        m = single(m);
        return;
    end
catch
end
s = zeros(H, W);
for i0 = 1:block:n
    nb = min(block, n - i0 + 1);
    s = s + sum(double(h5read(h5, '/mov', [1 1 i0], [H W nb])), 3);
end
m = single(s / n);
end


function [mask, info] = resolve_mask(spec, mean_image, radius, frac)
[H, W] = size(mean_image);
info = struct('kind', 'none', 'fraction', 1);
mask = [];
if isempty(spec)
    return;
end
if islogical(spec) || isnumeric(spec)
    mask = logical(spec);
    info.kind = 'given';
elseif (ischar(spec) || isstring(spec)) && strcmpi(spec, 'auto')
    mask = auto_lens_mask(mean_image, radius, frac);
    info.kind = 'auto';
    info.auto_frac = frac;
elseif ischar(spec) || isstring(spec)
    spec = char(spec);
    [~, ~, ext] = fileparts(spec);
    if strcmpi(ext, '.mat')
        s = load(spec);
        fn = fieldnames(s);
        mask = logical(s.(fn{1}));
    else
        mask = imread(spec) > 0;
        if ndims(mask) == 3
            mask = any(mask, 3);
        end
    end
    info.kind = ['file:' spec];
else
    error('extract_1p:mask', 'movie_mask must be [], ''auto'', a file path or a logical H x W');
end
if ~isequal(size(mask), [H W])
    error('extract_1p:mask', 'movie_mask is %s, movie is %dx%d', mat2str(size(mask)), H, W);
end
info.fraction = mean(mask(:));
if info.fraction < 0.1
    error('extract_1p:mask', 'movie_mask keeps only %.1f%% of the FOV', 100 * info.fraction);
end
end


function mask = auto_lens_mask(mean_image, radius, frac)
% The GRIN lens is the bright disk of the mean image: smooth away cells, threshold at
% frac of the way from the dark surround (5th pct) to the bright centre (99th pct), keep
% the largest blob, fill it, and pull the edge in by one cell radius so the dim lens rim
% does not turn into a ring of false cells. Otsu was tried first and kept only the
% bright centre (27% of the 2024 nVista3 FOV): vignetting makes the lens a slope, not a
% plateau, so a two-class split lands too high.
im = double(mean_image);
im = imgaussfilt(im, max(2, 2 * radius));
lo = prctile(im(:), 5);
hi = prctile(im(:), 99);
bw = (im - lo) / max(eps, hi - lo) > frac;
cc = bwconncomp(bw);
if cc.NumObjects == 0
    mask = true(size(im));
    return;
end
[~, k] = max(cellfun(@numel, cc.PixelIdxList));
mask = false(size(im));
mask(cc.PixelIdxList{k}) = true;
mask = imfill(mask, 'holes');
mask = imerode(mask, strel('disk', max(1, round(radius))));
end


function chk = check_movie(h5, ft)
% The lossless_mc day movie is an HDF5 virtual dataset over the recordings' _mc.h5 files.
% A missing source reads as fill values (NaN or 0) without any error, and EXTRACT would
% silently fit them; so every source must exist, and the first and last frame of every
% recording must read back finite and non-zero.
chk = struct('layout', 'stored', 'sources', {{}}, 'n_spot_frames', 0);
fid = H5F.open(h5, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
ds = H5D.open(fid, '/mov');
pl = H5D.get_create_plist(ds);
is_virtual = H5P.get_layout(pl) == H5ML.get_constant_value('H5D_VIRTUAL');
missing = {};
if is_virtual
    chk.layout = 'virtual';
    d = fileparts(h5);
    for i = 0:H5P.get_virtual_count(pl) - 1
        f = H5P.get_virtual_filename(pl, i);
        if strcmp(f, '.')
            p = h5;
        elseif ~isempty(regexp(f, '^([A-Za-z]:|\\\\|/)', 'once'))
            p = f;
        else
            p = fullfile(d, f);
        end
        chk.sources{end + 1} = p;
        if ~isfile(p)
            missing{end + 1} = p; %#ok<AGROW>
        end
    end
end
H5P.close(pl);
H5D.close(ds);
H5F.close(fid);
if ~isempty(missing)
    error('extract_1p:movie', ['the day movie %s is a virtual dataset and its source file(s) %s ', ...
          'are missing: they must stay next to it'], h5, strjoin(missing, ', '));
end
hi = h5info(h5, '/mov');
sz = hi.Dataspace.Size;
[ids, ~, g] = unique(ft.recording_id, 'stable');
for k = 1:numel(ids)
    rows = ft.h5_row(g == k & ft.valid == 1);
    if isempty(rows)
        continue;
    end
    for r = unique([rows(1), rows(end)])
        fr = h5read(h5, '/mov', [1 1 r], [sz(1) sz(2) 1]);
        if any(~isfinite(fr(:))) || ~any(fr(:))
            error('extract_1p:movie', '%s: h5 row %d (%s) reads as NaN or zeros', h5, r, ids{k});
        end
        chk.n_spot_frames = chk.n_spot_frames + 1;
    end
end
end


function [bh5, info] = binned_detection_movie(h5, H, W, n, dt, ds, out, block)
% time-binned (mean of dt frames) and 2 x 2 space-binned (mean) single movie for detection
Hb = ceil(H / ds); Wb = ceil(W / ds);
nb = floor(n / dt);
if isfile(out)
    try
        hi = h5info(out, '/mov');
        if isequal(hi.Dataspace.Size, [Hb Wb nb])
            info = struct('file', out, 'H', Hb, 'W', Wb, 'n', nb, 'bin_time', dt, 'bin_space', ds, ...
                          'mean', h5read(out, '/mean_image'), 'reused', true);
            bh5 = out;
            return;
        end
    catch
    end
    delete(out);
end
h5create(out, '/mov', [Hb Wb nb], 'Datatype', 'single', 'ChunkSize', [Hb Wb min(nb, 100)]);
acc = zeros(Hb, Wb);
blk = max(dt, floor(block / dt) * dt);
k = 0;
for i0 = 1:blk:nb * dt
    m = min(blk, nb * dt - i0 + 1);
    Y = single(h5read(h5, '/mov', [1 1 i0], [H W m]));
    Y = squeeze(mean(reshape(Y, H, W, dt, m / dt), 3));
    if m / dt == 1, Y = reshape(Y, H, W, 1); end
    Y = pad_even(Y);
    Y = squeeze(mean(mean(reshape(Y, ds, Hb, ds, Wb, []), 1), 3));
    if size(Y, 3) == 1 && m / dt > 1, Y = reshape(Y, Hb, Wb, []); end
    Y = reshape(Y, Hb, Wb, []);
    h5write(out, '/mov', single(Y), [1 1 k + 1], size(Y, 1:3));
    acc = acc + sum(double(Y), 3);
    k = k + size(Y, 3);
end
mn = single(acc / max(k, 1));
h5create(out, '/mean_image', [Hb Wb], 'Datatype', 'single');
h5write(out, '/mean_image', mn);
info = struct('file', out, 'H', Hb, 'W', Wb, 'n', nb, 'bin_time', dt, 'bin_space', ds, 'mean', mn, 'reused', false);
bh5 = out;
end


function A = pad_even(A)
% replicate the last row / column so both sizes are even (2 x 2 binning)
if mod(size(A, 1), 2), A = cat(1, A, A(end, :, :)); end
if mod(size(A, 2), 2), A = cat(2, A, A(:, end, :)); end
end


function [p, plan] = plan_partitions(H, W, T, radius, dss, use_gpu, ram_fraction, gpu_fraction)
ram = free_ram_bytes();
gpu = Inf;
if use_gpu
    gpu = gpu_free_bytes();
end
[p, plan] = plan_partitions_budget(H, W, T, radius, dss, use_gpu, ram_fraction * ram, gpu_fraction * gpu);
plan.ram_free_gb = ram / 2^30;
plan.gpu_free_gb = gpu / 2^30;
if ~use_gpu
    plan.gpu_free_gb = NaN;
end
end


function [p, plan] = plan_partitions_budget(H, W, T, radius, dss, use_gpu, ram_budget, gpu_budget)
% Start from EXTRACT's own automatic split (~250 px blocks on its detection grid) and add
% partitions along the longer block side until one partition fits:
%   RAM: EXTRACT holds the partition at input resolution (single) and ~3 working copies
%        on its detection grid:            block_px * T * 4 * (1.5 + 3 / dss^2)
%   GPU: the detection-grid movie and one working copy:  (block_px / dss^2) * T * 4 * 2
overlap = ceil(2 * radius);                 % extractor.m partition_overlap
p0 = [max(round(W / dss / 250), 1), max(round(H / dss / 250), 1)];
p = p0;
ram_copies = 1.5 + 3 / dss^2;
gpu_copies = 2;
fits = false;
while true
    bx = ceil((W + (p(1) - 1) * overlap) / p(1));
    by = ceil((H + (p(2) - 1) * overlap) / p(2));
    need_ram = bx * by * T * 4 * ram_copies;
    need_gpu = ceil(bx / dss) * ceil(by / dss) * T * 4 * gpu_copies;
    if need_ram <= ram_budget && (~use_gpu || need_gpu <= gpu_budget)
        fits = true;
        break;
    end
    if bx <= 4 * overlap && by <= 4 * overlap
        break;                              % blocks would be mostly overlap: give up
    end
    if bx >= by
        p(1) = p(1) + 1;
    else
        p(2) = p(2) + 1;
    end
end
plan = struct('source', 'planned', 'partitions', p, 'extract_auto', p0, 'block_hw_px', [by bx], ...
    'overlap_px', overlap, 'n_frames', T, 'ram_budget_gb', ram_budget / 2^30, ...
    'gpu_budget_gb', gpu_budget / 2^30, 'need_ram_gb', need_ram / 2^30, 'need_gpu_gb', need_gpu / 2^30, ...
    'ram_copies', ram_copies, 'gpu_copies', gpu_copies, 'fits', fits);
end


function b = free_ram_bytes()
b = 16 * 2^30;
try
    [~, s] = memory;
    b = s.PhysicalMemory.Available;
catch
end
end


function b = gpu_free_bytes()
b = 0;
try
    d = gpuDevice();
    b = d.AvailableMemory;
catch
end
end


function g = resolve_gpu(spec)
if (ischar(spec) || isstring(spec)) && strcmpi(spec, 'auto')
    g = false;
    try
        g = gpuDeviceCount > 0;
    catch
    end
else
    g = logical(spec);
end
end


function [output, attempts] = run_with_fallback(movie, config, H, W, fallback)
% Auto-partitioning in extractor.m uses ~250 px blocks; reproduce it so "more
% partitions" is well defined when the first attempt runs out of memory.
if isfield(config, 'num_partitions_x')
    p0 = [config.num_partitions_x, config.num_partitions_y];
else
    p0 = [max(round(W / 250), 1), max(round(H / 250), 1)];
end
plan = {struct('use_gpu', config.use_gpu, 'partitions', p0, 'auto', ~isfield(config, 'num_partitions_x'))};
if fallback
    p1 = min(p0 * 2, [max(1, floor(W / 64)), max(1, floor(H / 64))]);
    if config.use_gpu
        plan{end + 1} = struct('use_gpu', true, 'partitions', p1, 'auto', false);
    end
    plan{end + 1} = struct('use_gpu', false, 'partitions', p1, 'auto', false);
end
attempts = {};
for a = 1:numel(plan)
    c = config;
    c.use_gpu = plan{a}.use_gpu;
    if ~plan{a}.auto
        c.num_partitions_x = plan{a}.partitions(1);
        c.num_partitions_y = plan{a}.partitions(2);
    end
    rec = struct('use_gpu', c.use_gpu, 'partitions', plan{a}.partitions, 'error', '');
    try
        fprintf('extract_1p: EXTRACT attempt %d (use_gpu=%d, partitions %dx%d)\n', a, c.use_gpu, ...
                plan{a}.partitions(1), plan{a}.partitions(2));
        output = extractor(movie, c);
        attempts{end + 1} = rec; %#ok<AGROW>
        return;
    catch err
        rec.error = err.message;
        attempts{end + 1} = rec; %#ok<AGROW>
        fprintf(2, 'extract_1p: attempt %d failed: %s\n', a, err.message);
        if ~is_resource_error(err) || a == numel(plan)
            rethrow(err);
        end
        try
            if gpuDeviceCount > 0
                reset(gpuDevice);
            end
        catch
        end
    end
end
end


function tf = is_resource_error(err)
txt = lower([err.identifier ' ' err.message]);
tf = ~isempty(regexp(txt, 'out of memory|nomem|memory|gpu|cuda|parallel:gpu|maximum variable size', 'once'));
end


function [T, info] = least_squares_traces(h5, S, F, mask, radius, hp_cutoff, use_gpu, block)
% T = (S'S)^-1 S' M on the movie EXTRACT itself fits (per-pixel mean removed, the same
% spatial high-pass, the same mask), divided by the footprint-weighted mean F exactly as
% run_extract.m does, so the units match temporal_extract. No non-negativity and no
% robust down-weighting: a cell that goes quiet below its mean gives negative dF/F.
% Streaming in frame blocks keeps memory flat; every step is linear in the movie.
[H, W, C] = size(S);
hi = h5info(h5, '/mov');
n = hi.Dataspace.Size(3);
info = struct('n_frames', n, 'n_cells', C, 'rcond', NaN, 'solver', 'none', 'F_per_cell', zeros(0, 1));
T = zeros(n, C, 'single');
if C == 0
    return;
end
S2 = reshape(double(S), H * W, C);
Ssp = sparse(S2);
G = full(Ssp' * Ssp);
info.rcond = rcond(G);
B = zeros(C, n);
for i0 = 1:block:n
    nb = min(block, n - i0 + 1);
    Mb = h5read(h5, '/mov', [1 1 i0], [H W nb]);
    Mb = bsxfun(@minus, single(Mb), single(F));
    if ~isinf(hp_cutoff)
        Mb = spatial_bandpass(Mb, radius, hp_cutoff, inf, use_gpu);
        if isa(Mb, 'gpuArray'), Mb = gather(Mb); end
    end
    if ~isempty(mask)
        Mb = bsxfun(@times, Mb, single(mask));
    end
    B(:, i0:i0 + nb - 1) = Ssp' * double(reshape(Mb, H * W, nb));
end
if info.rcond > 1e-10
    X = G \ B;
    info.solver = 'normal_equations';
else
    % Near-duplicate footprints: the minimum-norm solution keeps the traces finite.
    X = pinv(G) * B;
    info.solver = 'pinv';
end
F_cell = (S2 ./ max(sum(S2, 1), eps))' * double(F(:));
info.F_per_cell = F_cell;
T = single(bsxfun(@rdivide, X', F_cell'));
end


function q = footprint_metrics(S, radius)
% Shape of each footprint above 0.3 x its peak (largest blob): somata are compact discs
% of about pi*r^2 px; streaks along vessels or motion edges have a large axis ratio.
C = size(S, 3);
q = struct('centroid_xy', NaN(C, 2), 'area_px', NaN(C, 1), 'axis_ratio', NaN(C, 1), ...
           'n_blobs', NaN(C, 1), 'area_over_expected', NaN(C, 1));
for k = 1:C
    im = S(:, :, k);
    bw = im > 0.3 * max(im(:));
    cc = bwconncomp(bw, 8);
    if cc.NumObjects == 0
        continue;
    end
    rp = regionprops(cc, 'Area', 'Centroid', 'MajorAxisLength', 'MinorAxisLength');
    [~, j] = max([rp.Area]);
    q.centroid_xy(k, :) = rp(j).Centroid;
    q.area_px(k) = rp(j).Area;
    q.axis_ratio(k) = rp(j).MajorAxisLength / max(rp(j).MinorAxisLength, eps);
    q.n_blobs(k) = cc.NumObjects;
end
q.area_over_expected = q.area_px / (pi * radius^2);
end


function fmt = frame_table_formats()
fmt = struct('recording_id', '%s', 'isx_frame', '%d', 'assay', '%s', 'valid', '%d', ...
             'invalid_reason', '%s', 'h5_row', '%d', 'extract_row', '%d', 'tsc_us', '%d');
end


function write_csv(file, ft, fmt)
% NaN -> empty field, integers without decimals, fixed column order.
cols = fieldnames(fmt);
n = numel(ft.(cols{1}));
txt = strings(n, numel(cols));
for c = 1:numel(cols)
    v = ft.(cols{c});
    if iscell(v) || isstring(v)
        txt(:, c) = string(v);
    else
        s = compose(string(fmt.(cols{c})), double(v(:)));
        s(isnan(v(:))) = "";
        txt(:, c) = s;
    end
end
fid = fopen(file, 'w');
if fid < 0
    error('extract_1p:io', 'cannot write %s', file);
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', strjoin(cols', ','));
fprintf(fid, '%s\n', join(txt, ',', 2));
end


function write_json(file, s)
fid = fopen(file, 'w');
if fid < 0
    error('extract_1p:io', 'cannot write %s', file);
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(s, 'PrettyPrint', true));
end


function plot_footprints(png, bg_raw, bg_flat, S, mask, tag, radius)
% Left: max projection of the MC movie as recorded (vignetting and all). Right: EXTRACT's
% max image of its own preprocessed movie (F0 removed, high-passed), where somata are
% visible; same outlines on both.
[H, W] = size(bg_raw);
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1800 900 * H / W + 80]);
panels = {bg_raw, 'max projection, MC movie (counts)'; bg_flat, 'EXTRACT max image (preprocessed, dF/F)'};
for p = 1:2
    ax = axes(fig, 'Position', [0.005 + (p - 1) * 0.5, 0.02, 0.49, 0.88]);
    im = double(panels{p, 1});
    if isempty(im) || ~isequal(size(im), [H W])
        im = double(bg_raw);
    end
    lim = prctile(im(:), [1 99.7]);
    imagesc(ax, im, lim + [0 eps]);
    colormap(ax, gray);
    axis(ax, 'image', 'off');
    hold(ax, 'on');
    draw_outlines(ax, S, mask);
    title(ax, panels{p, 2}, 'FontSize', 10);
end
annotation(fig, 'textbox', [0 0.93 1 0.06], 'String', sprintf(['%s: %d EXTRACT footprints ', ...
    '(outline at 0.3 x peak, radius %g px); dashed = movie mask'], tag, size(S, 3), radius), ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 12, 'Interpreter', 'none');
exportgraphics(fig, png, 'Resolution', 130);
close(fig);
end


function draw_outlines(ax, S, mask)
for k = 1:size(S, 3)
    im = S(:, :, k);
    b = bwboundaries(im > 0.3 * max(im(:)), 'noholes');
    if isempty(b)
        continue;
    end
    [~, j] = max(cellfun(@(x) size(x, 1), b));
    plot(ax, b{j}(:, 2), b{j}(:, 1), '-', 'Color', [1 0.75 0], 'LineWidth', 0.7);
end
if ~isempty(mask) && ~all(mask(:))
    mb = bwboundaries(mask, 'noholes');
    for j = 1:numel(mb)
        plot(ax, mb{j}(:, 2), mb{j}(:, 1), '--', 'Color', [0.3 0.8 1], 'LineWidth', 1);
    end
end
end


function plot_traces(png, T_ext, T_ls, ft, valid_rows, fps)
C = size(T_ext, 2);
if C == 0
    return;
end
% Show the cells with the largest transients relative to their noise.
snr = max(T_ext, [], 1) ./ max(1e-6, 1.4826 * mad(T_ext, 1, 1));
[~, order] = sort(snr, 'descend');
pick = order(1:min(8, C));
t = (0:size(T_ext, 1) - 1) / fps;
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1400 900]);
ax = axes(fig);
hold(ax, 'on');
off = 0;
for k = numel(pick):-1:1
    c = pick(k);
    e = double(T_ext(:, c));
    l = double(T_ls(:, c));
    span = max(prctile([e; l], 99.9) - min([e; l]), eps);
    plot(ax, t, l / span + off, 'Color', [0.85 0.35 0.1], 'LineWidth', 0.5);
    plot(ax, t, e / span + off, 'Color', [0 0.3 0.7], 'LineWidth', 0.5);
    text(ax, t(end) * 1.005, off, sprintf('cell %d', c), 'FontSize', 8);
    off = off + 1.2;
end
% Recording boundaries in the concatenated (valid-frame) time axis.
ids = ft.recording_id(valid_rows);
cut = find(~strcmp(ids(1:end - 1), ids(2:end)));
for j = 1:numel(cut)
    xline(ax, t(cut(j)), 'k--', ids{cut(j) + 1}, 'Interpreter', 'none');
end
xlabel(ax, 'time over valid frames (s)');
ylabel(ax, 'dF/F, each cell scaled to its own range');
title(ax, sprintf('8 highest-SNR cells: EXTRACT ''raw'' (blue) vs unconstrained least squares (orange), fps %.4f', fps));
xlim(ax, [0 t(end) * 1.06]);
exportgraphics(fig, png, 'Resolution', 120);
close(fig);
end


% =====================================================================================
function out = run_selftest()
% Synthetic 1p-like day: two recordings with trimmed and dropped frames, nine cells on a
% large drifting background, one of them tonically active and silenced for 12 s.
% Checks: frame rate from TSC, re-expansion with NaN, the frame table, cell recovery, and
% that the least-squares trace of the silenced cell goes below baseline.
fprintf('extract_1p selftest\n');
% Auto lens mask on a vignetted disk: must keep the centre, drop the dark corners.
[yy, xx] = ndgrid(1:120, 1:160);
rr = hypot(yy - 60, xx - 80);
lens = 300 + 2500 * exp(-(rr / 45).^2) .* (rr < 58) + 5 * randn(120, 160);
mk = auto_lens_mask(lens, 4, 0.15);
assert_true(mk(60, 80) && ~mk(1, 1) && ~mk(120, 160) && mean(mk(:)) > 0.1 && mean(mk(:)) < 0.6, ...
            'auto lens mask');
fprintf('  auto lens mask: ok (%.0f%% of FOV)\n', 100 * mean(mk(:)));
here = fileparts(mfilename('fullpath'));
root = fullfile(here, 'out', 'selftest_extract_1p');
if isfolder(root)
    rmdir(root, 's');
end
mkdir(root);
syn = make_synthetic_day(root);

cfg = struct('h5', syn.h5, 'frames_csv', syn.frames_csv, 'mouse_id', 'T01', 'day', 1, ...
             'out_dir', root, 'avg_cell_radius', syn.radius, 'use_gpu', false, ...
             'movie_mask', true(syn.H, syn.W));
r = extract_1p(cfg);
m = load(r.files.mat);

assert_true(abs(m.fps - syn.fps) / syn.fps < 1e-6, sprintf('fps from TSC (%.6f vs %.6f)', m.fps, syn.fps));
assert_true(r.downsample_time_by == 4, 'detection binned 4x in time (mini2p)');
n_full = syn.n_raw(1) + syn.n_raw(2);
C = size(m.spatial_weights, 3);
assert_true(isequal(size(m.temporal_raw), [n_full C]) && isequal(size(m.temporal_extract), [n_full C]), ...
            'traces span every raw frame');
inv = m.frame_table.valid ~= 1;
assert_true(all(isnan(m.temporal_raw(inv, :)), 'all') && all(isfinite(m.temporal_raw(~inv, :)), 'all'), ...
            'NaN exactly at trimmed/dropped frames');
assert_true(nnz(inv) == syn.n_invalid, 'number of invalid frames');
assert_true(all(isnan(m.frame_table.extract_row(inv))) && ...
            isequal(m.frame_table.extract_row(~inv), find(~inv)), 'extract_row');
tb = readtable(r.files.frame_table_csv, 'Delimiter', ',', 'TextType', 'char');
assert_true(height(tb) == n_full && isequaln(tb.extract_row, m.frame_table.extract_row) && ...
            isequal(tb.recording_id, m.frame_table.recording_id), 'frame_table csv == mat');
assert_true(isfile(r.files.png_footprints), 'footprint PNG');
assert_true(numel(m.cell_qc.axis_ratio) == C && median(m.cell_qc.axis_ratio) < 1.5, ...
            'footprint metrics (synthetic cells are round)');

% Match found footprints to the true centres.
cen = zeros(C, 2);
for k = 1:C
    im = double(m.spatial_weights(:, :, k));
    [yy, xx] = ndgrid(1:syn.H, 1:syn.W);
    cen(k, :) = [sum(yy(:) .* im(:)), sum(xx(:) .* im(:))] / sum(im(:));
end
found = false(1, size(syn.centres, 1));
match = zeros(1, size(syn.centres, 1));
for j = 1:size(syn.centres, 1)
    if C == 0
        break;
    end
    [d, k] = min(hypot(cen(:, 1) - syn.centres(j, 1), cen(:, 2) - syn.centres(j, 2)));
    found(j) = d < 3;
    match(j) = k;
end
fprintf('  cells: %d found, %d of %d true cells matched\n', C, nnz(found), numel(found));
assert_true(nnz(found) >= numel(found) - 2, 'most synthetic cells recovered');
j = syn.inhibited;
assert_true(found(j), 'silenced cell recovered');
tr = m.temporal_raw(:, match(j));
te = m.temporal_extract(:, match(j));
in_dip = syn.dip_rows;
out_dip = setdiff(find(~inv), in_dip);
d_ls = mean(tr(in_dip)) - median(tr(out_dip));
% dF/F is relative to the footprint-weighted mean of the movie, background included.
expect = -syn.tonic_dff * syn.F0 / m.F_per_cell(match(j));
fprintf('  silenced cell: LS dip %.3f dF/F (expected about %.3f, mean in dip %.3f), EXTRACT dip %.3f\n', ...
        d_ls, expect, mean(tr(in_dip)), mean(te(in_dip)) - median(te(out_dip)));
% The high-pass removes part of a small blob's amplitude (EXTRACT's own trace shrinks the
% same way), so the test asks for a clear negative dip, not the full expected size.
noise = 1.4826 * mad(diff(tr(out_dip)), 1) / sqrt(2);
assert_true(mean(tr(in_dip)) < 0 && d_ls < 0.25 * expect && d_ls < -3 * noise, ...
            'least-squares trace keeps the decrease');

% ---- partition planner -------------------------------------------------------------
[p, plan] = plan_partitions_budget(800, 1280, 4629, 13, 2, true, 1e12, 1e12);
assert_true(isequal(p, [3 2]) && plan.fits, 'planner keeps EXTRACT''s own split when it fits');
[p, plan] = plan_partitions_budget(800, 1280, 67200, 13, 2, true, 40e9, 9e9);
assert_true(plan.fits && prod(p) > 6 && plan.need_ram_gb * 2^30 <= 40e9 && plan.need_gpu_gb * 2^30 <= 9e9, ...
            'planner adds partitions until a 28-min day fits the budget');
fprintf('  partition planner: ok (full day on 40 GB RAM / 9 GB GPU -> %dx%d)\n', p(1), p(2));

% ---- virtual day movie guard ---------------------------------------------------------
vd = fullfile(root, 'vds');
mkdir(vd);
a = single(100 + rand(8, 9, 3));
b = single(100 + rand(8, 9, 2));
h5create(fullfile(vd, 'a_mc.h5'), '/mov', size(a), 'Datatype', 'single');
h5write(fullfile(vd, 'a_mc.h5'), '/mov', a);
h5create(fullfile(vd, 'b_mc.h5'), '/mov', size(b), 'Datatype', 'single');
h5write(fullfile(vd, 'b_mc.h5'), '/mov', b);
view = fullfile(vd, 'day_mc.h5');
write_test_view(view, {'a_mc.h5', 'b_mc.h5'}, [3 2], 8, 9);
vft = struct('recording_id', {[repmat({'a'}, 3, 1); repmat({'b'}, 2, 1)]}, 'valid', ones(5, 1), 'h5_row', (1:5)');
chk = check_movie(view, vft);
assert_true(strcmp(chk.layout, 'virtual') && numel(chk.sources) == 2 && chk.n_spot_frames == 4, ...
            'virtual day movie accepted with its sources');
movefile(fullfile(vd, 'b_mc.h5'), fullfile(vd, 'b_moved.h5'));
caught = false;
try
    check_movie(view, vft);
catch e
    caught = strcmp(e.identifier, 'extract_1p:movie');
end
assert_true(caught, 'virtual day movie with a missing source is refused');
z = cat(3, a, zeros(8, 9, 1, 'single'));
h5create(fullfile(vd, 'zero.h5'), '/mov', size(z), 'Datatype', 'single');
h5write(fullfile(vd, 'zero.h5'), '/mov', z);
zft = struct('recording_id', {repmat({'a'}, 4, 1)}, 'valid', ones(4, 1), 'h5_row', (1:4)');
caught = false;
try
    check_movie(fullfile(vd, 'zero.h5'), zft);
catch e
    caught = strcmp(e.identifier, 'extract_1p:movie');
end
assert_true(caught, 'a zero last frame is refused');
fprintf('  day movie guard: ok\n');

% ---- full-resolution path: EXTRACT bins 2x itself, mask applied to the result -----------
% The first synthetic day scaled 2x (radius 8): after EXTRACT's own 2x binning it is the
% case above. One extra active cell sits in a band the mask removes.
[gx, gy] = meshgrid(36:52:192, 36:56:160);
o = struct('H', 192, 'W', 224, 'radius', 8, 'name', 'T02_d1', 'bg', [80 120 70], 'centres', [gy(:), gx(:)], ...
           'outside', [92 216]);
syn2 = make_synthetic_day(fullfile(root, 'dss2'), o);
mask2 = true(o.H, o.W);
mask2(:, 207:end) = false;
cfg2 = struct('h5', syn2.h5, 'frames_csv', syn2.frames_csv, 'mouse_id', 'T02', 'day', 1, ...
              'out_dir', fullfile(root, 'dss2'), 'avg_cell_radius', o.radius, 'use_gpu', false, ...
              'detect_downsample_space_by', 2, 'movie_mask', mask2);
r2 = extract_1p(cfg2);
m2 = load(r2.files.mat);
js2 = jsondecode(fileread(r2.files.json));
C2 = size(m2.spatial_weights, 3);
assert_true(r2.downsample_space_by == 2 && js2.downsample_space_by == 2, 'EXTRACT downsample_space_by = 2');
assert_true(size(m2.spatial_weights, 1) == o.H && size(m2.spatial_weights, 2) == o.W, ...
            'footprints come back on the full-resolution grid');
assert_true(js2.movie_mask.n_dropped_outside >= 1, 'the off-mask cell was found and dropped after EXTRACT');
cen2 = zeros(C2, 2);
[yy, xx] = ndgrid(1:o.H, 1:o.W);
for k = 1:C2
    im = double(m2.spatial_weights(:, :, k));
    cen2(k, :) = [sum(yy(:) .* im(:)), sum(xx(:) .* im(:))] / sum(im(:));
end
assert_true(all(cen2(:, 2) < 207), 'no kept footprint in the masked-out band');
found2 = arrayfun(@(j) min(hypot(cen2(:, 1) - o.centres(j, 1), cen2(:, 2) - o.centres(j, 2))) < 4, ...
                  1:size(o.centres, 1));
fprintf('  2x detection binning: %d cells kept (%d dropped off-mask), %d of %d true cells matched\n', ...
        C2, js2.movie_mask.n_dropped_outside, nnz(found2), numel(found2));
assert_true(nnz(found2) >= numel(found2) - 2, 'cells recovered with EXTRACT''s own 2x binning');
inv2 = m2.frame_table.valid ~= 1;
assert_true(isequal(size(m2.temporal_raw), [numel(inv2) C2]) && all(isnan(m2.temporal_raw(inv2, :)), 'all') ...
            && all(isfinite(m2.temporal_raw(~inv2, :)), 'all'), 'full-resolution LS traces re-expanded');
out = struct('selftest', 'ok', 'n_cells', C);
fprintf('extract_1p selftest PASSED\n');
end


function write_test_view(view, srcs, n, H, W)
fid = H5F.create(view, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
type = H5T.copy('H5T_NATIVE_FLOAT');
vspace = H5S.create_simple(3, fliplr([H W sum(n)]), []);
dcpl = H5P.create('H5P_DATASET_CREATE');
H5P.set_fill_value(dcpl, type, single(NaN));
off = 0;
for r = 1:numel(srcs)
    sspace = H5S.create_simple(3, fliplr([H W n(r)]), []);
    H5S.select_hyperslab(vspace, 'H5S_SELECT_SET', fliplr([0 0 off]), [], fliplr([H W n(r)]), []);
    H5P.set_virtual(dcpl, vspace, srcs{r}, '/mov', sspace);
    H5S.close(sspace);
    off = off + n(r);
end
dset = H5D.create(fid, '/mov', type, vspace, 'H5P_DEFAULT', dcpl, 'H5P_DEFAULT');
H5D.close(dset);
H5P.close(dcpl);
H5S.close(vspace);
H5T.close(type);
H5F.close(fid);
end


function syn = make_synthetic_day(root, o)
% o (optional): H, W, radius, n_raw, centres, outside (extra active cells, not in centres),
% name, bg ([row col sigma] of the background blob), flicker (SD of the global per-frame gain)
rng(7);
if nargin < 2
    o = struct();
end
[gx, gy] = meshgrid(18:26:96, 18:28:80);
d = struct('H', 96, 'W', 112, 'radius', 4, 'n_raw', [760 640], 'centres', [gy(:), gx(:)], ...
           'outside', zeros(0, 2), 'name', 'T01_d1', 'bg', [40 60 35], 'flicker', 0.02);
f = fieldnames(d);
for k = 1:numel(f)
    if ~isfield(o, f{k})
        o.(f{k}) = d.(f{k});
    end
end
if ~isfolder(root)
    mkdir(root);
end
syn.H = o.H;
syn.W = o.W;
syn.fps = 1e6 / 49962;       % the nVista3 period in us, as in the 2024 recordings
syn.radius = o.radius;
syn.n_raw = o.n_raw;
n_trim = [3 0];
dropped = {200, []};         % raw indices
syn.centres = o.centres;        % (row, col) of the cells that should be found
all_centres = [o.centres; o.outside];
nc = size(all_centres, 1);
syn.inhibited = 5;
syn.tonic_dff = 0.4;
[yy, xx] = ndgrid(1:syn.H, 1:syn.W);
fp = zeros(syn.H, syn.W, nc);
for k = 1:nc
    fp(:, :, k) = exp(-((yy - all_centres(k, 1)).^2 + (xx - all_centres(k, 2)).^2) / (2 * (0.6 * syn.radius)^2));
end
bgimg = 800 + 600 * exp(-((yy - o.bg(1)).^2 + (xx - o.bg(2)).^2) / (2 * o.bg(3)^2));
n_valid = syn.n_raw - n_trim - cellfun(@numel, dropped);
n_tot = sum(n_valid);
% Calcium activity on the concatenated valid frames (cell x frame), units of F0.
kern = exp(-(0:60) / (0.5 * syn.fps));
act = zeros(nc, n_tot);
for k = 1:nc
    rate = 0.006;
    if k > size(o.centres, 1)
        rate = 0.012;           % off-mask cells are made easy to find
    end
    sp = double(rand(1, n_tot) < rate) .* (0.8 + 0.6 * rand(1, n_tot));
    act(k, :) = filter(kern, 1, sp);
end
dip = n_valid(1) + (150:150 + round(12 * syn.fps));   % 12 s in the second recording
tonic = syn.tonic_dff * ones(1, n_tot);
tonic(dip) = 0;
act(syn.inhibited, :) = act(syn.inhibited, :) + tonic;
drift = 1 + 0.05 * sin((1:n_tot) / 900);
glob = 1 + o.flicker * randn(1, n_tot);
M = zeros(syn.H, syn.W, n_tot, 'single');
F0 = 300;
syn.F0 = F0;
for i = 1:n_tot
    cells = reshape(reshape(fp, [], nc) * (F0 * (1 + act(:, i))), syn.H, syn.W);
    fr = bgimg * drift(i) * glob(i) + cells;
    M(:, :, i) = single(fr + 12 * randn(syn.H, syn.W));
end
syn.h5 = fullfile(root, [o.name '_mc.h5']);
h5create(syn.h5, '/mov', size(M), 'Datatype', 'single', 'ChunkSize', [syn.H syn.W 1]);
h5write(syn.h5, '/mov', M);
h5create(syn.h5, '/mean_image', [syn.H syn.W], 'Datatype', 'single');
h5write(syn.h5, '/mean_image', mean(M, 3));
h5create(syn.h5, '/max_image', [syn.H syn.W], 'Datatype', 'single');
h5write(syn.h5, '/max_image', max(M, [], 3));

% Frame table in inscopix_day's format.
rid = {[o.name '_pre'], [o.name '_post']};
assay = {'pre', 'post'};
ft = struct('recording_id', {{}}, 'assay', {{}}, 'isx_frame', [], 'pp_frame', [], 'valid', [], ...
            'invalid_reason', {{}}, 'h5_row', [], 'tsc_us', []);
row = 0;
for r = 1:2
    n = syn.n_raw(r);
    isxf = (0:n - 1)';
    v = isxf >= n_trim(r) & ~ismember(isxf, dropped{r});
    reason = repmat({''}, n, 1);
    reason(isxf < n_trim(r)) = {'trimmed'};
    reason(ismember(isxf, dropped{r})) = {'dropped'};
    h5r = NaN(n, 1);
    h5r(v) = row + (1:nnz(v))';
    row = row + nnz(v);
    pp = isxf - n_trim(r);
    pp(pp < 0) = NaN;
    tsc = 5e8 * r + isxf * 49962;
    tsc(ismember(isxf, dropped{r})) = NaN;
    ft.recording_id = [ft.recording_id; repmat(rid(r), n, 1)];
    ft.assay = [ft.assay; repmat(assay(r), n, 1)];
    ft.isx_frame = [ft.isx_frame; isxf];
    ft.pp_frame = [ft.pp_frame; pp];
    ft.valid = [ft.valid; double(v)];
    ft.invalid_reason = [ft.invalid_reason; reason];
    ft.h5_row = [ft.h5_row; h5r];
    ft.tsc_us = [ft.tsc_us; tsc];
end
syn.frames_csv = fullfile(root, [o.name '_mc_frames.csv']);
write_csv(syn.frames_csv, ft, struct('recording_id', '%s', 'assay', '%s', 'isx_frame', '%d', ...
    'pp_frame', '%d', 'valid', '%d', 'invalid_reason', '%s', 'h5_row', '%d', 'tsc_us', '%d'));
syn.n_invalid = nnz(ft.valid ~= 1);
% Full-table rows of the dip (dip is in valid-frame order).
full_rows = find(ft.valid == 1);
syn.dip_rows = full_rows(dip);
end


function assert_true(cond, what)
if ~cond
    error('extract_1p:selftest', 'selftest FAILED: %s', what);
end
end
