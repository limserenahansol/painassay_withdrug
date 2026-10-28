function info = inscopix_day(cfg)
%INSCOPIX_DAY  One mouse-day of raw nVista .isxd movies -> motion-corrected HDF5 for EXTRACT.
%
%   info = inscopix_day(cfg)        process one mouse-day
%   inscopix_day('selftest')        synthetic test, errors (non-zero exit under -batch) on failure
%
%   cfg.mode selects the path:
%
%   'lossless_mc' (DEFAULT, decided 2026-10-19): before EXTRACT only (1) a lossless copy
%   and (2) motion correction. EXTRACT itself does dF/F, background removal, spatial
%   high-pass, masking and detection downsampling.
%     1. raw TSC timestamps of every recording -> <recording_id>_raw_tsc.csv (as before)
%     2. isxd_to_h5, one call per recording -> <recording_id>_raw.h5:/mov uint16, full
%        sensor frame, every valid frame, verified pixel-exact; <recording_id>_raw_frames.csv,
%        <recording_id>_raw.json. Nothing is trimmed, cropped, binned or filtered.
%     3. motion_correct_h5 of all recordings against ONE template from the reference
%        recording (the first 'pre') -> <recording_id>_mc.h5:/mov (uint16 + lossless gzip by
%        default, same rows as the raw h5), <recording_id>_mc_shifts.csv, _mc_qc.png; <tag>_mc_template.mat,
%        <tag>_mc_qc.png, <tag>_mc.json
%     4. <tag>_mc.h5: the day movie for EXTRACT. /mov is an HDF5 virtual dataset that
%        concatenates the recordings' _mc.h5:/mov (pre then post) without copying them; it
%        reads like an ordinary dataset but needs the _mc.h5 files next to it.
%        /mean_image, /max_image, /frame_mean are real datasets.
%     5. <tag>_mc_frames.csv: one row per raw frame of every recording (columns below plus
%        rec_h5_row = row in that recording's _raw.h5 and _mc.h5), and <tag>_inscopix.json
%     Optional cfg fields (default) of this mode
%       mc_reference      (first 'pre')  index into cfg.recordings of the template recording
%       mc_mode           ('rigid')      | 'piecewise'
%       mc_max_shift_px   (40)  full-resolution px
%       mc_filter_sigma_px (7)
%       mc_estimate_bin   (2)
%       mc_output_class   ('uint16')     | 'single'
%       mc_deflate        (1)            gzip level of the mc movie (lossless), 0 = none
%       delete_raw        (true)         delete <recording_id>_raw.h5 at the end, but only
%                                        when its pixel-exact verification passed and motion
%                                        correction finished. The .isxd stays the original;
%                                        the frame table, checksums and JSON are kept.
%       mc_workers        (8)
%       mc_options        (struct())     any other motion_correct_h5 field
%       h5_chunk          ([])           isxd_to_h5 chunk
%       verify            ('full')       isxd_to_h5 verification pass
%       toolbox_dir       folder of mini2p_toolbox_paths.m
%     Disk (1280 x 800): raw uint16 = 2.05 MB/frame (temporary), mc uint16 + gzip 1 =
%     ~1.16 MB/frame. A 16.5-min session at 30 Hz (~30,000 frames) needs ~61 GB raw while it
%     is processed and keeps ~35 GB mc; a mouse-day (pre + post) keeps ~70 GB.
%
%   'idps_preprocess' (the earlier default, kept): IDPS preprocessing and IDPS motion
%   correction, described below.
%
%   Steps of 'idps_preprocess' (IDPS MATLAB API, +isx):
%     1. raw metadata and raw per-frame TSC timestamps of every recording
%        (isx.export_movie_timestamps_to_csv, time_ref 'tsc')
%     2. isx.preprocess, one call per recording: spatial downsampling, temporal
%        downsampling 1, defective pixels, trim early frames, optional crop.
%        n_trim = raw numTimes - PP numTimes, verified frame by frame on the timestamps
%     3. one mean-projection reference (isx.project_movie) of the reference recording(s)
%     4. isx.motion_correct of ALL recordings in one call against that reference, so pre
%        and post share one reference and one crop rectangle (one pixel grid for EXTRACT)
%     5. <tag>_mc.h5:/mov  single, H x W x n_valid, valid frames only (never zero-filled)
%        <tag>_mc_frames.csv  one row per RAW frame (see "Frame table" below)
%     6. <tag>_inscopix.json  every parameter, count, shift summary and runtime
%   with <tag> = <mouse_id>_d<day>.
%
%   Required cfg fields
%     mouse_id    char, e.g. 'M01'
%     day         numeric day (1 or 2)
%     recordings  struct array, one element per recording, in the order they should be
%                 concatenated (normally pre then post). Fields:
%                   assay         'pre' | 'post'
%                   isxd          path of the raw IDAS movie
%                   recording_id  optional, default <mouse_id>_d<day>_<assay>
%     out_dir     output folder (created if missing)
%
%   Optional cfg fields (default)
%     spatial_downsample_factor  (2)
%     temporal_downsample_factor (1)     anything else is refused: only TDS = 1 keeps the
%                                        per-frame timestamps and the 1:1 frame <-> SYNC
%                                        pulse rule of CONTRACTS.md section 1
%     fix_defective_pixels       (true)
%     trim_early_frames          (true)
%     crop_rect                  ([])    [top_left_y top_left_x bottom_right_y bottom_right_x]
%                                        in raw sensor pixels (IDPS convention)
%     mc_max_translation         (20)    PP pixels
%     mc_low_bandpass_cutoff     (0.004) the cutoffs only filter the copy used to ESTIMATE
%     mc_high_bandpass_cutoff    (0.016) shifts; IDPS shifts the unfiltered movie
%     mc_reference               (1)     index/indices into cfg.recordings whose mean
%                                        projection is the motion-correction reference
%     idps_dir   (fullfile(getenv('ProgramFiles'), 'Inscopix', 'Data Processing'))
%     overwrite  (false)  false re-uses PP / reference / MC files already in out_dir
%     h5_block_frames (500)
%
%   Frame table (<tag>_mc_frames.csv), one row per raw frame of every recording:
%     recording_id, assay, isx_frame  0-based raw IDAS index, including dropped frames
%     pp_frame        0-based index in the PP/MC movie (empty for trimmed frames)
%     valid           1 if the frame is in the h5
%     invalid_reason  '' | trimmed | dropped | cropped | blank | invalid
%     h5_row          1-based index along /mov's 3rd (frame) dimension (MATLAB order;
%                     h5py sees /mov as frames x W x H, 0-based row = h5_row - 1)
%     tsc_us          raw hardware TSC timestamp (us), empty if not available
%
%   The spatial_filter output is deliberately NOT produced: its global-minimum
%   subtraction would make EXTRACT's per-pixel F0 (and so dF/F) meaningless.

if nargin == 1 && (ischar(cfg) || isstring(cfg)) && strcmpi(cfg, 'selftest')
    info = run_selftest();
    return;
end

t_all = tic;
mode = 'lossless_mc';
if isstruct(cfg) && isfield(cfg, 'mode') && ~isempty(cfg.mode)
    mode = lower(char(cfg.mode));
end
if ~ismember(mode, {'lossless_mc', 'idps_preprocess'})
    error('inscopix_day:cfg', 'cfg.mode must be lossless_mc (default) or idps_preprocess, got "%s"', mode);
end
lossless = strcmp(mode, 'lossless_mc');
if lossless
    cfg = apply_defaults_lossless(cfg);
else
    cfg = apply_defaults(cfg);
end
cfg.mode = mode;
ensure_idps(cfg.idps_dir);
if ~isfolder(cfg.out_dir)
    mkdir(cfg.out_dir);
end
tag = sprintf('%s_d%d', cfg.mouse_id, cfg.day);
if lossless
    files = output_files_lossless(cfg.out_dir, tag);
else
    files = output_files(cfg.out_dir, tag);
end

info = struct();
info.module = 'inscopix_day';
info.mode = mode;
info.created = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
info.matlab_version = version;
info.status = 'running';
info.mouse_id = cfg.mouse_id;
info.day = cfg.day;
info.params = cfg;
info.files = files;
info.warnings = {};
info.runtime_s = struct();
try
    if lossless
        info = run_day_lossless(cfg, files, info, tag);
    else
        info = run_day(cfg, files, info);
    end
    info.status = 'ok';
catch err
    info.status = 'failed';
    info.error = err.message;
    info.error_report = getReport(err, 'extended', 'hyperlinks', 'off');
    info.runtime_s.total = toc(t_all);
    write_json(files.json, info);
    rethrow(err);
end
info.runtime_s.total = toc(t_all);
info.peak_ram_gb = peak_ram_gb();
write_json(files.json, info);
fprintf('inscopix_day: %s (%s) done in %.1f s -> %s\n', tag, mode, info.runtime_s.total, files.h5);
end


% =====================================================================================
function info = run_day_lossless(cfg, files, info, tag)
n_rec = numel(cfg.recordings);
R = cell(1, n_rec);

% ---- 1. raw metadata + raw TSC timestamps (IDPS export, for the manifest) ------------
t = tic;
for r = 1:n_rec
    rec = cfg.recordings(r);
    m = read_meta(rec.isxd);
    rec.n_frames_isx = m.n;
    rec.n_dropped = numel(m.dropped);
    rec.raw_dropped = m.dropped;
    rec.raw_blank = m.blank;
    rec.raw_cropped = m.cropped;
    rec.period_s_timing = m.period_s;
    rec.start_utc = m.start_utc;
    rec.start_epoch_s = m.start_epoch_s;
    rec.raw_num_pixels = m.num_pixels;
    rec.raw_data_type = m.data_type;
    rec.acquisition = m.acquisition;
    rec.raw_tsc_csv = fullfile(cfg.out_dir, [rec.recording_id '_raw_tsc.csv']);
    [tsc, note] = export_tsc(rec.isxd, rec.raw_tsc_csv, m.n, cfg.overwrite);
    tsc(~m.valid) = NaN;
    rec.tsc_export = tsc;
    rec.tsc_note = note;
    [rec.period_s_tsc, rec.fps_tsc] = period_from_tsc(tsc);
    R{r} = rec;
end
info.warnings = [info.warnings, day_consistency_warnings(R)];
info.runtime_s.raw_metadata = toc(t);

% ---- 2. lossless copy, one h5 per recording ---------------------------------------
t = tic;
for r = 1:n_rec
    rec = R{r};
    c = struct('isxd', rec.isxd, 'recording_id', rec.recording_id, 'assay', rec.assay, ...
               'name', rec.recording_id, 'out_dir', cfg.out_dir, 'chunk', cfg.h5_chunk, ...
               'verify', cfg.verify, 'overwrite', cfg.overwrite, 'idps_dir', cfg.idps_dir);
    li = isxd_to_h5(c);
    rec.raw_h5 = li.files.h5;
    rec.raw_frames_csv = li.files.frames_csv;
    rec.raw_json = li.files.json;
    rec.raw_h5_bytes = li.h5_bytes;
    rec.lossless_verification = li.verification;
    rec.lossless_reused = li.reused;
    if ~li.verification.passed
        error('inscopix_day:lossless', '%s: lossless verification did not pass', rec.recording_id);
    end
    ft = read_raw_frame_table(rec.raw_frames_csv);
    rec.n_valid = nnz(ft.valid == 1);
    % The TSC read per frame by isxd_to_h5 must be the IDPS export, frame for frame.
    a = ft.tsc_us(:);
    b = rec.tsc_export(:);
    both = isfinite(a) & isfinite(b);
    if any(both) && ~isequal(a(both), b(both))
        error('inscopix_day:tsc', '%s: per-frame TSC of the frame table differs from the IDPS export', ...
              rec.recording_id);
    end
    rec.tsc_check = sprintf('%d frames: frame table TSC == IDPS export', nnz(both));
    if ~any(both)
        rec.tsc_check = 'no TSC to compare';
    end
    rec.ft = ft;
    R{r} = rec;
end
info.runtime_s.lossless = toc(t);

% ---- 3. motion correction, one template for the day ---------------------------------
t = tic;
mc = cfg.mc_options;
mc.inputs = cellfun(@(x) x.raw_h5, R, 'UniformOutput', false);
mc.frames_csv = cellfun(@(x) x.raw_frames_csv, R, 'UniformOutput', false);
mc.names = cellfun(@(x) x.recording_id, R, 'UniformOutput', false);
mc.tag = tag;
mc.out_dir = cfg.out_dir;
mc.template_from = cfg.mc_reference;
mc.mode = cfg.mc_mode;
mc.max_shift_px = cfg.mc_max_shift_px;
mc.filter_sigma_px = cfg.mc_filter_sigma_px;
mc.estimate_bin = cfg.mc_estimate_bin;
mc.output_class = cfg.mc_output_class;
mc.deflate = cfg.mc_deflate;
mc.workers = cfg.mc_workers;
mc.overwrite = cfg.overwrite;
mc.toolbox_dir = cfg.toolbox_dir;
mi = motion_correct_h5(mc);
info.mc = struct('json', mi.files.json, 'template_mat', mi.files.template_mat, 'qc_png', mi.files.qc_png, ...
                 'reused', mi.reused, 'template', mi.template, 'chunk_frames', mi.chunk_frames, ...
                 'runtime_s', mi.runtime_s);
mw = mi.warnings;
if isempty(mw)
    mw = {};
elseif ~iscell(mw)
    mw = cellstr(mw);
end
info.warnings = [info.warnings, reshape(mw, 1, [])];
mrecs = mi.recordings;
if isstruct(mrecs)
    mrecs = num2cell(mrecs);
end
for r = 1:n_rec
    R{r}.mc_h5 = mrecs{r}.mc_h5;
    R{r}.mc_shifts_csv = mrecs{r}.shifts_csv;
    R{r}.mc_qc_png = mrecs{r}.qc_png;
    R{r}.mc_h5_bytes = mrecs{r}.mc_bytes;
    R{r}.mc_shifts = mrecs{r}.stats;
    R{r}.mc_runtime_s = mrecs{r}.runtime_s;
    R{r}.mc_ms_per_frame = mrecs{r}.ms_per_frame;
    hi = h5info(R{r}.mc_h5, '/mov');
    if hi.Dataspace.Size(3) ~= R{r}.n_valid
        error('inscopix_day:mc', '%s: mc.h5 has %d frames, raw has %d', R{r}.recording_id, ...
              hi.Dataspace.Size(3), R{r}.n_valid);
    end
    R{r}.mc_class = mrecs{r}.output_class;
end
info.runtime_s.motion_correct = toc(t);

% ---- 4. day movie: virtual concatenation, no copy -----------------------------------
t = tic;
img = write_day_view(files.h5, R);
info.h5_size = [img.H, img.W, img.n_total];
info.h5_layout = 'virtual dataset over the recordings'' _mc.h5:/mov (pre then post)';
info.runtime_s.day_view = toc(t);

% ---- 5. day frame table --------------------------------------------------------
[ft, counts] = build_day_table(R);
write_csv(files.frames_csv, ft, day_table_formats());
info.frame_counts = counts;
raw_b = sum(cellfun(@(x) x.raw_h5_bytes, R));
mc_b = sum(cellfun(@(x) x.mc_h5_bytes, R));
info.disk = struct('raw_h5_bytes', raw_b, 'mc_h5_bytes', mc_b, 'total_bytes', raw_b + mc_b, ...
    'raw_bytes_per_frame', raw_b / img.n_total, 'mc_bytes_per_frame', mc_b / img.n_total);

% ---- 6. free the disk: delete the verified raw copies ------------------------------
% Only after the lossless check passed AND motion correction wrote every frame (checked
% above). The .isxd is the original; the raw frame table (checksums), _raw.json and the
% verification result stay, so the copy can be re-made and re-verified at any time.
for r = 1:n_rec
    R{r}.raw_h5_deleted = false;
    if cfg.delete_raw && R{r}.lossless_verification.passed && isfile(R{r}.mc_h5) && ...
            isfile(R{r}.raw_h5)
        delete(R{r}.raw_h5);
        R{r}.raw_h5_deleted = true;
        fprintf('inscopix_day: deleted verified %s (%.1f GB freed)\n', R{r}.raw_h5, ...
                R{r}.raw_h5_bytes / 1e9);
    end
end
info.disk.raw_h5_deleted = all(cellfun(@(x) x.raw_h5_deleted, R));
info.disk.kept_bytes = mc_b + raw_b * ~info.disk.raw_h5_deleted;
drop = {'ft', 'tsc_export'};
info.recordings = cellfun(@(x) json_lists(rmfield(x, intersect(fieldnames(x), drop))), R, ...
                          'UniformOutput', false);
end


function img = write_day_view(view_file, R)
% /mov as an HDF5 virtual dataset: rows 1..n_1 map to recording 1's _mc.h5:/mov, the next
% n_2 rows to recording 2's, and so on. Source files in the same folder are named without
% a path, which HDF5 resolves next to the view file (so the folder can be moved).
n = cellfun(@(x) x.n_valid, R);
hi = h5info(R{1}.mc_h5, '/mov');
H = hi.Dataspace.Size(1);
W = hi.Dataspace.Size(2);
cls = R{1}.mc_class;
for r = 2:numel(R)
    if ~strcmp(R{r}.mc_class, cls)
        error('inscopix_day:view', 'mc movies differ in class (%s vs %s)', cls, R{r}.mc_class);
    end
end
N = sum(n);
delete_if_exists(view_file);
vdir = fileparts(view_file);
fid = H5F.create(view_file, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
if strcmp(cls, 'single')
    type = H5T.copy('H5T_NATIVE_FLOAT');
    fill = single(NaN);
else
    type = H5T.copy('H5T_NATIVE_UINT16');
    fill = uint16(0);
end
vspace = H5S.create_simple(3, fliplr([H W N]), []);
dcpl = H5P.create('H5P_DATASET_CREATE');
H5P.set_fill_value(dcpl, type, fill);
off = 0;
src_names = cell(1, numel(R));
for r = 1:numel(R)
    [sdir, nm, ext] = fileparts(R{r}.mc_h5);
    src = [nm ext];
    if ~strcmpi(strip(sdir, 'right', filesep), strip(vdir, 'right', filesep))
        src = R{r}.mc_h5;
    end
    src_names{r} = src;
    sspace = H5S.create_simple(3, fliplr([H W n(r)]), []);
    H5S.select_hyperslab(vspace, 'H5S_SELECT_SET', fliplr([0 0 off]), [], fliplr([H W n(r)]), []);
    H5P.set_virtual(dcpl, vspace, src, '/mov', sspace);
    H5S.close(sspace);
    off = off + n(r);
end
dset = H5D.create(fid, '/mov', type, vspace, 'H5P_DEFAULT', dcpl, 'H5P_DEFAULT');
H5D.close(dset);
H5P.close(dcpl);
H5S.close(vspace);
H5T.close(type);
H5F.close(fid);
% day mean / max / frame_mean from the recordings' own (exact sums of the same frames)
s = zeros(H, W);
mx = -inf(H, W, 'single');
fm = zeros(1, 0, 'single');
for r = 1:numel(R)
    s = s + double(h5read(R{r}.mc_h5, '/mean_image')) * n(r);
    mx = max(mx, single(h5read(R{r}.mc_h5, '/max_image')));
    fm = [fm, single(h5read(R{r}.mc_h5, '/frame_mean'))]; %#ok<AGROW>
end
h5create(view_file, '/mean_image', [H W], 'Datatype', 'single');
h5write(view_file, '/mean_image', single(s / N));
h5create(view_file, '/max_image', [H W], 'Datatype', 'single');
h5write(view_file, '/max_image', mx);
h5create(view_file, '/frame_mean', [1 N], 'Datatype', 'single');
h5write(view_file, '/frame_mean', fm);
h5writeatt(view_file, '/mov', 'recording_ids', strjoin(cellfun(@(x) x.recording_id, R, 'UniformOutput', false), ','));
h5writeatt(view_file, '/mov', 'virtual_sources', strjoin(src_names, ';'));
h5writeatt(view_file, '/mov', 'note', ['virtual dataset: the listed _mc.h5 files must stay next to ', ...
           'this file; rows map to the day frame table h5_row']);
% Read through the view once at every seam: a missing source would read as fill values.
for r = 1:numel(R)
    r0 = sum(n(1:r - 1)) + 1;
    for k = unique([r0, r0 + n(r) - 1])
        a = h5read(view_file, '/mov', [1 1 k], [H W 1]);
        b = h5read(R{r}.mc_h5, '/mov', [1 1 k - r0 + 1], [H W 1]);
        if ~isequal(a, b)
            error('inscopix_day:view', 'day view row %d does not read back %s row %d', k, R{r}.mc_h5, k - r0 + 1);
        end
    end
end
img = struct('H', H, 'W', W, 'n_total', N);
end


function [ft, counts] = build_day_table(R)
cols = {'recording_id', 'assay', 'isx_frame', 'valid', 'invalid_reason', 'h5_row', 'rec_h5_row', 'tsc_us'};
parts = cell(numel(R), numel(cols));
counts = struct('recording_id', {}, 'n_frames_isx', {}, 'n_valid', {}, 'n_trimmed', {}, ...
                'n_dropped', {}, 'n_cropped', {}, 'n_blank', {}, 'n_other_invalid', {});
off = 0;
for r = 1:numel(R)
    t = R{r}.ft;
    n = numel(t.isx_frame);
    if ~isequal(t.isx_frame(:)', 0:n - 1) || n ~= R{r}.n_frames_isx
        error('inscopix_day:table', '%s: raw frame table is not one row per raw frame', R{r}.recording_id);
    end
    rec_row = t.h5_row(:);
    day_row = rec_row + off;
    parts(r, :) = {t.recording_id(:), repmat({R{r}.assay}, n, 1), t.isx_frame(:), t.valid(:), ...
                   t.invalid_reason(:), day_row, rec_row, t.tsc_us(:)};
    off = off + nnz(t.valid == 1);
    reason = t.invalid_reason;
    counts(r) = struct('recording_id', R{r}.recording_id, 'n_frames_isx', n, 'n_valid', nnz(t.valid == 1), ...
        'n_trimmed', 0, 'n_dropped', nnz(strcmp(reason, 'dropped')), 'n_cropped', nnz(strcmp(reason, 'cropped')), ...
        'n_blank', nnz(strcmp(reason, 'blank')), 'n_other_invalid', nnz(strcmp(reason, 'invalid')));
end
ft = struct();
for c = 1:numel(cols)
    ft.(cols{c}) = vertcat(parts{:, c});
end
end


function fmt = day_table_formats()
fmt = struct('recording_id', '%s', 'assay', '%s', 'isx_frame', '%d', 'valid', '%d', ...
             'invalid_reason', '%s', 'h5_row', '%d', 'rec_h5_row', '%d', 'tsc_us', '%d');
end


function ft = read_raw_frame_table(file)
opts = detectImportOptions(file, 'Delimiter', ',', 'TextType', 'char');
txt = intersect({'recording_id', 'assay', 'invalid_reason', 'checksum_sum', 'checksum_wsum'}, opts.VariableNames);
opts = setvartype(opts, txt, 'char');
opts = setvartype(opts, intersect({'isx_frame', 'valid', 'h5_row', 'tsc_us'}, opts.VariableNames), 'double');
tb = readtable(file, opts);
r = tb.invalid_reason;
r(cellfun(@isempty, r)) = {''};
ft = struct('recording_id', {tb.recording_id}, 'isx_frame', tb.isx_frame, 'valid', tb.valid, ...
            'invalid_reason', {r}, 'h5_row', tb.h5_row, 'tsc_us', tb.tsc_us);
end


function g = peak_ram_gb()
g = NaN;
try
    p = System.Diagnostics.Process.GetCurrentProcess();
    g = double(p.PeakWorkingSet64) / 2^30;
catch
end
end


% =====================================================================================
function info = run_day(cfg, files, info)
n_rec = numel(cfg.recordings);
R = cell(1, n_rec);
for r = 1:n_rec
    R{r} = cfg.recordings(r);
end

% ---- 1. raw metadata + raw TSC timestamps ------------------------------------------
t = tic;
for r = 1:n_rec
    rec = R{r};
    m = read_meta(rec.isxd);
    rec.n_frames_isx = m.n;
    rec.raw_dropped = m.dropped;
    rec.raw_cropped = m.cropped;
    rec.raw_blank = m.blank;
    rec.n_dropped = numel(m.dropped);
    rec.raw_valid = m.valid;
    rec.period_s_timing = m.period_s;
    rec.start_utc = m.start_utc;
    rec.start_epoch_s = m.start_epoch_s;
    rec.raw_num_pixels = m.num_pixels;
    rec.raw_data_type = m.data_type;
    rec.acquisition = m.acquisition;
    rec.raw_tsc_csv = fullfile(cfg.out_dir, [rec.recording_id '_raw_tsc.csv']);
    [tsc, note] = export_tsc(rec.isxd, rec.raw_tsc_csv, m.n, cfg.overwrite);
    tsc(~m.valid) = NaN;   % IDPS has no real timestamp for a dropped frame
    rec.tsc_us = tsc;
    rec.tsc_note = note;
    [rec.period_s_tsc, rec.fps_tsc] = period_from_tsc(tsc);
    if isnan(rec.period_s_tsc)
        info.warnings{end+1} = sprintf('%s: no raw TSC timestamps (%s)', rec.recording_id, note);
    elseif abs(rec.period_s_tsc - rec.period_s_timing) > 1e-4
        info.warnings{end+1} = sprintf('%s: TSC period %.6f s differs from timing period %.6f s', ...
            rec.recording_id, rec.period_s_tsc, rec.period_s_timing);
    end
    R{r} = rec;
end
info.warnings = [info.warnings, day_consistency_warnings(R)];
info.runtime_s.raw_metadata = toc(t);

% ---- 2. preprocess, one call per recording -----------------------------------------
% One call per recording so that trim_early_frames acts on each movie's own start and
% n_trim is unambiguous per recording.
t = tic;
pp_files = cell(1, n_rec);
for r = 1:n_rec
    rec = R{r};
    pp_files{r} = fullfile(cfg.out_dir, [rec.recording_id '-PP.isxd']);
    rec.pp_file = pp_files{r};
    if isfile(pp_files{r}) && ~cfg.overwrite
        rec.pp_reused = true;
    else
        delete_if_exists(pp_files{r});
        args = {'temporal_downsample_factor', cfg.temporal_downsample_factor, ...
                'spatial_downsample_factor', cfg.spatial_downsample_factor, ...
                'fix_defective_pixels', cfg.fix_defective_pixels, ...
                'trim_early_frames', cfg.trim_early_frames};
        if ~isempty(cfg.crop_rect)
            args = [args, {'crop_rect', cfg.crop_rect}]; %#ok<AGROW>
        end
        fprintf('  preprocess %s ...\n', rec.recording_id);
        isx.preprocess({rec.isxd}, {pp_files{r}}, args{:});
        rec.pp_reused = false;
    end
    pm = read_meta(pp_files{r});
    % A re-used PP file from a run with other settings would silently change the pixel
    % grid; the size is the one cheap thing that exposes it.
    expect_px = floor(rec.raw_num_pixels / cfg.spatial_downsample_factor);
    if rec.pp_reused && isempty(cfg.crop_rect) && ~isequal(pm.num_pixels, expect_px)
        error('inscopix_day:reuse', ['%s: existing PP is %s, expected %s for these settings; ', ...
              'rerun with cfg.overwrite = true'], rec.recording_id, mat2str(pm.num_pixels), mat2str(expect_px));
    end
    rec.n_pp = pm.n;
    rec.n_trim = rec.n_frames_isx - pm.n;
    if rec.n_trim < 0
        error('inscopix_day:trim', '%s: PP has more frames (%d) than raw (%d)', ...
              rec.recording_id, pm.n, rec.n_frames_isx);
    end
    rec.pp_valid = pm.valid;
    rec.pp_dropped = pm.dropped;
    rec.pp_cropped = pm.cropped;
    rec.pp_blank = pm.blank;
    rec.pp_num_pixels = pm.num_pixels;
    rec.pp_period_s_timing = pm.period_s;

    % The whole frame <-> pulse rule rests on "PP frame s = raw frame s + n_trim".
    % With TDS = 1 IDPS keeps the raw TSC per frame, so check it frame by frame.
    pp_tsc_csv = fullfile(cfg.out_dir, [rec.recording_id '-PP_tsc.csv']);
    [pp_tsc, pp_note] = export_tsc(pp_files{r}, pp_tsc_csv, pm.n, cfg.overwrite);
    [rec.trim_check, msg] = check_trim(rec.tsc_us, pp_tsc, rec.n_trim, pm.valid, pp_note);
    if strcmp(rec.trim_check, 'tsc_mismatch')
        error('inscopix_day:trim', '%s: %s', rec.recording_id, msg);
    elseif ~strcmp(rec.trim_check, 'tsc_exact')
        info.warnings{end+1} = sprintf('%s: %s', rec.recording_id, msg);
    end
    % A PP frame can only be valid if the raw frame it maps to was valid (e.g. not
    % dropped); otherwise IDPS removed frames somewhere other than the start and the
    % shift by n_trim would be wrong.
    shifted = rec.raw_valid(rec.n_trim + 1:end);
    if numel(shifted) ~= numel(pm.valid) || any(pm.valid & ~shifted)
        error('inscopix_day:trim', '%s: PP valid frames are not raw valid frames shifted by n_trim', ...
              rec.recording_id);
    end
    n_extra = nnz(shifted & ~pm.valid);
    if n_extra > 0
        info.warnings{end+1} = sprintf('%s: PP marks %d raw-valid frames invalid (cropped=%s)', ...
            rec.recording_id, n_extra, mat2str(pm.cropped));
    end
    R{r} = rec;
end
info.runtime_s.preprocess = toc(t);

sizes = cellfun(@(x) x.pp_num_pixels, R, 'UniformOutput', false);
if numel(unique(cellfun(@mat2str, sizes, 'UniformOutput', false))) > 1
    error('inscopix_day:size', 'PP movies differ in size: %s', strjoin(cellfun(@mat2str, sizes, ...
          'UniformOutput', false), ' vs '));
end

% ---- 3. mean-projection reference ------------------------------------------------
t = tic;
ref_idx = cfg.mc_reference;
if isfile(files.reference) && ~cfg.overwrite
    info.reference_reused = true;
else
    delete_if_exists(files.reference);
    isx.project_movie(pp_files(ref_idx), files.reference, 'stat_type', 'mean');
    info.reference_reused = false;
end
info.reference_recordings = cellfun(@(x) x.recording_id, R(ref_idx), 'UniformOutput', false);
info.runtime_s.reference = toc(t);

% ---- 4. motion correction, all recordings in one call -----------------------------
t = tic;
mc_files = cell(1, n_rec);
tr_files = cell(1, n_rec);
for r = 1:n_rec
    mc_files{r} = fullfile(cfg.out_dir, [R{r}.recording_id '-PP-MC.isxd']);
    tr_files{r} = fullfile(cfg.out_dir, [R{r}.recording_id '-PP-MC_translations.csv']);
end
mc_args = {'max_translation', cfg.mc_max_translation, ...
           'low_bandpass_cutoff', cfg.mc_low_bandpass_cutoff, ...
           'high_bandpass_cutoff', cfg.mc_high_bandpass_cutoff, ...
           'reference_file_name', files.reference, ...
           'preserve_input_dimensions', false};
have_all = all(cellfun(@isfile, mc_files)) && all(cellfun(@isfile, tr_files)) && isfile(files.crop_rect);
if have_all && ~cfg.overwrite
    info.mc_reused = true;
    info.mc_mode = 'joint';
else
    info.mc_reused = false;
    cellfun(@delete_if_exists, [mc_files, tr_files]);
    delete_if_exists(files.crop_rect);
    fprintf('  motion_correct %d recording(s) against one reference ...\n', n_rec);
    isx.motion_correct(pp_files, mc_files, mc_args{:}, ...
        'output_translation_files', tr_files, 'output_crop_rect_file', files.crop_rect);
    info.mc_mode = 'joint';
end
info.mc_crop_rect_xywh = read_crop_rect(files.crop_rect);
info.runtime_s.motion_correct = toc(t);

for r = 1:n_rec
    rec = R{r};
    rec.mc_file = mc_files{r};
    rec.translations_csv = tr_files{r};
    mm = read_meta(mc_files{r});
    if mm.n ~= rec.n_pp
        error('inscopix_day:mc', '%s: MC has %d frames, PP has %d', rec.recording_id, mm.n, rec.n_pp);
    end
    rec.mc_num_pixels = mm.num_pixels;
    rec.mc_data_type = mm.data_type;
    rec.write_mask = rec.pp_valid(:)' & mm.valid(:)';   % PP/MC frames that go into the h5
    rec.n_valid = nnz(rec.write_mask);
    rec.mc_shifts = summarize_translations(tr_files{r}, rec.write_mask);
    R{r} = rec;
end
sizes = cellfun(@(x) x.mc_num_pixels, R, 'UniformOutput', false);
if numel(unique(cellfun(@mat2str, sizes, 'UniformOutput', false))) > 1
    error('inscopix_day:size', 'MC movies differ in size (%s); one crop for all recordings is required', ...
          strjoin(cellfun(@mat2str, sizes, 'UniformOutput', false), ' vs '));
end

% ---- 5. HDF5 movie + frame table --------------------------------------------------
t = tic;
[R, img] = write_h5(files.h5, R, cfg.h5_block_frames);
info.runtime_s.write_h5 = toc(t);
info.h5_size = [size(img.mean, 1), size(img.mean, 2), img.n_total];
info.h5_bytes = dir(files.h5).bytes;

[ft, counts] = build_frame_table(R);
write_csv(files.frames_csv, ft, frame_table_formats());
info.frame_counts = counts;

% Keep only small, human-readable fields in the JSON (no per-frame vectors).
drop = {'raw_valid', 'tsc_us', 'pp_valid', 'write_mask'};
info.recordings = cellfun(@(x) json_lists(rmfield(x, intersect(fieldnames(x), drop))), R, ...
                          'UniformOutput', false);
end


% =====================================================================================
function cfg = apply_defaults(cfg)
if ~isstruct(cfg)
    error('inscopix_day:cfg', 'cfg must be a struct (or ''selftest'')');
end
need = {'mouse_id', 'day', 'recordings', 'out_dir'};
for k = 1:numel(need)
    if ~isfield(cfg, need{k}) || isempty(cfg.(need{k}))
        error('inscopix_day:cfg', 'cfg.%s is required', need{k});
    end
end
cfg.mouse_id = char(cfg.mouse_id);
cfg.out_dir = char(cfg.out_dir);
def = struct( ...
    'spatial_downsample_factor', 2, ...
    'temporal_downsample_factor', 1, ...
    'fix_defective_pixels', true, ...
    'trim_early_frames', true, ...
    'crop_rect', [], ...
    'mc_max_translation', 20, ...
    'mc_low_bandpass_cutoff', 0.004, ...
    'mc_high_bandpass_cutoff', 0.016, ...
    'mc_reference', 1, ...
    'idps_dir', fullfile(getenv('ProgramFiles'), 'Inscopix', 'Data Processing'), ...
    'overwrite', false, ...
    'h5_block_frames', 500);
f = fieldnames(def);
for k = 1:numel(f)
    if ~isfield(cfg, f{k}) || (isempty(cfg.(f{k})) && ~strcmp(f{k}, 'crop_rect'))
        cfg.(f{k}) = def.(f{k});
    end
end
if cfg.temporal_downsample_factor ~= 1
    error('inscopix_day:tds', ['temporal_downsample_factor must be 1: only TDS = 1 keeps the ', ...
          'per-frame timestamps and the 1:1 frame <-> SYNC pulse mapping']);
end
if ~isempty(cfg.crop_rect) && numel(cfg.crop_rect) ~= 4
    error('inscopix_day:cfg', 'crop_rect must be [top_left_y top_left_x bottom_right_y bottom_right_x]');
end

cfg.recordings = check_recordings(cfg);
if any(cfg.mc_reference < 1 | cfg.mc_reference > numel(cfg.recordings))
    error('inscopix_day:cfg', 'mc_reference must index cfg.recordings');
end
end


function cfg = apply_defaults_lossless(cfg)
if ~isstruct(cfg)
    error('inscopix_day:cfg', 'cfg must be a struct (or ''selftest'')');
end
need = {'mouse_id', 'day', 'recordings', 'out_dir'};
for k = 1:numel(need)
    if ~isfield(cfg, need{k}) || isempty(cfg.(need{k}))
        error('inscopix_day:cfg', 'cfg.%s is required', need{k});
    end
end
cfg.mouse_id = char(cfg.mouse_id);
cfg.out_dir = char(cfg.out_dir);
% Settings of the IDPS path make no sense here; refuse them rather than ignore them.
idps_only = {'spatial_downsample_factor', 'temporal_downsample_factor', 'fix_defective_pixels', ...
             'trim_early_frames', 'crop_rect', 'mc_max_translation', 'mc_low_bandpass_cutoff', ...
             'mc_high_bandpass_cutoff', 'h5_block_frames'};
given = intersect(fieldnames(cfg), idps_only);
if ~isempty(given)
    error('inscopix_day:cfg', ['%s belong(s) to mode ''idps_preprocess''; mode ''lossless_mc'' ', ...
          'copies every pixel unchanged'], strjoin(given, ', '));
end
def = struct( ...
    'mc_reference', [], ...
    'mc_mode', 'rigid', ...
    'mc_max_shift_px', 40, ...
    'mc_filter_sigma_px', 7, ...
    'mc_estimate_bin', 2, ...
    'mc_output_class', 'uint16', ...
    'mc_deflate', 1, ...
    'delete_raw', true, ...
    'mc_workers', 8, ...
    'mc_options', struct(), ...
    'h5_chunk', [], ...
    'verify', 'full', ...
    'idps_dir', fullfile(getenv('ProgramFiles'), 'Inscopix', 'Data Processing'), ...
    'toolbox_dir', fullfile(getenv('USERPROFILE'), 'OneDrive', 'Desktop', 'cursor', 'pain_2plane_split'), ...
    'overwrite', false);
f = fieldnames(def);
may_be_empty = {'mc_reference', 'h5_chunk'};
for k = 1:numel(f)
    if ~isfield(cfg, f{k}) || (isempty(cfg.(f{k})) && ~ismember(f{k}, may_be_empty))
        cfg.(f{k}) = def.(f{k});
    end
end
cfg.recordings = check_recordings(cfg);
if isempty(cfg.mc_reference)
    k = find(strcmp({cfg.recordings.assay}, 'pre'), 1);
    if isempty(k)
        k = 1;
    end
    cfg.mc_reference = k;
end
if ~isscalar(cfg.mc_reference) || cfg.mc_reference < 1 || cfg.mc_reference > numel(cfg.recordings)
    error('inscopix_day:cfg', 'mc_reference must be one index into cfg.recordings');
end
end


function out = check_recordings(cfg)
recs = cfg.recordings;
if ~isstruct(recs) || ~all(isfield(recs, {'assay', 'isxd'}))
    error('inscopix_day:cfg', 'cfg.recordings must be a struct array with fields assay and isxd');
end
out = struct('recording_id', {}, 'assay', {}, 'isxd', {});
for r = 1:numel(recs)
    a = lower(char(recs(r).assay));
    if ~ismember(a, {'pre', 'post'})
        error('inscopix_day:cfg', 'recordings(%d).assay must be pre or post, got "%s"', r, a);
    end
    rid = sprintf('%s_d%d_%s', cfg.mouse_id, cfg.day, a);
    if isfield(recs, 'recording_id') && ~isempty(recs(r).recording_id)
        rid = char(recs(r).recording_id);
    end
    isxd = char(recs(r).isxd);
    if ~isfile(isxd)
        error('inscopix_day:cfg', 'raw movie not found: %s', isxd);
    end
    out(r) = struct('recording_id', rid, 'assay', a, 'isxd', isxd); %#ok<AGROW>
end
if numel(unique({out.recording_id})) < numel(out)
    error('inscopix_day:cfg', 'recording_id values must be unique');
end
end


function ensure_idps(idps_dir)
% The 'Data Processing' folder itself goes on the path (it contains +isx).
if ~isfolder(fullfile(idps_dir, '+isx'))
    error('inscopix_day:idps', 'IDPS MATLAB API not found: %s\\+isx', idps_dir);
end
addpath(idps_dir);
end


function files = output_files_lossless(out_dir, tag)
files = struct( ...
    'h5', fullfile(out_dir, [tag '_mc.h5']), ...
    'frames_csv', fullfile(out_dir, [tag '_mc_frames.csv']), ...
    'json', fullfile(out_dir, [tag '_inscopix.json']));
end


function files = output_files(out_dir, tag)
files = struct( ...
    'h5', fullfile(out_dir, [tag '_mc.h5']), ...
    'frames_csv', fullfile(out_dir, [tag '_mc_frames.csv']), ...
    'json', fullfile(out_dir, [tag '_inscopix.json']), ...
    'reference', fullfile(out_dir, [tag '_mc_reference_mean.isxd']), ...
    'crop_rect', fullfile(out_dir, [tag '_mc_crop_rect.csv']));
end


function m = read_meta(file)
mv = isx.Movie.read(file);
tm = mv.timing;
m.n = double(tm.num_samples);
m.dropped = double(tm.dropped(:)');
m.cropped = double(tm.cropped);
m.blank = double(tm.blank(:)');
m.valid = logical(tm.get_valid_samples_mask());
m.valid = m.valid(:)';
m.period_s = tm.period.secs_float;
m.num_pixels = double(mv.spacing.num_pixels);
m.data_type = mv.data_type;
m.start_epoch_s = NaN;
m.start_utc = '';
try
    m.start_epoch_s = tm.start.to_secs_since_epoch().secs_float;
    m.start_utc = char(datetime(m.start_epoch_s, 'ConvertFrom', 'posixtime', ...
        'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSS''Z'''));
catch
end
m.acquisition = struct();
try
    ai = mv.get_acquisition_info();
    keep = {'MicroscopeType', 'MicroscopeSerialNumber', 'AcquisitionSWVersion', ...
            'ExposureTime_ms_', 'MicroscopeGain', 'MicroscopeFocus', ...
            'MicroscopeEXLED1Power_mw_mm_2_', 'SessionName'};
    for k = 1:numel(keep)
        if isfield(ai, keep{k})
            m.acquisition.(keep{k}) = ai.(keep{k});
        end
    end
catch
end
end


function [tsc, note] = export_tsc(movie_file, csv_file, n_expected, overwrite)
% TSC = hardware counter (us) of the acquisition box; the only per-frame clock.
tsc = NaN(n_expected, 1);
note = 'ok';
try
    if ~isfile(csv_file) || overwrite
        delete_if_exists(csv_file);
        isx.export_movie_timestamps_to_csv({movie_file}, csv_file, 'time_ref', 'tsc');
    end
    [tsc, note] = parse_tsc_csv(csv_file, n_expected);
catch err
    note = ['no timestamps: ' err.message];
end
end


function [tsc, note] = parse_tsc_csv(csv_file, n_expected)
% Rows are placed by their "Local Frame Number", not by row order, so a CSV that leaves
% out dropped frames still lines up with isx_frame.
tsc = NaN(n_expected, 1);
a = readmatrix(csv_file, 'NumHeaderLines', 1);
if size(a, 2) < 4 || isempty(a)
    note = 'timestamp CSV has fewer than 4 columns';
    return;
end
idx = a(:, 3);
if any(idx < 0 | idx >= n_expected | idx ~= round(idx)) || any(diff(idx) <= 0)
    note = 'timestamp CSV local frame numbers are not increasing indices within the movie';
    return;
end
tsc(idx + 1) = a(:, 4);
tsc(tsc <= 0) = NaN;
note = 'ok';
if numel(idx) ~= n_expected
    note = sprintf('ok (%d of %d frames have a timestamp row)', numel(idx), n_expected);
end
end


function [period_s, fps] = period_from_tsc(tsc)
% IDPS's own convention: (last - first valid timestamp) / number of periods between them.
idx = find(isfinite(tsc));
period_s = NaN;
fps = NaN;
if numel(idx) >= 2
    period_s = (tsc(idx(end)) - tsc(idx(1))) / (idx(end) - idx(1)) / 1e6;
    fps = 1 / period_s;
end
end


function [status, msg] = check_trim(raw_tsc, pp_tsc, n_trim, pp_valid, pp_note)
n_pp = numel(pp_tsc);
ok_raw = any(isfinite(raw_tsc));
ok_pp = any(isfinite(pp_tsc));
if ~ok_raw || ~ok_pp
    status = 'no_tsc';
    msg = sprintf('trim not verified on timestamps (raw TSC %d, PP TSC %d; %s); relying on frame counts', ...
                  ok_raw, ok_pp, pp_note);
    return;
end
expect = raw_tsc(n_trim + (1:n_pp));
v = pp_valid(:) & isfinite(pp_tsc(:)) & isfinite(expect(:));
if isequal(pp_tsc(v), expect(v))
    status = 'tsc_exact';
    msg = sprintf('PP frame s = raw frame s + %d verified on %d timestamps', n_trim, nnz(v));
else
    status = 'tsc_mismatch';
    bad = find(v & pp_tsc(:) ~= expect(:), 1);
    msg = sprintf(['PP timestamps are not raw timestamps shifted by n_trim = %d (first mismatch at ', ...
                   'PP frame %d). The frame <-> pulse mapping would be wrong.'], n_trim, bad - 1);
end
end


function w = day_consistency_warnings(R)
% Cheap checks that catch a swapped or mismatched manifest row before hours of processing.
w = {};
t0 = cellfun(@(x) x.start_epoch_s, R);
if all(isfinite(t0)) && any(diff(t0) <= 0)
    w{end+1} = sprintf('recordings are not in chronological order of their start times (%s)', ...
        strjoin(cellfun(@(x) [x.recording_id ' ' x.start_utc], R, 'UniformOutput', false), ', '));
end
% A different e-focus is a different focal plane: one reference and one EXTRACT run
% for both recordings then pairs different cells.
foc = cellfun(@(x) get_field(x.acquisition, 'MicroscopeFocus'), R);
if all(isfinite(foc)) && numel(unique(foc)) > 1
    w{end+1} = sprintf('MicroscopeFocus differs between recordings (%s): different focal planes', ...
        mat2str(foc));
end
px = cellfun(@(x) mat2str(x.raw_num_pixels), R, 'UniformOutput', false);
if numel(unique(px)) > 1
    w{end+1} = sprintf('raw frame sizes differ: %s', strjoin(px, ' vs '));
end
end


function v = get_field(s, f)
v = NaN;
if isstruct(s) && isfield(s, f) && isnumeric(s.(f)) && isscalar(s.(f))
    v = double(s.(f));
end
end


function xywh = read_crop_rect(file)
xywh = [];
if ~isfile(file)
    return;
end
txt = strtrim(fileread(file));
num = regexp(txt, '-?\d+', 'match');
num = str2double(num);
if numel(num) >= 4
    xywh = num(end - 3:end);
end
end


function s = summarize_translations(file, valid)
% IDPS writes one row per PP frame (dropped ones included, with zero shift), so only
% the valid rows are summarised. Column names differ between IDPS versions; pick the
% numeric columns whose names mention x and y.
n_pp = numel(valid);
s = struct('file', file, 'n_rows', 0, 'columns', {{}});
if ~isfile(file)
    s.note = 'translation file missing';
    return;
end
tb = readtable(file, 'VariableNamingRule', 'preserve');
s.n_rows = height(tb);
s.columns = tb.Properties.VariableNames;
names = lower(tb.Properties.VariableNames);
ix = find(contains(names, 'x') & ~contains(names, 'frame') & varfun(@isnumeric, tb, 'OutputFormat', 'uniform'), 1);
iy = find(contains(names, 'y') & ~contains(names, 'frame') & varfun(@isnumeric, tb, 'OutputFormat', 'uniform'), 1);
if isempty(ix) || isempty(iy)
    s.note = 'could not identify x/y translation columns';
    return;
end
dx = tb{:, ix};
dy = tb{:, iy};
if numel(dx) == n_pp
    dx = dx(valid(:));
    dy = dy(valid(:));
end
d = hypot(dx, dy);
s.x_column = tb.Properties.VariableNames{ix};
s.y_column = tb.Properties.VariableNames{iy};
s.rows_match_frames = (s.n_rows == n_pp);
s.x_median = median(dx, 'omitnan');
s.y_median = median(dy, 'omitnan');
s.x_range = [min(dx), max(dx)];
s.y_range = [min(dy), max(dy)];
s.abs_disp_median = median(d, 'omitnan');
s.abs_disp_p95 = prctile(d, 95);
s.abs_disp_max = max(d);
s.frame_to_frame_p95 = prctile(hypot(diff(dx), diff(dy)), 95);
end


function [R, img] = write_h5(h5, R, block)
% Valid frames only: a dropped frame read through get_frame_data would come back as
% zeros, and a zero frame would corrupt EXTRACT's per-pixel mean (its F0).
n_total = sum(cellfun(@(x) x.n_valid, R));
sz = R{1}.mc_num_pixels;
H = sz(1);
W = sz(2);
delete_if_exists(h5);
h5create(h5, '/mov', [H W n_total], 'Datatype', 'single', 'ChunkSize', [H W 1]);
sum_all = zeros(H, W);
max_all = -inf(H, W, 'single');
frame_mean = zeros(1, n_total);
row = 0;
for r = 1:numel(R)
    rec = R{r};
    mv = isx.Movie.read(rec.mc_file);
    idx = find(rec.write_mask) - 1;     % 0-based MC frame indices
    rec.h5_row_first = row + 1;
    s = zeros(H, W);
    for b = 1:block:numel(idx)
        kk = idx(b:min(numel(idx), b + block - 1));
        buf = zeros(H, W, numel(kk), 'single');
        for j = 1:numel(kk)
            f = single(mv.get_frame_data(kk(j)));
            if ~any(f(:))
                error('inscopix_day:zero', '%s: MC frame %d is all zeros but marked valid', ...
                      rec.recording_id, kk(j));
            end
            if any(~isfinite(f(:)))
                error('inscopix_day:nan', '%s: MC frame %d has %d non-finite pixels', ...
                      rec.recording_id, kk(j), nnz(~isfinite(f)));
            end
            buf(:, :, j) = f;
        end
        h5write(h5, '/mov', buf, [1 1 row + 1], [H W numel(kk)]);
        frame_mean(row + (1:numel(kk))) = squeeze(mean(buf, [1 2]))';
        s = s + sum(double(buf), 3);
        max_all = max(max_all, max(buf, [], 3));
        row = row + numel(kk);
    end
    rec.h5_row_last = row;
    sum_all = sum_all + s;
    rm = single(s / max(1, rec.n_valid));
    ds = ['/mean_image_' rec.recording_id];
    h5create(h5, ds, [H W], 'Datatype', 'single');
    h5write(h5, ds, rm);
    fm = frame_mean(rec.h5_row_first:rec.h5_row_last);
    % Dim leading frames that trim_early_frames left behind would show up here.
    rec.first_frames_rel_mean = fm(1:min(5, end)) / median(fm);
    rec.frame_mean_range = [min(fm), max(fm)];
    R{r} = rec;
    clear mv;
end
if row ~= n_total
    error('inscopix_day:h5', 'wrote %d frames, expected %d', row, n_total);
end
img.mean = single(sum_all / max(1, n_total));
img.max = max_all;
img.n_total = n_total;
h5create(h5, '/mean_image', [H W], 'Datatype', 'single');
h5write(h5, '/mean_image', img.mean);
h5create(h5, '/max_image', [H W], 'Datatype', 'single');
h5write(h5, '/max_image', img.max);
h5create(h5, '/frame_mean', [1 n_total], 'Datatype', 'single');
h5write(h5, '/frame_mean', single(frame_mean));
h5writeatt(h5, '/mov', 'recording_ids', strjoin(cellfun(@(x) x.recording_id, R, 'UniformOutput', false), ','));
h5writeatt(h5, '/mov', 'note', 'valid frames only; map rows with the _mc_frames.csv h5_row column');
end


function [ft, counts] = build_frame_table(R)
% One row per RAW frame. Pure function of the per-recording metadata (selftested).
cols = {'recording_id', 'assay', 'isx_frame', 'pp_frame', 'valid', 'invalid_reason', 'h5_row', 'tsc_us'};
parts = cell(numel(R), numel(cols));
counts = struct('recording_id', {}, 'n_frames_isx', {}, 'n_valid', {}, 'n_trimmed', {}, ...
                'n_dropped', {}, 'n_cropped', {}, 'n_blank', {}, 'n_other_invalid', {});
h5_offset = 0;
for r = 1:numel(R)
    rec = R{r};
    n = rec.n_frames_isx;
    isx_frame = (0:n - 1)';
    pp = isx_frame - rec.n_trim;
    in_pp = pp >= 0;
    pp(~in_pp) = NaN;
    valid = false(n, 1);
    valid(in_pp) = rec.write_mask(pp(in_pp) + 1);
    reason = repmat({''}, n, 1);
    reason(~in_pp) = {'trimmed'};

    raw_dropped = false(n, 1);
    raw_dropped(rec.raw_dropped + 1) = true;
    pp_dropped = false(n, 1);
    pp_cropped = false(n, 1);
    pp_blank = false(n, 1);
    pd = rec.pp_dropped(:) + rec.n_trim + 1;
    pp_dropped(pd(pd <= n)) = true;
    lead_cropped = false(n, 1);
    for c = 1:size(rec.pp_cropped, 1)
        rr = (rec.pp_cropped(c, 1):rec.pp_cropped(c, 2)) + rec.n_trim + 1;
        rr = rr(rr <= n);
        pp_cropped(rr) = true;
        if rec.pp_cropped(c, 1) == 0
            % A cropped run at the very start is IDPS's other way of trimming early frames.
            lead_cropped(rr) = true;
        end
    end
    pb = rec.pp_blank(:) + rec.n_trim + 1;
    pp_blank(pb(pb <= n)) = true;

    bad = in_pp & ~valid;
    reason(bad & (raw_dropped | pp_dropped)) = {'dropped'};
    still = bad & strcmp(reason, '');
    reason(still & lead_cropped) = {'trimmed'};
    still = bad & strcmp(reason, '');
    reason(still & pp_cropped) = {'cropped'};
    still = bad & strcmp(reason, '');
    reason(still & pp_blank) = {'blank'};
    still = bad & strcmp(reason, '');
    reason(still) = {'invalid'};

    h5_row = NaN(n, 1);
    h5_row(valid) = h5_offset + (1:nnz(valid))';
    h5_offset = h5_offset + nnz(valid);

    tsc = NaN(n, 1);
    if isfield(rec, 'tsc_us') && numel(rec.tsc_us) == n
        tsc = rec.tsc_us(:);
    end
    parts(r, :) = {repmat({rec.recording_id}, n, 1), repmat({rec.assay}, n, 1), isx_frame, ...
                   pp, double(valid), reason, h5_row, tsc};
    counts(r) = struct('recording_id', rec.recording_id, 'n_frames_isx', n, 'n_valid', nnz(valid), ...
        'n_trimmed', nnz(strcmp(reason, 'trimmed')), 'n_dropped', nnz(strcmp(reason, 'dropped')), ...
        'n_cropped', nnz(strcmp(reason, 'cropped')), 'n_blank', nnz(strcmp(reason, 'blank')), ...
        'n_other_invalid', nnz(strcmp(reason, 'invalid')));
end
ft = struct();
for c = 1:numel(cols)
    ft.(cols{c}) = vertcat(parts{:, c});
end
end


function fmt = frame_table_formats()
fmt = struct('recording_id', '%s', 'assay', '%s', 'isx_frame', '%d', 'pp_frame', '%d', ...
             'valid', '%d', 'invalid_reason', '%s', 'h5_row', '%d', 'tsc_us', '%d');
end


function write_csv(file, ft, fmt)
% Explicit writer: NaN -> empty field, integers without decimals, fixed column order.
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
    error('inscopix_day:io', 'cannot write %s', file);
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', strjoin(cols', ','));
fprintf(fid, '%s\n', join(txt, ',', 2));
end


function write_json(file, s)
fid = fopen(file, 'w');
if fid < 0
    error('inscopix_day:io', 'cannot write %s', file);
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(s, 'PrettyPrint', true));
end


function rec = json_lists(rec)
% jsonencode writes a 1-element vector as a bare number; index lists must stay lists.
for f = {'raw_dropped', 'raw_blank', 'pp_dropped', 'pp_blank'}
    if isfield(rec, f{1})
        rec.(f{1}) = num2cell(rec.(f{1})(:)');
    end
end
end


function delete_if_exists(f)
if isfile(f)
    delete(f);
end
end


% =====================================================================================
function info = run_selftest()
% 1) frame-table logic with trimmed + dropped + leading-cropped frames (no IDPS needed)
% 2) CSV writer round trip
% 3) mode 'idps_preprocess' end to end on two synthetic .isxd movies written with the IDPS
%    API (one with a dropped frame and a known drift), if IDPS is installed
% 4) mode 'lossless_mc' (the default) end to end on the same movies: lossless raw h5,
%    one-template motion correction, the virtual day movie and the day frame table
fprintf('inscopix_day selftest\n');
here = fileparts(mfilename('fullpath'));
root = fullfile(here, 'out', 'selftest_inscopix_day');
if isfolder(root)
    rmdir(root, 's');
end
mkdir(root);

% ---- 1. frame table logic --------------------------------------------------------
rec.recording_id = 'T01_d1_pre';
rec.assay = 'pre';
rec.n_frames_isx = 12;
rec.n_trim = 2;
rec.raw_dropped = 6;                            % raw frame 6 -> PP frame 4
rec.pp_dropped = 4;
rec.pp_cropped = [0 0];                         % PP frame 0 (raw 2) cropped at start
rec.pp_blank = [];
rec.write_mask = true(1, 10);
rec.write_mask([1 5]) = false;
rec.tsc_us = (1:12)' * 50000;
rec2 = rec;
rec2.recording_id = 'T01_d1_post';
rec2.assay = 'post';
rec2.n_trim = 0;
rec2.n_frames_isx = 5;
rec2.raw_dropped = [];
rec2.pp_dropped = [];
rec2.pp_cropped = zeros(0, 2);
rec2.write_mask = true(1, 5);
rec2.tsc_us = NaN(5, 1);
[ft, counts] = build_frame_table({rec, rec2});
expect_reason = [{'trimmed'; 'trimmed'; 'trimmed'; ''; ''; ''; 'dropped'}; repmat({''}, 5, 1); repmat({''}, 5, 1)];
assert_true(isequal(ft.invalid_reason, expect_reason), 'invalid_reason pattern');
assert_true(isequal(ft.isx_frame(1:12)', 0:11) && isequal(ft.isx_frame(13:end)', 0:4), 'isx_frame index');
assert_true(isequaln(ft.h5_row', [NaN NaN NaN 1 2 3 NaN 4 5 6 7 8 9 10 11 12 13]), 'h5_row');
assert_true(isequaln(ft.pp_frame(1:4)', [NaN NaN 0 1]), 'pp_frame');
assert_true(counts(1).n_valid == 8 && counts(1).n_trimmed == 3 && counts(1).n_dropped == 1, 'counts');
fprintf('  frame table logic: ok\n');

% ---- 2. CSV writer ------------------------------------------------------------
csv = fullfile(root, 'ft.csv');
write_csv(csv, ft, frame_table_formats());
tb = readtable(csv, 'Delimiter', ',', 'TextType', 'char');
assert_true(height(tb) == numel(ft.isx_frame), 'csv rows');
assert_true(isequaln(tb.h5_row', ft.h5_row'), 'csv NaN round trip');
assert_true(strcmp(tb.invalid_reason{7}, 'dropped'), 'csv text column');
fprintf('  csv round trip: ok\n');

% ---- 2b. TSC CSV with a dropped frame left out ------------------------------------
tsc_csv = fullfile(root, 'tsc.csv');
fid = fopen(tsc_csv, 'w');
fprintf(fid, 'Global Frame Number,Movie Number,Local Frame Number,Frame Timestamp (us)\n');
fprintf(fid, '%d,0,%d,%d\n', [0 1 3 4; 0 1 3 4; 1000 51000 151000 201000]);
fclose(fid);
[tsc, note] = parse_tsc_csv(tsc_csv, 5);
assert_true(isequaln(tsc', [1000 51000 NaN 151000 201000]) && startsWith(note, 'ok'), ...
            'TSC rows placed by local frame number');
fprintf('  TSC csv with a missing row: ok\n');

% ---- 3. IDPS end to end ------------------------------------------------------
idps = fullfile(getenv('ProgramFiles'), 'Inscopix', 'Data Processing');
if ~isfolder(fullfile(idps, '+isx'))
    fprintf('  IDPS not installed: end-to-end part SKIPPED\n');
    info = struct('selftest', 'ok (IDPS part skipped)');
    return;
end
addpath(idps);
[raw_pre, raw_post, truth] = make_synthetic_isxd(root);
cfg = struct();
cfg.mouse_id = 'T01';
cfg.day = 1;
cfg.recordings = struct('assay', {'pre', 'post'}, 'isxd', {raw_pre, raw_post});
cfg.out_dir = fullfile(root, 'day');
cfg.mode = 'idps_preprocess';
cfg.mc_max_translation = 10;
out = inscopix_day(cfg);

ft = readtable(out.files.frames_csv, 'Delimiter', ',', 'TextType', 'char');
info_h5 = h5info(out.files.h5, '/mov');
n_h5 = info_h5.Dataspace.Size(3);
assert_true(n_h5 == nnz(ft.valid), 'h5 frames == valid rows');
assert_true(height(ft) == truth.n_pre + truth.n_post, 'one row per raw frame');
pre = strcmp(ft.recording_id, 'T01_d1_pre');
row_drop = find(pre & ft.isx_frame == truth.dropped_pre);
assert_true(~ft.valid(row_drop) && strcmp(ft.invalid_reason{row_drop}, 'dropped') ...
            && isnan(ft.h5_row(row_drop)), 'dropped frame flagged and not written');
mov = h5read(out.files.h5, '/mov');
assert_true(all(squeeze(max(max(mov, [], 1), [], 2)) > 0), 'no zero-filled frames in the h5');
% The pixel pattern encodes the raw frame number in its brightness, so the h5 rows
% must come back in raw order with the dropped frame skipped.
fm = double(h5read(out.files.h5, '/frame_mean'));
v = pre & ft.valid == 1;
step = diff(fm(ft.h5_row(v)));
expect = truth.ramp * diff(ft.isx_frame(v))';
assert_true(all(abs(step - expect) < truth.ramp / 2), 'h5 rows are raw frames in order, dropped one skipped');
rp = out.recordings{1};
assert_true(rp.n_trim == truth.n_pre - rp.n_pp, 'n_trim logged');
assert_true(isfield(rp.mc_shifts, 'abs_disp_max'), 'shift summary present');
fprintf('  IDPS end to end: ok (n_trim pre=%d post=%d, h5 %s, max shift %.2f px)\n', ...
        out.recordings{1}.n_trim, out.recordings{2}.n_trim, mat2str(out.h5_size), rp.mc_shifts.abs_disp_max);

% ---- 4. lossless_mc end to end ----------------------------------------------------
cfg = struct('mouse_id', 'T01', 'day', 1, 'out_dir', fullfile(root, 'lossless'), ...
             'mc_max_shift_px', 10, 'mc_filter_sigma_px', 3, 'mc_workers', 2, ...
             'mc_options', struct('template_blocks', 3, 'template_block_frames', 10, 'chunk_frames', 20), ...
             'delete_raw', false);           % keep the raw copy here: it is checked below
cfg.recordings = struct('assay', {'pre', 'post'}, 'isxd', {raw_pre, raw_post});
out = inscopix_day(cfg);
assert_true(strcmp(out.mode, 'lossless_mc') && strcmp(out.status, 'ok'), 'lossless_mc is the default mode');
ft = readtable(out.files.frames_csv, 'Delimiter', ',', 'TextType', 'char');
n_valid = nnz(ft.valid == 1);
assert_true(height(ft) == truth.n_pre + truth.n_post && n_valid == truth.n_pre + truth.n_post - 1, ...
            'day table: one row per raw frame, only the dropped frame invalid');
pre = strcmp(ft.recording_id, 'T01_d1_pre');
row_drop = find(pre & ft.isx_frame == truth.dropped_pre);
assert_true(strcmp(ft.invalid_reason{row_drop}, 'dropped') && isnan(ft.h5_row(row_drop)), 'dropped frame not written');
assert_true(isequal(ft.h5_row(ft.valid == 1)', 1:n_valid), 'day h5_row 1..n in table order');
post = ~pre & ft.valid == 1;
assert_true(isequal(ft.rec_h5_row(post)', 1:nnz(post)), 'rec_h5_row restarts at 1 for post');
hi = h5info(out.files.h5, '/mov');
assert_true(isequal(hi.Dataspace.Size, [120 160 n_valid]), 'day movie is H x W x all valid frames');
day = h5read(out.files.h5, '/mov');
r_post = out.recordings{2};
mp = h5read(r_post.mc_h5, '/mov');
assert_true(isequal(day(:, :, ft.h5_row(post)), mp(:, :, ft.rec_h5_row(post))), ...
            'day movie rows are the post _mc.h5 rows (virtual concatenation)');
assert_true(all(squeeze(max(max(day, [], 1), [], 2)) > 0), 'no zero frames in the day movie');
% the raw h5 holds the pixels exactly as written
r_pre = out.recordings{1};
raw = h5read(r_pre.raw_h5, '/mov');
mv = isx.Movie.read(raw_pre);
k = truth.dropped_pre + 1;                      % first frame after the dropped one (0-based)
fr = mv.get_frame_data(k);
rft = readtable(r_pre.raw_frames_csv, 'Delimiter', ',', 'TextType', 'char');
assert_true(isa(raw, 'uint16') && isequal(raw(:, :, rft.h5_row(k + 1)), fr), 'raw h5 frame == isx frame');
assert_true(r_pre.lossless_verification.passed && r_pre.lossless_verification.max_abs_diff == 0, ...
            'lossless verification logged');
% both recordings registered to one template. make_synthetic_isxd crops at +d, so the
% content moves by -d and the applied correction is +d: shift - d is one constant.
c = [];
for rr = 1:2
    sh = readtable(out.recordings{rr}.mc_shifts_csv, 'Delimiter', ',', 'TextType', 'char');
    k = sh.isx_frame;
    c = [c; [sh.x_shift_px - round(3 * cos(k / 9)), sh.y_shift_px - round(3 * sin(k / 7))]]; %#ok<AGROW>
end
dev = hypot(c(:, 1) - median(c(:, 1)), c(:, 2) - median(c(:, 2)));
assert_true(max(dev) < 0.5, sprintf('pre and post on one template (max error %.2f px)', max(dev)));
js = jsondecode(fileread(out.files.json));
recs = js.recordings;
if iscell(recs)
    recs = recs{1};
end
assert_true(isfield(js.files, 'h5') && isfield(js.files, 'frames_csv') && ...
            isfield(recs(1), 'period_s_timing'), 'day JSON has what extract_1p reads');
fprintf('  lossless_mc end to end: ok (day movie %s, MC error max %.2f px)\n', mat2str(hi.Dataspace.Size), max(dev));
mi = h5info(out.recordings{1}.mc_h5, '/mov');
assert_true(strcmp(mi.Datatype.Type, 'H5T_STD_U16LE') && any(strcmp({mi.Filters.Name}, 'deflate')), ...
            'mc movie is uint16 + gzip by default');

% ---- 5. default delete_raw: the verified raw copy is removed, the mc movie stays ------
cfg.out_dir = fullfile(root, 'lossless_delete');
cfg = rmfield(cfg, 'delete_raw');
out = inscopix_day(cfg);
for rr = 1:2
    assert_true(~isfile(out.recordings{rr}.raw_h5) && isfile(out.recordings{rr}.mc_h5) && ...
                out.recordings{rr}.raw_h5_deleted, 'raw.h5 deleted after verification + MC');
    assert_true(isfile(out.recordings{rr}.raw_frames_csv), 'raw frame table (checksums) kept');
end
day2 = h5read(out.files.h5, '/mov');
assert_true(isequal(day2, day), 'same day movie whether or not raw.h5 is kept');
fprintf('  delete_raw: raw copies removed, day movie unchanged\n');
info = struct('selftest', 'ok');
fprintf('inscopix_day selftest PASSED\n');
end


function [f_pre, f_post, truth] = make_synthetic_isxd(root)
% Smooth random texture plus blobs, drifting by a known amount; brightness ramps with
% the frame number so that frame order is checkable after the round trip.
rng(1);
H = 120;
W = 160;
truth.n_pre = 60;
truth.n_post = 50;
truth.dropped_pre = 10;
truth.ramp = 20;
base = imgaussfilt(rand(H + 40, W + 40) * 4000, 3) + 500;
f_pre = fullfile(root, 'synthetic_pre.isxd');
f_post = fullfile(root, 'synthetic_post.isxd');
write_one(f_pre, truth.n_pre, truth.dropped_pre, 0);
write_one(f_post, truth.n_post, [], truth.n_pre * 0.05 + 30);

    function write_one(file, n, dropped, t0)
        start = isx.Time.from_secs_since_epoch(isx.Duration.from_secs(1.7e9 + t0));
        targs = {'num_samples', n, 'period', isx.Duration.from_msecs(50), 'start', start};
        if ~isempty(dropped)
            targs = [targs, {'dropped', dropped}]; %#ok<AGROW>
        end
        tm = isx.Timing(targs{:});
        sp = isx.Spacing('num_pixels', [H, W]);
        mv = isx.Movie.write(file, tm, sp, 'uint16');
        for k = 0:n - 1
            if ismember(k, dropped)
                continue;
            end
            dy = round(3 * sin(k / 7));
            dx = round(3 * cos(k / 9));
            fr = base(21 + dy:20 + dy + H, 21 + dx:20 + dx + W) + truth.ramp * k;
            mv.set_frame_data(k, uint16(fr));
        end
        mv.flush();
        clear mv;
    end
end


function assert_true(cond, what)
if ~cond
    error('inscopix_day:selftest', 'selftest FAILED: %s', what);
end
end
