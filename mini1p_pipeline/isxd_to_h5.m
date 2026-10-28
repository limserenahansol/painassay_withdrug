function info = isxd_to_h5(cfg)
%ISXD_TO_H5  Lossless copy of raw IDAS .isxd movie(s) into one HDF5 file for EXTRACT.
%
%   info = isxd_to_h5(cfg)      convert
%   isxd_to_h5('selftest')      synthetic test, errors (non-zero exit under -batch) on failure
%
%   Every recorded pixel value is copied unchanged: uint16, full sensor frame, every valid
%   frame in raw order. No spatial or temporal downsampling, no crop, no defective-pixel
%   fix, no early-frame trimming, no filtering. Invalid frames (IDPS timing: dropped,
%   cropped, blank) have no pixel data and are LEFT OUT of the h5 (never zero-filled: a
%   zero frame corrupts EXTRACT's per-pixel F0); the frame table records them.
%
%   Frames are read with the IDPS 1.9.6 MATLAB API (isx.Movie.read, get_frame_data) and
%   streamed block by block, so memory stays constant whatever the movie length.
%   A verification pass then re-reads EVERY frame from the closed h5 and from the .isxd
%   (get_frame_data again) and requires max |difference| == 0 and equal checksums.
%   verify_lossless.py repeats the check without IDPS, from the .isxd bytes.
%
%   Required cfg fields
%     isxd          char or cellstr: one or more raw movies, concatenated in this order
%     out_dir       output folder (created if missing)
%   Optional (default)
%     recording_id  char or cellstr, one per isxd (file name stem)
%     assay         char or cellstr, one per isxd ('')
%     name          output stem (recording_id when there is one movie, else required)
%     chunk         ([]) HDF5 chunk [rows cols frames]; [] = [ceil(H/8) ceil(W/8) block_frames]
%                   (100 x 160 x 50 for 800 x 1280): spatial tiles keep EXTRACT's partition
%                   reads (a block of pixels over all frames) from reading the whole file,
%                   and whole-frame block reads stay whole-chunk reads
%     block_frames  (50)    frames buffered per write; the chunk depth by default
%     deflate       (0)     0 = no compression; 1-9 = lossless gzip (+shuffle), slower
%     verify        ('full') 'full' | 'none'
%     max_frames    (inf)   copy only the first max_frames frames of each movie (tests)
%     overwrite     (false) false re-uses an earlier output of the same sources whose JSON
%                   says status ok and verification passed
%     idps_dir      (fullfile(getenv('ProgramFiles'), 'Inscopix', 'Data Processing'))
%
%   Outputs, with <name> as above
%     <name>_raw.h5          /mov  uint16 [H W n_valid] (MATLAB order; EXTRACT reads it as
%                                  'file.h5:/mov', frame k = mov(:,:,k); h5py sees
%                                  (n_valid, W, H): frame = ds[k].T)
%                            /mean_image (single), /max_image (uint16), /frame_mean (single)
%     <name>_raw_frames.csv  one row per RAW frame of every movie (CONTRACTS.md frame table):
%                            recording_id, assay, isx_frame (0-based, dropped included),
%                            valid, invalid_reason ('' | dropped | cropped | blank),
%                            h5_row (1-based along /mov's 3rd dim; empty if invalid),
%                            tsc_us (per-frame hardware TSC, empty if none),
%                            checksum_sum   = sum of the frame's pixel values (uint64),
%                            checksum_wsum  = sum over i of i * v(i), i = 1..H*W in MATLAB
%                                             column-major order (= numpy ravel of ds[k])
%     <name>_raw.json        sources, metadata, counts, chunking, sizes, runtimes, peak RAM,
%                            verification result
%   The h5 is written as <name>_raw.h5.partial and renamed only after it is complete, so a
%   crash never leaves a file that looks finished.

if nargin == 1 && (ischar(cfg) || isstring(cfg)) && strcmpi(cfg, 'selftest')
    info = run_selftest();
    return;
end

t_all = tic;
cfg = apply_defaults(cfg);
ensure_idps(cfg.idps_dir);
if ~isfolder(cfg.out_dir)
    mkdir(cfg.out_dir);
end
files = output_files(cfg.out_dir, cfg.name);

prev = reusable(files, cfg);
if ~isempty(prev)
    info = prev;
    info.reused = true;
    fprintf('isxd_to_h5: %s re-used (verified %s)\n', cfg.name, prev.created);
    return;
end

info = struct();
info.module = 'isxd_to_h5';
info.created = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
info.matlab_version = version;
info.status = 'running';
info.name = cfg.name;
info.params = rmfield(cfg, 'idps_dir');
info.idps_dir = cfg.idps_dir;
info.files = files;
info.lossless = ['uint16 pixel values copied unchanged; no downsampling, crop, ', ...
                 'defective-pixel fix, trimming or filtering; invalid frames left out'];
info.warnings = {};
info.runtime_s = struct();
info.reused = false;
try
    info = run_convert(cfg, files, info);
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
fprintf('isxd_to_h5: %s %d frames in %.1f s -> %s (%.2f GB)\n', cfg.name, info.n_valid_total, ...
        info.runtime_s.total, files.h5, info.h5_bytes / 1e9);
end


% =====================================================================================
function info = run_convert(cfg, files, info)
% ---- metadata of every movie -----------------------------------------------------
t = tic;
n_rec = numel(cfg.isxd);
R = cell(1, n_rec);
for r = 1:n_rec
    R{r} = read_meta(cfg.isxd{r});
    if R{r}.n > cfg.max_frames                      % test runs: the first max_frames only
        k = cfg.max_frames;
        R{r}.n_frames_in_file = R{r}.n;
        R{r}.n = k;
        R{r}.valid = R{r}.valid(1:k);
        R{r}.reason = R{r}.reason(1:k);
        R{r}.dropped = R{r}.dropped(R{r}.dropped < k);
        R{r}.blank = R{r}.blank(R{r}.blank < k);
        R{r}.n_dropped = numel(R{r}.dropped);
        info.warnings{end + 1} = sprintf('%s: only the first %d of %d frames (max_frames)', ...
            cfg.isxd{r}, k, R{r}.n_frames_in_file);
    end
    R{r}.recording_id = cfg.recording_id{r};
    R{r}.assay = cfg.assay{r};
end
sz = R{1}.num_pixels;
for r = 1:n_rec
    if ~strcmp(R{r}.data_type, 'uint16')
        error('isxd_to_h5:dtype', '%s: data type %s; only uint16 raw movies are supported', ...
              R{r}.file, R{r}.data_type);
    end
    if ~isequal(R{r}.num_pixels, sz)
        error('isxd_to_h5:size', 'frame sizes differ: %s is %s, %s is %s', R{1}.file, ...
              mat2str(sz), R{r}.file, mat2str(R{r}.num_pixels));
    end
end
H = sz(1);
W = sz(2);
n_valid = cellfun(@(x) nnz(x.valid), R);
N = sum(n_valid);
if N == 0
    error('isxd_to_h5:empty', 'no valid frames');
end
info.runtime_s.metadata = toc(t);
info.frame_size_hw = [H W];
info.n_valid_total = N;

% ---- create the h5 ---------------------------------------------------------------
chunk = cfg.chunk;
if isempty(chunk)
    chunk = [ceil(H / 8), ceil(W / 8), cfg.block_frames];
end
chunk = min(chunk, [H W N]);
info.chunk = chunk;
tmp = [files.h5 '.partial'];
delete_if_exists(tmp);
args = {'Datatype', 'uint16', 'ChunkSize', chunk};
if cfg.deflate > 0
    args = [args, {'Deflate', cfg.deflate, 'Shuffle', true}];
end
h5create(tmp, '/mov', [H W N], args{:});

% ---- stream every valid frame ----------------------------------------------------
t = tic;
idx64 = uint64((1:H * W)');
sum_all = zeros(H, W);
max_all = zeros(H, W, 'uint16');
frame_mean = zeros(1, N, 'single');
row = 0;
peak = 0;
for r = 1:n_rec
    rec = R{r};
    mv = isx.Movie.read(rec.file);
    n = rec.n;
    rec.tsc_us = NaN(n, 1);
    rec.checksum_sum = zeros(n, 1, 'uint64');
    rec.checksum_wsum = zeros(n, 1, 'uint64');
    rec.h5_row = NaN(n, 1);
    kv = find(rec.valid) - 1;                       % 0-based valid frames
    rec.h5_row_first = row + 1;
    n_tsc_fail = 0;
    % Movies written by IDPS itself (e.g. the selftest's) store no TSC: probe once, so the
    % C library does not print an exception for every frame.
    rec.has_tsc = false;
    if ~isempty(kv)
        try
            mv.get_frame_timestamp(kv(1));
            rec.has_tsc = true;
        catch
        end
    end
    fprintf('  %s: %d of %d frames -> h5 rows %d..%d\n', rec.recording_id, numel(kv), n, row + 1, row + numel(kv));
    for b = 1:cfg.block_frames:numel(kv)
        kk = kv(b:min(numel(kv), b + cfg.block_frames - 1));
        nb = numel(kk);
        buf = zeros(H, W, nb, 'uint16');
        for j = 1:nb
            k = kk(j);
            fr = mv.get_frame_data(k);
            if ~isa(fr, 'uint16') || ~isequal(size(fr), [H W])
                error('isxd_to_h5:frame', '%s frame %d: got %s %s, expected uint16 %dx%d', ...
                      rec.recording_id, k, class(fr), mat2str(size(fr)), H, W);
            end
            if ~any(fr(:))
                error('isxd_to_h5:zero', ['%s frame %d is all zeros but IDPS marks it valid; a zero ', ...
                      'frame would corrupt EXTRACT''s F0. Check the recording.'], rec.recording_id, k);
            end
            buf(:, :, j) = fr;
            v = uint64(fr(:));
            rec.checksum_sum(k + 1) = sum(v, 'native');
            rec.checksum_wsum(k + 1) = sum(v .* idx64, 'native');
            if rec.has_tsc
                try
                    rec.tsc_us(k + 1) = double(mv.get_frame_timestamp(k));
                catch
                    n_tsc_fail = n_tsc_fail + 1;
                end
            end
        end
        h5write(tmp, '/mov', buf, [1 1 row + 1], [H W nb]);
        rec.h5_row(kk + 1) = row + (1:nb)';
        sum_all = sum_all + sum(double(buf), 3);
        max_all = max(max_all, max(buf, [], 3));
        frame_mean(row + (1:nb)) = squeeze(mean(single(buf), [1 2]))';
        row = row + nb;
        if mod(b - 1, 20 * cfg.block_frames) == 0
            peak = max(peak, peak_ram_gb());
            fprintf('    %d / %d frames (%.0f s)\n', b - 1 + nb, numel(kv), toc(t));
        end
    end
    clear mv;
    rec.h5_row_last = row;
    rec.tsc_us(rec.tsc_us <= 0) = NaN;
    rec.n_tsc_read_errors = n_tsc_fail;
    rec.n_tsc_missing_valid = nnz(rec.valid(:) & ~isfinite(rec.tsc_us(:)));
    [rec.period_s_tsc, rec.fps_tsc] = period_from_tsc(rec.tsc_us);
    if ~rec.has_tsc
        info.warnings{end+1} = sprintf('%s: the movie stores no per-frame TSC timestamps', rec.recording_id);
    elseif rec.n_tsc_missing_valid > 0
        info.warnings{end+1} = sprintf('%s: %d valid frames have no TSC timestamp', ...
            rec.recording_id, rec.n_tsc_missing_valid);
    end
    R{r} = rec;
end
if row ~= N
    error('isxd_to_h5:count', 'wrote %d frames, expected %d', row, N);
end
info.runtime_s.convert = toc(t);
info.convert_ms_per_frame = 1000 * info.runtime_s.convert / N;

h5create(tmp, '/mean_image', [H W], 'Datatype', 'single');
h5write(tmp, '/mean_image', single(sum_all / N));
h5create(tmp, '/max_image', [H W], 'Datatype', 'uint16');
h5write(tmp, '/max_image', max_all);
h5create(tmp, '/frame_mean', [1 N], 'Datatype', 'single');
h5write(tmp, '/frame_mean', frame_mean);
h5writeatt(tmp, '/mov', 'recording_ids', strjoin(cellfun(@(x) x.recording_id, R, 'UniformOutput', false), ','));
h5writeatt(tmp, '/mov', 'source_isxd', strjoin(cellfun(@(x) x.file, R, 'UniformOutput', false), ';'));
h5writeatt(tmp, '/mov', 'lossless', info.lossless);
h5writeatt(tmp, '/mov', 'orientation', ['MATLAB [H W T] = h5py (T, W, H); frame k (1-based) = ', ...
           'mov(:,:,k) = ds[k-1].T; rows of the frame table name it by h5_row']);
h5writeatt(tmp, '/mov', 'frame_table', [cfg.name '_raw_frames.csv']);

% ---- frame table -----------------------------------------------------------------
ft = build_frame_table(R);
write_frame_table(files.frames_csv, ft);

% ---- verification: every frame, h5 vs IDPS ---------------------------------------
if strcmpi(cfg.verify, 'full')
    t = tic;
    v = verify_against_isxd(tmp, R, ft, cfg.block_frames);
    v.runtime_s = toc(t);
    info.verification = v;
    if ~v.passed
        error('isxd_to_h5:verify', ['LOSSLESS CHECK FAILED: %d frames compared, max |diff| = %g, ', ...
              '%d checksum mismatches, %d zero frames'], v.n_frames_compared, v.max_abs_diff, ...
              v.n_checksum_mismatch, v.n_zero_frames);
    end
    fprintf('  verification: %d frames, max |diff| = %g, checksums equal (%.0f s)\n', ...
            v.n_frames_compared, v.max_abs_diff, v.runtime_s);
else
    info.verification = struct('passed', false, 'mode', 'none', 'note', 'verification skipped');
    info.warnings{end+1} = 'verification skipped (cfg.verify = none)';
end
peak = max(peak, peak_ram_gb());
info.peak_ram_gb_during_convert = peak;

delete_if_exists(files.h5);
movefile(tmp, files.h5);
info.h5_bytes = dir(files.h5).bytes;
info.h5_bytes_per_frame = info.h5_bytes / N;
info.frame_counts = frame_counts(R);
drop = {'valid', 'tsc_us', 'checksum_sum', 'checksum_wsum', 'h5_row', 'reason'};
info.recordings = cellfun(@(x) json_lists(rmfield(x, intersect(fieldnames(x), drop))), R, ...
                          'UniformOutput', false);
end


% =====================================================================================
function v = verify_against_isxd(h5, R, ft, block)
% Re-read every valid frame twice over independent paths: h5read of the closed file and
% get_frame_data of a freshly opened .isxd. Pixel-exact equality and the checksums of the
% h5 frames are both required.
hi = h5info(h5, '/mov');
sz = hi.Dataspace.Size;
H = sz(1);
W = sz(2);
idx64 = uint64((1:H * W)');
v = struct('mode', 'full', 'n_frames_compared', 0, 'max_abs_diff', 0, 'n_frames_differ', 0, ...
           'n_checksum_mismatch', 0, 'n_zero_frames', 0, 'dtype', hi.Datatype.Type, ...
           'h5_size', sz, 'passed', false);
for r = 1:numel(R)
    rec = R{r};
    mv = isx.Movie.read(rec.file);
    sel = find(strcmp(ft.recording_id, rec.recording_id) & ft.valid == 1);
    rows = ft.h5_row(sel);
    kk = ft.isx_frame(sel);
    for b = 1:block:numel(sel)
        jj = b:min(numel(sel), b + block - 1);
        r0 = rows(jj(1));
        if ~isequal(rows(jj)', r0:r0 + numel(jj) - 1)
            error('isxd_to_h5:verify', 'h5_row is not consecutive in the frame table');
        end
        M = h5read(h5, '/mov', [1 1 r0], [H W numel(jj)]);
        if ~isa(M, 'uint16')
            error('isxd_to_h5:verify', 'h5 /mov reads back as %s, not uint16', class(M));
        end
        for j = 1:numel(jj)
            a = M(:, :, j);
            f = mv.get_frame_data(kk(jj(j)));
            if ~isequal(a, f)
                d = max(abs(double(a(:)) - double(f(:))));
                v.max_abs_diff = max(v.max_abs_diff, d);
                v.n_frames_differ = v.n_frames_differ + 1;
            end
            av = uint64(a(:));
            if sum(av, 'native') ~= ft.checksum_sum(sel(jj(j))) || ...
                    sum(av .* idx64, 'native') ~= ft.checksum_wsum(sel(jj(j)))
                v.n_checksum_mismatch = v.n_checksum_mismatch + 1;
            end
            if ~any(a(:))
                v.n_zero_frames = v.n_zero_frames + 1;
            end
        end
        v.n_frames_compared = v.n_frames_compared + numel(jj);
    end
    clear mv;
end
n_valid = nnz(ft.valid == 1);
v.all_frames = (v.n_frames_compared == n_valid) && (sz(3) == n_valid);
v.passed = v.all_frames && v.max_abs_diff == 0 && v.n_frames_differ == 0 && ...
           v.n_checksum_mismatch == 0 && v.n_zero_frames == 0 && strcmp(v.dtype, 'H5T_STD_U16LE');
end


% =====================================================================================
function cfg = apply_defaults(cfg)
if ~isstruct(cfg)
    error('isxd_to_h5:cfg', 'cfg must be a struct (or ''selftest'')');
end
for f = {'isxd', 'out_dir'}
    if ~isfield(cfg, f{1}) || isempty(cfg.(f{1}))
        error('isxd_to_h5:cfg', 'cfg.%s is required', f{1});
    end
end
cfg.isxd = cellstr(cfg.isxd);
n = numel(cfg.isxd);
for r = 1:n
    cfg.isxd{r} = char(cfg.isxd{r});
    if ~isfile(cfg.isxd{r})
        error('isxd_to_h5:cfg', 'raw movie not found: %s', cfg.isxd{r});
    end
end
cfg.out_dir = char(cfg.out_dir);
if ~isfield(cfg, 'recording_id') || isempty(cfg.recording_id)
    cfg.recording_id = cell(1, n);
    for r = 1:n
        [~, cfg.recording_id{r}] = fileparts(cfg.isxd{r});
    end
end
cfg.recording_id = cellstr(cfg.recording_id);
if ~isfield(cfg, 'assay') || isempty(cfg.assay)
    cfg.assay = repmat({''}, 1, n);
end
cfg.assay = cellstr(cfg.assay);
if numel(cfg.recording_id) ~= n || numel(cfg.assay) ~= n
    error('isxd_to_h5:cfg', 'recording_id and assay need one entry per isxd');
end
if numel(unique(cfg.recording_id)) < n
    error('isxd_to_h5:cfg', 'recording_id values must be unique');
end
if ~isfield(cfg, 'name') || isempty(cfg.name)
    if n > 1
        error('isxd_to_h5:cfg', 'cfg.name is required when several movies go into one h5');
    end
    cfg.name = cfg.recording_id{1};
end
cfg.name = char(cfg.name);
def = struct('chunk', [], 'block_frames', 50, 'deflate', 0, 'verify', 'full', 'overwrite', false, 'max_frames', inf, ...
             'idps_dir', fullfile(getenv('ProgramFiles'), 'Inscopix', 'Data Processing'));
f = fieldnames(def);
for k = 1:numel(f)
    if ~isfield(cfg, f{k}) || (isempty(cfg.(f{k})) && ~strcmp(f{k}, 'chunk'))
        cfg.(f{k}) = def.(f{k});
    end
end
if ~ismember(lower(cfg.verify), {'full', 'none'})
    error('isxd_to_h5:cfg', 'verify must be full or none');
end
if ~isempty(cfg.chunk) && numel(cfg.chunk) ~= 3
    error('isxd_to_h5:cfg', 'chunk must be [rows cols frames]');
end
end


function ensure_idps(idps_dir)
if ~isfolder(fullfile(idps_dir, '+isx'))
    error('isxd_to_h5:idps', 'IDPS MATLAB API not found: %s\\+isx', idps_dir);
end
addpath(idps_dir);
end


function files = output_files(out_dir, name)
files = struct('h5', fullfile(out_dir, [name '_raw.h5']), ...
               'frames_csv', fullfile(out_dir, [name '_raw_frames.csv']), ...
               'json', fullfile(out_dir, [name '_raw.json']));
end


function prev = reusable(files, cfg)
% An earlier output is re-used only if it finished, passed verification, and was made
% from the same files (path, size) into the same layout.
prev = [];
if cfg.overwrite || ~isfile(files.json) || ~isfile(files.h5) || ~isfile(files.frames_csv)
    return;
end
try
    js = jsondecode(fileread(files.json));
    recs = js.recordings;
    if isstruct(recs)
        recs = num2cell(recs);
    end
    ok = strcmp(js.status, 'ok') && js.verification.passed && numel(recs) == numel(cfg.isxd);
    for r = 1:numel(recs)
        if ~ok
            break;
        end
        d = dir(cfg.isxd{r});
        ok = strcmpi(recs{r}.file, cfg.isxd{r}) && recs{r}.file_bytes == d.bytes && ...
             strcmp(recs{r}.recording_id, cfg.recording_id{r});
    end
    ok = ok && dir(files.h5).bytes == js.h5_bytes;
    if ok
        prev = js;
    end
catch
    prev = [];
end
end


function m = read_meta(file)
mv = isx.Movie.read(file);
tm = mv.timing;
m.file = file;
m.file_bytes = dir(file).bytes;
m.n = double(tm.num_samples);
m.valid = logical(tm.get_valid_samples_mask());
m.valid = m.valid(:)';
m.dropped = double(tm.dropped(:)');
m.cropped = double(tm.cropped);
m.blank = double(tm.blank(:)');
m.n_dropped = numel(m.dropped);
m.period_s_timing = tm.period.secs_float;
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
% Reason per frame: dropped first (no data at all), then cropped, then blank.
reason = repmat({''}, m.n, 1);
for c = 1:size(m.cropped, 1)
    rr = (m.cropped(c, 1):m.cropped(c, 2)) + 1;
    reason(rr(rr >= 1 & rr <= m.n)) = {'cropped'};
end
b = m.blank + 1;
reason(b(b >= 1 & b <= m.n)) = {'blank'};
d = m.dropped + 1;
reason(d(d >= 1 & d <= m.n)) = {'dropped'};
still = ~m.valid(:) & strcmp(reason, '');
reason(still) = {'invalid'};
reason(m.valid) = {''};
m.reason = reason;
clear mv;
end


function [period_s, fps] = period_from_tsc(tsc)
idx = find(isfinite(tsc));
period_s = NaN;
fps = NaN;
if numel(idx) >= 2
    period_s = (tsc(idx(end)) - tsc(idx(1))) / (idx(end) - idx(1)) / 1e6;
    fps = 1 / period_s;
end
end


function ft = build_frame_table(R)
c = cell(numel(R), 9);
for r = 1:numel(R)
    rec = R{r};
    n = rec.n;
    c(r, :) = {repmat({rec.recording_id}, n, 1), repmat({rec.assay}, n, 1), (0:n - 1)', ...
               double(rec.valid(:)), rec.reason, rec.h5_row, rec.tsc_us, rec.checksum_sum, rec.checksum_wsum};
end
names = {'recording_id', 'assay', 'isx_frame', 'valid', 'invalid_reason', 'h5_row', 'tsc_us', ...
         'checksum_sum', 'checksum_wsum'};
ft = struct();
for k = 1:numel(names)
    ft.(names{k}) = vertcat(c{:, k});
end
end


function counts = frame_counts(R)
counts = cellfun(@(x) struct('recording_id', x.recording_id, 'n_frames_isx', x.n, ...
    'n_valid', nnz(x.valid), 'n_dropped', nnz(strcmp(x.reason, 'dropped')), ...
    'n_cropped', nnz(strcmp(x.reason, 'cropped')), 'n_blank', nnz(strcmp(x.reason, 'blank')), ...
    'n_other_invalid', nnz(strcmp(x.reason, 'invalid'))), R);
end


function write_frame_table(file, ft)
% NaN -> empty; uint64 checksums printed exactly (beyond 2^53) and empty for invalid rows.
cols = {'recording_id', 'assay', 'isx_frame', 'valid', 'invalid_reason', 'h5_row', 'tsc_us', ...
        'checksum_sum', 'checksum_wsum'};
n = numel(ft.isx_frame);
txt = strings(n, numel(cols));
inv = ft.valid ~= 1;
for c = 1:numel(cols)
    v = ft.(cols{c});
    if iscell(v)
        txt(:, c) = string(v);
    elseif isa(v, 'uint64')
        s = compose("%d", v(:));
        s(inv) = "";
        txt(:, c) = s;
    else
        s = compose("%d", double(v(:)));
        s(isnan(v(:))) = "";
        txt(:, c) = s;
    end
end
fid = fopen(file, 'w');
if fid < 0
    error('isxd_to_h5:io', 'cannot write %s', file);
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', strjoin(cols, ','));
fprintf(fid, '%s\n', join(txt, ',', 2));
end


function ft = read_frame_table(file)
% Reader for the selftest round trip (checksums kept as uint64 via text).
opts = detectImportOptions(file, 'Delimiter', ',', 'TextType', 'char');
opts = setvartype(opts, {'recording_id', 'assay', 'invalid_reason', 'checksum_sum', 'checksum_wsum'}, 'char');
opts = setvartype(opts, {'isx_frame', 'valid', 'h5_row', 'tsc_us'}, 'double');
tb = readtable(file, opts);
ft = table2struct(tb, 'ToScalar', true);
for f = {'checksum_sum', 'checksum_wsum'}
    s = ft.(f{1});
    u = zeros(numel(s), 1, 'uint64');
    for k = 1:numel(s)
        if ~isempty(s{k})
            u(k) = sscanf(s{k}, '%lu');
        end
    end
    ft.(f{1}) = u;
end
end


function g = peak_ram_gb()
% Peak working set of this MATLAB process (Windows, via .NET); NaN where unavailable.
g = NaN;
try
    p = System.Diagnostics.Process.GetCurrentProcess();
    g = double(p.PeakWorkingSet64) / 2^30;
catch
end
end


function write_json(file, s)
fid = fopen(file, 'w');
if fid < 0
    error('isxd_to_h5:io', 'cannot write %s', file);
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(s, 'PrettyPrint', true));
end


function rec = json_lists(rec)
for f = {'dropped', 'blank'}
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
% Synthetic raw movies written with the IDPS API: full uint16 range (0 and 65535 pixels),
% a dropped frame, a cropped range, two movies into one h5. Checks: every h5 frame equals
% the frame that was written, invalid frames left out and named, checksums exact, the
% uint64 CSV round trip, re-use, and that the verification pass catches a changed pixel,
% a swapped pair of frames, and a wrong checksum.
fprintf('isxd_to_h5 selftest\n');
here = fileparts(mfilename('fullpath'));
root = fullfile(here, 'out', 'selftest_isxd_to_h5');
if isfolder(root)
    rmdir(root, 's');
end
mkdir(root);

% ---- 1. pure parts (no IDPS) ------------------------------------------------------
v = uint64([65535 65535 1]');
assert_true(sum(v .* uint64((1:3)'), 'native') == uint64(65535 * 3 + 3), 'weighted checksum arithmetic');
big = uint64(2)^60 + uint64(3);
ft0 = struct('recording_id', {{'a'; 'a'}}, 'assay', {{''; ''}}, 'isx_frame', [0; 1], 'valid', [1; 0], ...
             'invalid_reason', {{''; 'dropped'}}, 'h5_row', [1; NaN], 'tsc_us', [10; NaN], ...
             'checksum_sum', uint64([big; 0]), 'checksum_wsum', uint64([5; 0]));
write_frame_table(fullfile(root, 'rt.csv'), ft0);
rt = read_frame_table(fullfile(root, 'rt.csv'));
assert_true(rt.checksum_sum(1) == big && isnan(rt.h5_row(2)) && strcmp(rt.invalid_reason{2}, 'dropped'), ...
            'frame table CSV round trip (uint64 beyond 2^53, empty fields)');
fprintf('  checksum arithmetic and CSV round trip: ok\n');

idps = fullfile(getenv('ProgramFiles'), 'Inscopix', 'Data Processing');
if ~isfolder(fullfile(idps, '+isx'))
    fprintf('  IDPS not installed: conversion part SKIPPED\n');
    info = struct('selftest', 'ok (IDPS part skipped)');
    return;
end
addpath(idps);

% ---- 2. synthetic raw movies ------------------------------------------------------
rng(3);
H = 37;
W = 52;                                   % not multiples of the chunk tile
spec = struct('name', {'syn_a', 'syn_b'}, 'n', {23, 17}, 'dropped', {[4 11], []}, ...
              'cropped', {zeros(0, 2), [5 6]});
truth = cell(1, 2);
files = cell(1, 2);
for s = 1:2
    files{s} = fullfile(root, [spec(s).name '.isxd']);
    truth{s} = write_synthetic(files{s}, spec(s), H, W);
end
cfg = struct('isxd', {files}, 'recording_id', {{'T01_d1_pre', 'T01_d1_post'}}, ...
             'assay', {{'pre', 'post'}}, 'name', 'T01_d1_both', 'out_dir', fullfile(root, 'conv'), ...
             'block_frames', 6);
out = isxd_to_h5(cfg);
assert_true(strcmp(out.status, 'ok') && out.verification.passed && out.verification.max_abs_diff == 0, ...
            'verification passed');
hi = h5info(out.files.h5, '/mov');
assert_true(strcmp(hi.Datatype.Type, 'H5T_STD_U16LE'), 'h5 dtype uint16');
n_valid = sum(cellfun(@(t) size(t.frames, 3), truth));
assert_true(isequal(hi.Dataspace.Size, [H W n_valid]), 'h5 size [H W n_valid]');
mov = h5read(out.files.h5, '/mov');
ft = read_frame_table(out.files.frames_csv);
assert_true(numel(ft.isx_frame) == spec(1).n + spec(2).n, 'one table row per raw frame');
row = 0;
for s = 1:2
    sel = strcmp(ft.recording_id, cfg.recording_id{s});
    assert_true(isequal(ft.isx_frame(sel)', 0:spec(s).n - 1), 'isx_frame 0..n-1');
    kv = truth{s}.valid_idx;
    nv = numel(kv);
    assert_true(isequal(mov(:, :, row + (1:nv)), truth{s}.frames), ...
                sprintf('%s: every h5 frame equals the written frame, in raw order', spec(s).name));
    rows = ft.h5_row(sel);
    assert_true(isequal(rows(kv + 1)', row + (1:nv)) && all(isnan(rows(setdiff(1:spec(s).n, kv + 1)))), ...
                'h5_row of valid frames, empty for invalid');
    reason = ft.invalid_reason(sel);
    assert_true(all(strcmp(reason(spec(s).dropped + 1), 'dropped')), 'dropped frames named');
    for c = 1:size(spec(s).cropped, 1)
        assert_true(all(strcmp(reason((spec(s).cropped(c, 1):spec(s).cropped(c, 2)) + 1), 'cropped')), ...
                    'cropped frames named');
    end
    cs = ft.checksum_sum(sel);
    for j = 1:nv
        fr = truth{s}.frames(:, :, j);
        assert_true(cs(kv(j) + 1) == uint64(sum(double(fr(:)))), 'checksum_sum exact');
    end
    row = row + nv;
end
assert_true(any(mov(:) == 0) && any(mov(:) == 65535), 'full uint16 range preserved (0 and 65535)');
fprintf('  two movies, dropped + cropped frames, %d valid frames: lossless ok\n', n_valid);

% ---- 3. re-use, and the verification pass must catch damage -------------------------
out2 = isxd_to_h5(cfg);
assert_true(out2.reused, 're-use of a verified output');
R = cell(1, 2);
for s = 1:2
    R{s} = read_meta(files{s});
    R{s}.recording_id = cfg.recording_id{s};
end
bad = fullfile(root, 'damaged.h5');
copyfile(out.files.h5, bad);
fr = h5read(bad, '/mov', [1 1 3], [H W 1]);
fr(5, 7) = fr(5, 7) + 1;
h5write(bad, '/mov', fr, [1 1 3], [H W 1]);
v = verify_against_isxd(bad, R, ft, 6);
assert_true(~v.passed && v.max_abs_diff == 1 && v.n_frames_differ == 1 && v.n_checksum_mismatch == 1, ...
            'verification catches one changed pixel');
copyfile(out.files.h5, bad, 'f');
a = h5read(bad, '/mov', [1 1 8], [H W 1]);
b = h5read(bad, '/mov', [1 1 9], [H W 1]);
h5write(bad, '/mov', b, [1 1 8], [H W 1]);
h5write(bad, '/mov', a, [1 1 9], [H W 1]);
v = verify_against_isxd(bad, R, ft, 6);
assert_true(~v.passed && v.n_frames_differ == 2, 'verification catches two swapped frames');
ft_bad = ft;
k = find(ft.valid == 1, 1, 'last');
ft_bad.checksum_wsum(k) = ft_bad.checksum_wsum(k) + 1;
v = verify_against_isxd(out.files.h5, R, ft_bad, 6);
assert_true(~v.passed && v.n_checksum_mismatch == 1 && v.max_abs_diff == 0, 'verification catches a wrong checksum');
fprintf('  re-use and damage detection: ok\n');
info = struct('selftest', 'ok');
fprintf('isxd_to_h5 selftest PASSED\n');
end


function t = write_synthetic(file, spec, H, W)
start = isx.Time.from_secs_since_epoch(isx.Duration.from_secs(1.7e9));
targs = {'num_samples', spec.n, 'period', isx.Duration.from_msecs(50), 'start', start};
if ~isempty(spec.dropped)
    targs = [targs, {'dropped', spec.dropped}];
end
if ~isempty(spec.cropped)
    targs = [targs, {'cropped', spec.cropped}];
end
tm = isx.Timing(targs{:});
mv = isx.Movie.write(file, tm, isx.Spacing('num_pixels', [H, W]), 'uint16');
invalid = spec.dropped;
for c = 1:size(spec.cropped, 1)
    invalid = [invalid, spec.cropped(c, 1):spec.cropped(c, 2)]; %#ok<AGROW>
end
t.valid_idx = setdiff(0:spec.n - 1, invalid);
t.frames = zeros(H, W, numel(t.valid_idx), 'uint16');
for j = 1:numel(t.valid_idx)
    fr = uint16(randi([0 65535], H, W));
    fr(1, 1) = 0;
    fr(end, end) = 65535;
    mv.set_frame_data(t.valid_idx(j), fr);
    t.frames(:, :, j) = fr;
end
mv.flush();
clear mv;
end


function assert_true(cond, what)
if ~cond
    error('isxd_to_h5:selftest', 'selftest FAILED: %s', what);
end
end
