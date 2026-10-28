function out = merge_mini1p_scoring(extract_mat, recording_id, scoring_mat, out_dir, opts)
%MERGE_MINI1P_SCORING  mini1p EXTRACT day file + one ScoringAB file -> mini2p's
%   final_neuron_behavior.mat, so the mini2p downstream (classify_pain_neurons and the
%   rest) runs on mini1p data unchanged.
%
%   out = merge_mini1p_scoring(extract_mat, recording_id, scoring_mat, out_dir, opts)
%   merge_mini1p_scoring('selftest')
%
%   extract_mat  : <mouse>_d<day>_extract.mat of extract_1p (both sessions of the day,
%                  one set of footprints, so cell k is the same cell before and after
%                  the injection)
%   recording_id : which session of that file (frame_table.recording_id), e.g. '5941_d1_pre'
%   scoring_mat  : ScoringAB_<bottom video>.mat of score_AB_dual_view(_mini1p) for it
%   out_dir      : analysis folder of this session (gets final_neuron_behavior.mat)
%
%   opts (default)
%     trace          ('raw')  'raw' = temporal_raw (least squares, keeps decreases);
%                             'extract' = temporal_extract
%     n_cam_frames   ([])     camera frame count; [] = from the scored video file
%     allow_count_mismatch (false)
%
%   HOW THE TWO ARE JOINED (CONTRACTS.md 1a, K = 0). The cameras are triggered by the
%   Inscopix SYNC pulse that ends each miniscope frame, so camera frame n (1-based, the
%   frame index every ScoringAB mark carries) was exposed during isx_frame n (0-based).
%   The join is by FRAME NUMBER, never by seconds, so the AVI header rate (60) cannot
%   enter it. isx_frame 0 has no camera frame. The join is refused unless the camera has
%   exactly as many frames as the recording (one pulse per frame).
%
%   UNITS. extract_1p traces are dF/F around the day mean (about 0). mini2p's downstream
%   divides each peri-event window by its own pre-event mean, which only works for a
%   positive F-like trace (mini2p EXTRACT ran without dF/F). So deltaF_over_F here is
%   F/F0 = 1 + dF/F: same shape, same z-scores, positive, and the downstream's local
%   (F - F_pre) / F_pre is then a proper local dF/F. metadata.trace_units says so.
%   Dropped / trimmed frames stay NaN.

if nargin == 1 && strcmpi(extract_mat, 'selftest')
    out = run_selftest();
    return;
end
if nargin < 5, opts = struct(); end
trace_kind = getopt(opts, 'trace', 'raw');
n_cam = getopt(opts, 'n_cam_frames', []);
allow_mismatch = getopt(opts, 'allow_count_mismatch', false);
if ~isfolder(out_dir), mkdir(out_dir); end

% ---- EXTRACT day file: rows of this recording -----------------------------------
ex = load(extract_mat);
ft = ex.frame_table;
rid = cellstr(string(ft.recording_id));
sel = find(strcmp(rid, recording_id));
if isempty(sel)
    error('merge_mini1p:recording', '%s is not in %s (has: %s)', recording_id, ...
          extract_mat, strjoin(unique(rid, 'stable'), ', '));
end
isx = double(ft.isx_frame(sel));
[isx, o] = sort(isx);
sel = sel(o);
if any(diff(isx) ~= 1) || isx(1) ~= 0
    error('merge_mini1p:frames', ['%s: isx_frame must run 0..N-1 without gaps (dropped ' ...
          'frames are rows with valid = 0, not missing rows).'], recording_id);
end
n_isx = numel(isx);
switch trace_kind
    case 'raw',     dff = double(ex.temporal_raw(sel, :));
    case 'extract', dff = double(ex.temporal_extract(sel, :));
    otherwise, error('merge_mini1p:opts', 'opts.trace must be raw or extract');
end
n_cells = size(dff, 2);
fps = double(ex.fps);

% time axis from the per-frame TSC timestamps of this recording, else isx_frame / fps
t = isx / fps;
t_source = 'isx_frame / fps';
if isfield(ft, 'tsc_us')
    tsc = double(ft.tsc_us(sel));
    ok = isfinite(tsc);
    if nnz(ok) >= 2
        p = polyfit(isx(ok), tsc(ok) * 1e-6, 1);
        t = polyval(p, isx);
        t(ok) = tsc(ok) * 1e-6;
        t = t - t(1);
        t_source = 'TSC timestamps (fit at frames without one)';
    end
end

% ---- scoring file ----------------------------------------------------------------
sc = load(scoring_mat);
for f = {'score', 'reflexEvents', 'dFrames', 'dTypes', 'stimNames', 'nUsed'}
    if ~isfield(sc, f{1})
        error('merge_mini1p:scoring', '%s has no %s - not a ScoringAB file.', scoring_mat, f{1});
    end
end
if isempty(n_cam)
    if isfield(sc, 'botFile') && isfile(sc.botFile)
        vr = VideoReader(sc.botFile);
        n_cam = round(vr.Duration * vr.FrameRate);
    else
        error('merge_mini1p:count', ['Cannot open the scored video to count its frames; ' ...
              'pass opts.n_cam_frames.']);
    end
end
if n_cam ~= n_isx
    msg = sprintf(['%s: camera has %d frames, the Inscopix recording %d. With one SYNC ' ...
        'pulse per frame they must be equal (K = 0).'], recording_id, n_cam, n_isx);
    if allow_mismatch
        warning('merge_mini1p:count', '%s Continuing because allow_count_mismatch.', msg);
    else
        error('merge_mini1p:count', '%s', msg);
    end
end
behav_fps = fps;
if isfield(sc, 'fpsTrue') && ~isempty(sc.fpsTrue), behav_fps = double(sc.fpsTrue); end
if isfield(sc, 'hdrFps') && abs(double(sc.frameRate) - double(sc.hdrFps)) < 1e-6 && ...
        abs(double(sc.hdrFps) - fps) > 1
    warning('merge_mini1p:fps', ['%s was scored with the AVI header rate %.1f fps, not ' ...
        'the triggered %.1f Hz: its seconds (dSec, windows, guarding > 2 s) are off; the ' ...
        'frame join below is still exact.'], scoring_mat, sc.hdrFps, fps);
end

% camera frame n -> row of isx_frame n (1-based row = n + 1)
cam2row = @(n) double(n(:)) + 1;
    function v = point_vector(frames)
        v = false(n_isx, 1);
        pr = cam2row(frames);
        pr = pr(pr >= 1 & pr <= n_isx);
        v(pr) = true;
    end
    function v = hold_vector(code)
        v = false(n_isx, 1);
        hs = double(sc.score(:));
        hn = min(numel(hs), double(sc.nUsed));
        hr = cam2row(find(hs(1:hn) == code));
        hr = hr(hr <= n_isx);
        v(hr) = true;
    end

stimNames = cellstr(string(sc.stimNames(:)'));
dF = double(sc.dFrames(:));
dT = double(sc.dTypes(:));
stim_field = struct('light', 'light_touch_vector', 'mild', 'mild_touch_vector', ...
                    'heat', 'thermo_pain_vector', 'pin', 'mechanic_pain_vector');
words = struct('light', {{'light'}}, 'mild', {{'mild'}}, 'heat', {{'heat', 'hot'}}, ...
               'pin', {{'pin'}});
keys = fieldnames(stim_field);
vec = struct();
n_stim = struct();
for i = 1:numel(keys)
    vec.(stim_field.(keys{i})) = false(n_isx, 1);
    n_stim.(keys{i}) = 0;
end
for s = 1:4
    nm = lower(stimNames{s});
    hit = keys(cellfun(@(k) any(cellfun(@(w) contains(nm, w), words.(k))), keys));
    if numel(hit) ~= 1
        error('merge_mini1p:stim', 'Stimulus key %d name "%s" matches %d of light/mild/heat/pin.', ...
              s, stimNames{s}, numel(hit));
    end
    fr = dF(dT == s);
    vec.(stim_field.(hit{1})) = vec.(stim_field.(hit{1})) | point_vector(fr);
    n_stim.(hit{1}) = n_stim.(hit{1}) + numel(fr);
end
if any(dT == 0)
    warning('merge_mini1p:stim', '%d delivery tap(s) of UNKNOWN type are left out.', nnz(dT == 0));
end

te = double(sc.reflexEvents);
if isempty(te), te = zeros(0, 2); end
withdrawal_vector = point_vector(te(te(:, 2) == 1, 1));
flinch_vector     = point_vector(te(te(:, 2) == 2, 1));
attending_vector  = hold_vector(1);
lickbite_vector   = hold_vector(2);
guarding_vector   = hold_vector(3);
escape_vector     = hold_vector(4);
uncertain_vector  = false(n_isx, 1);
if isfield(sc, 'uncert')
    ur = cam2row(find(logical(sc.uncert(:))));
    ur = ur(ur <= n_isx);
    uncertain_vector(ur) = true;
end

mechanic_pain_vector = vec.mechanic_pain_vector;
thermo_pain_vector   = vec.thermo_pain_vector;
light_touch_vector   = vec.light_touch_vector;
mild_touch_vector    = vec.mild_touch_vector;
pin_vector  = mechanic_pain_vector;
heat_vector = thermo_pain_vector;
event_vector = pin_vector | heat_vector | light_touch_vector | mild_touch_vector;

% ---- traces in the mini2p field names ----------------------------------------------
deltaF_over_F = 1 + dff;                        % F/F0, see UNITS above
traces_raw = deltaF_over_F;
mu = mean(traces_raw, 1, 'omitnan');
sd = std(traces_raw, 0, 1, 'omitnan');
sd(sd < 1e-12) = 1;
traces_z = (traces_raw - mu) ./ sd;
neuron_seconds = t(:);
neuron_timestamps = seconds(neuron_seconds);
orig_id = (1:n_cells)';
if isfield(ex, 'kept_ids') && numel(ex.kept_ids) == n_cells
    orig_id = double(ex.kept_ids(:));    % the numbers on the cell-review map
end
spatial_weights = ex.spatial_weights;
valid = true(n_isx, 1);
if isfield(ft, 'valid'), valid = double(ft.valid(sel)) == 1; end

metadata = struct();
metadata.neuron_frame_rate_hz = fps;
metadata.behav_frame_rate_hz = behav_fps;
metadata.n_cells = n_cells;
metadata.recording_id = recording_id;
metadata.extract_file = extract_mat;
metadata.scoring_file = scoring_mat;
metadata.trace = trace_kind;
metadata.trace_units = 'F/F0 = 1 + dF/F (extract_1p dF/F, F0 = per-pixel mean of the day movie)';
metadata.join = 'camera frame n (1-based) = isx_frame n (0-based) = row n+1 (K = 0)';
metadata.time_source = t_source;
metadata.n_isx_frames = n_isx;
metadata.n_cam_frames = n_cam;
metadata.n_valid_frames = nnz(valid);
metadata.n_pin = n_stim.pin;
metadata.n_heat = n_stim.heat;
metadata.n_light = n_stim.light;
metadata.n_mild = n_stim.mild;
metadata.stimNames = stimNames;
metadata.orig_planB_id = orig_id;       % mini2p field name; here = component id of the day review

out_path = fullfile(out_dir, 'final_neuron_behavior.mat');
save(out_path, 'deltaF_over_F', 'traces_raw', 'traces_z', ...
    'neuron_timestamps', 'neuron_seconds', 'spatial_weights', 'orig_id', ...
    'pin_vector', 'heat_vector', 'mechanic_pain_vector', 'thermo_pain_vector', ...
    'light_touch_vector', 'mild_touch_vector', ...
    'withdrawal_vector', 'flinch_vector', 'attending_vector', ...
    'lickbite_vector', 'guarding_vector', 'escape_vector', 'uncertain_vector', ...
    'event_vector', 'valid', 'metadata', '-v7.3');

Traw = array2table([(0:n_isx - 1)', neuron_seconds], 'VariableNames', {'isx_frame', 'time_s'});
Tz = Traw;
for c = 1:n_cells
    Traw.(sprintf('cell_%03d', orig_id(c))) = traces_raw(:, c);
    Tz.(sprintf('cell_%03d', orig_id(c))) = traces_z(:, c);
end
writetable(Traw, fullfile(out_dir, 'traces_raw_all_cells.csv'));
writetable(Tz,   fullfile(out_dir, 'traces_zscore_all_cells.csv'));

fprintf(['merge_mini1p_scoring %s: %d cells x %d frames; deliveries light %d, mild %d, ' ...
         'heat %d, pin %d; withdrawal %d, flinch %d\n'], recording_id, n_cells, n_isx, ...
         n_stim.light, n_stim.mild, n_stim.heat, n_stim.pin, nnz(withdrawal_vector), ...
         nnz(flinch_vector));
out = struct('file', out_path, 'n_cells', n_cells, 'n_frames', n_isx, 'metadata', metadata);
end


function v = getopt(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end


% =====================================================================================
function out = run_selftest()
root = fullfile(tempdir, 'merge_mini1p_selftest');
if isfolder(root), rmdir(root, 's'); end
mkdir(root);
fps = 30; n = 600; C = 3;
% day file with two recordings; frame 100 of 'B_pre' is dropped
rec = [repmat({'B_pre'}, n, 1); repmat({'B_post'}, n, 1)];
ft = struct();
ft.recording_id = rec;
ft.isx_frame = [(0:n - 1)'; (0:n - 1)'];
ft.valid = ones(2 * n, 1); ft.valid(101) = 0;
ft.tsc_us = [(0:n - 1)' / fps * 1e6; (0:n - 1)' / fps * 1e6];
ft.tsc_us(101) = NaN;
tr = 0.01 * randn(2 * n, C, 'single');
tr(201:205, 2) = 0.5;                        % post-frame 200..204 of B_pre: a transient
tr(101, :) = NaN;
ex = struct('spatial_weights', rand(8, 8, C, 'single'), 'temporal_raw', tr, ...
    'temporal_extract', max(tr, 0), 'frame_table', ft, 'fps', fps);
exf = fullfile(root, 'B_d1_extract.mat');
save(exf, '-struct', 'ex');
% scoring: pin (key 4) taps at camera frames 200 and 400; heat (key 3) at 300
sc = struct();
sc.score = zeros(n, 1); sc.score(250:259) = 3;
sc.uncert = false(n, 1);
sc.reflexEvents = [201 1; 401 2];
sc.dFrames = [200; 300; 400]; sc.dTypes = [4; 3; 4];
sc.stimNames = {'Light touch', 'Mild touch', 'Heat', 'Pin prick'};
sc.nUsed = n; sc.frameRate = fps; sc.fpsTrue = fps; sc.hdrFps = 60;
scf = fullfile(root, 'ScoringAB_x.mat');
save(scf, '-struct', 'sc');

o = merge_mini1p_scoring(exf, 'B_pre', scf, fullfile(root, 'out'), struct('n_cam_frames', n));
M = load(o.file);
assert_true(isequal(find(M.mechanic_pain_vector), [201; 401]), 'pin: camera frame n -> row n+1');
assert_true(isequal(find(M.thermo_pain_vector), 301), 'heat row');
assert_true(~any(M.light_touch_vector) && ~any(M.mild_touch_vector), 'no light / mild');
assert_true(isequal(find(M.guarding_vector), (251:260)'), 'guarding hold rows');
assert_true(isequal(find(M.withdrawal_vector), 202) && isequal(find(M.flinch_vector), 402), 'reflex rows');
assert_true(all(isnan(M.deltaF_over_F(101, :))) && ~M.valid(101), 'dropped frame stays NaN');
assert_true(abs(mean(M.deltaF_over_F(:, 1), 'omitnan') - 1) < 0.01, 'F/F0 around 1');
assert_true(abs(M.deltaF_over_F(201, 2) - 1.5) < 1e-6, 'transient on the right row');
assert_true(abs(M.neuron_seconds(301) - 10) < 1e-9, 'time from TSC');
% refusals
threw = false;
try
    merge_mini1p_scoring(exf, 'B_pre', scf, fullfile(root, 'out2'), struct('n_cam_frames', n - 1));
catch err
    threw = strcmp(err.identifier, 'merge_mini1p:count');
end
assert_true(threw, 'count mismatch refused');
threw = false;
try
    merge_mini1p_scoring(exf, 'nope', scf, fullfile(root, 'out3'), struct('n_cam_frames', n));
catch err
    threw = strcmp(err.identifier, 'merge_mini1p:recording');
end
assert_true(threw, 'unknown recording refused');
o2 = merge_mini1p_scoring(exf, 'B_post', scf, fullfile(root, 'out4'), struct('n_cam_frames', n));
assert_true(o2.n_frames == n, 'second recording of the day');
fprintf('merge_mini1p_scoring selftest PASSED\n');
out = true;
end


function assert_true(c, msg)
if ~c, error('merge_mini1p:selftest', 'FAILED: %s', msg); end
end
