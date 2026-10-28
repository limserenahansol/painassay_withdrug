function sessions = mini1p_pair_sessions(videoRoot, mice, dayNo, bottomRoot, maxFrameDiff)
%MINI1P_PAIR_SESSIONS  Pair the bottom and side videos of each mini1p session.
%
%   sessions = mini1p_pair_sessions(videoRoot, mice, dayNo)
%
%   videoRoot : folder with the SIDE videos; its 'bottom' subfolder holds the BOTTOM ones
%               (K:\mini1p_pain_behavior layout, IC Capture names such as
%               '5941 female0004 26-10-19 11-58-28-014.avi')
%   mice      : struct array, fields id (e.g. '5941') and sex ('F' / 'M')
%   dayNo     : day number for the scorer dialog (1 or 2)
%   bottomRoot: folder of the BOTTOM videos (default <videoRoot>\bottom); e.g. the scoring
%               copies of make_scoring_videos.py (same names, .mp4, same frames, 30 fps)
%
%   A bottom and a side file are one session when they carry the same mouse id, start
%   within 60 s of each other (the two IC Capture windows are started by hand) and have the
%   SAME number of frames (both cameras are triggered by the Inscopix SYNC pulses). A pair
%   that fails the frame-count test is refused, not guessed - unless maxFrameDiff (default 0) allows
%   a few frames dropped by one camera (Day 2, 5941 post: side 29763, bottom 29781 = logger pulses);
%   such a pair is kept with a warning, and nFrames / nSide hold both counts.
%   Per mouse, the earlier session is 'Baseline' and the later one 'Post-treatment'. The
%   treatment is never read from the file name and never shown.
%
%   Returns the struct array score_AB_dual_view_mini1p expects (label, botFile, sidFile,
%   preset, nFrames, tStart).

if nargin < 3 || isempty(dayNo), dayNo = 1; end
if nargin < 4 || isempty(bottomRoot), bottomRoot = fullfile(videoRoot, 'bottom'); end
if nargin < 5 || isempty(maxFrameDiff), maxFrameDiff = 0; end
side = scan(videoRoot);
bot  = scan(bottomRoot);
if isempty(side) || isempty(bot)
    error('mini1p_pair_sessions:noVideos', 'No videos in %s or its bottom folder.', videoRoot);
end

sessions = struct('label', {}, 'botFile', {}, 'sidFile', {}, 'preset', {}, ...
                  'nFrames', {}, 'nSide', {}, 'tStart', {}, 'mouse', {});
used = false(numel(bot), 1);
for i = 1:numel(side)
    s = side(i);
    cand = find(strcmp({bot.mouse}, s.mouse) & ~used' & ...
                abs([bot.t] - s.t) * 86400 <= 60);
    if isempty(cand)
        error('mini1p_pair_sessions:noPartner', 'No bottom video for %s.', s.name);
    end
    [~, k] = min(abs([bot(cand).t] - s.t));
    b = bot(cand(k));
    if abs(b.n - s.n) > maxFrameDiff
        error('mini1p_pair_sessions:frameCount', ...
              ['%s has %d frames but its bottom partner %s has %d. Both cameras are ' ...
               'triggered by the same SYNC pulses, so this pair is not trusted.'], ...
              s.name, s.n, b.name, b.n);
    end
    if b.n ~= s.n
        warning('mini1p_pair_sessions:frameDiff', ['%s has %d frames, its bottom partner %s has %d: ' ...
                'paired anyway (limit %d frames).'], s.name, s.n, b.name, b.n, maxFrameDiff);
    end
    used(cand(k)) = true;
    sessions(end + 1) = struct('label', '', 'botFile', b.path, 'sidFile', s.path, ...
        'preset', struct(), 'nFrames', b.n, 'nSide', s.n, 'tStart', min(b.t, s.t), 'mouse', s.mouse); %#ok<AGROW>
end
if any(~used)
    warning('Unpaired bottom video(s): %s', strjoin({bot(~used).name}, ', '));
end

% phase by time order within a mouse; session number by time order over the day
[~, ord] = sort([sessions.tStart]);
sessions = sessions(ord);
for i = 1:numel(sessions)
    m = sessions(i).mouse;
    same = find(strcmp({sessions.mouse}, m));
    if numel(same) > 2
        error('mini1p_pair_sessions:tooMany', '%d sessions for mouse %s.', numel(same), m);
    end
    if same(1) == i, phase = 'Baseline'; else, phase = 'Post-treatment'; end
    sex = '';
    j = find(strcmp({mice.id}, m), 1);
    if ~isempty(j), sex = mice(j).sex; end
    sessions(i).preset = struct('sessionNo', sprintf('%d', i), 'mouseID', m, ...
        'sexID', sex, 'dayNo', sprintf('%d', dayNo), 'phase', phase);
    % per-session stimulus order and session number from the randomisation sheet
    if isfield(mice, 'order') && ~isempty(j)
        o = mice(j).order;
        k = find(strcmp({o.phase}, phase), 1);
        if ~isempty(k)
            sessions(i).preset.stimNames = o(k).stim;
            sessions(i).preset.sessionNo = sprintf('%d', o(k).session);
        end
    end
    keys = '(order not set)';
    if isfield(sessions(i).preset, 'stimNames')
        keys = strjoin(sessions(i).preset.stimNames, ' > ');
    end
    sessions(i).label = sprintf('session %s   mouse %s  %s   started %s   %s   %.1f min   keys 1-4: %s', ...
        sessions(i).preset.sessionNo, m, sex, datestr(sessions(i).tStart, 'HH:MM'), phase, ...
        sessions(i).nFrames / 30 / 60, keys);
end
end


function v = scan(folder)
L = [dir(fullfile(folder, '*.avi')); dir(fullfile(folder, '*.mp4'))];
v = struct('name', {}, 'path', {}, 'mouse', {}, 't', {}, 'n', {});
for i = 1:numel(L)
    nm = L(i).name;
    tok = regexp(nm, '^\s*(\d+)', 'tokens', 'once');
    ts  = regexp(nm, '(\d{2}-\d{2}-\d{2}) (\d{2}-\d{2}-\d{2})', 'tokens', 'once');
    if isempty(tok) || isempty(ts)
        warning('Skipped (no mouse id / time stamp in the name): %s', nm);
        continue;
    end
    t = datenum([ts{1} ' ' ts{2}], 'yy-mm-dd HH-MM-SS');
    r = VideoReader(fullfile(L(i).folder, nm));
    n = round(r.Duration * r.FrameRate);
    v(end + 1) = struct('name', nm, 'path', fullfile(L(i).folder, nm), ...
                        'mouse', tok{1}, 't', t, 'n', n); %#ok<AGROW>
end
end
