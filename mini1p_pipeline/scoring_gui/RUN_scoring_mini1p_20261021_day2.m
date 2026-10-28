%% RUN_scoring_mini1p_20261021_day2.m
% Dual-view manual scoring of the Day-2 (2026-10-21) mini1p sessions (4 mice x baseline + post,
% crossover: 5941 / 5939 vehicle, 5937 / 1191 SBI-553 - never shown while scoring).
% Same scorer and keys as Day 1 (RUN_scoring_mini1p_20261019.m).
%
% NEW on Day 2: the deliveries come from the pressure logger (logger_deliveries.py ->
% logger_to_delivery_times.py), so choose MODE 2 (BEHAVIOUR ONLY): every delivery is already on
% the track with its stimulus, and only the behaviour is scored. The launcher hands each session
% its own logger file (no file dialog); the scorer refuses a file whose frame count differs from
% the video. A delivery the logger missed can still be added in mode 3.
%
% Open this file in the MATLAB desktop and press F5 (needs a window + keyboard).
%
% Keys (same as Day 1):
%   HOLD  a attending   s lick/bite   d guarding   f escape/rear   u uncertain
%   HOLD  g general movement (turning / a few steps)   h grooming
%   TAP   1 (or w) withdrawal   2 (or e) flinch        (mode 2: deliveries are pre-loaded)
%   SPACE pause   arrows step   . faster   , slower
%   z undo   BACKSPACE delete nearest   x save and quit (q does nothing)

clear; clc; close all;

VIDEO_ROOT = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_SBI_day2\side';   % side view (original AVI)
% bottom view: the scoring copies (paws made visible, same frames, 30 fps) of make_scoring_videos.py;
% the original bottom AVIs (mini1p_SBI_day2 itself) are used if the copies are not all there
BOTTOM_DIR = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos_scoring\20261021\bottom_enhanced';
if ~isfolder(BOTTOM_DIR) || numel(dir(fullfile(BOTTOM_DIR, '*.mp4'))) < 8
    BOTTOM_DIR = fileparts(VIDEO_ROOT);
end
OUT_DIR    = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261021_day2';
LOGGER_DIR = OUT_DIR;            % DeliveryTimes_<mouse>_d2_<pre|post>_LOGGER.mat
FPS_TRUE   = 30;                 % Inscopix frame rate = camera trigger rate (AVI header says 60)
DAY_NO     = 2;

% sex and stimulus ORDER per session (Day-2 randomisation table, sessions 13-16 and 19-22);
% keys 1-4 follow it: key 1 = the first stimulus block of that session
mice = struct('id', {'5941', '5937', '5939', '1191'}, 'sex', {'F', 'M', 'M', 'F'});
ord = @(ph, ses, s) struct('phase', ph, 'session', ses, 'stim', {s});
mice(1).order = [ord('Baseline', 13, {'Light touch', 'Pin prick', 'Mild touch', 'Heat'}), ...
                 ord('Post-treatment', 19, {'Light touch', 'Heat', 'Mild touch', 'Pin prick'})];
mice(2).order = [ord('Baseline', 14, {'Mild touch', 'Heat', 'Light touch', 'Pin prick'}), ...
                 ord('Post-treatment', 20, {'Mild touch', 'Pin prick', 'Heat', 'Light touch'})];
mice(3).order = [ord('Baseline', 15, {'Light touch', 'Heat', 'Pin prick', 'Mild touch'}), ...
                 ord('Post-treatment', 21, {'Mild touch', 'Light touch', 'Pin prick', 'Heat'})];
mice(4).order = [ord('Baseline', 16, {'Heat', 'Light touch', 'Mild touch', 'Pin prick'}), ...
                 ord('Post-treatment', 22, {'Pin prick', 'Light touch', 'Mild touch', 'Heat'})];

addpath(fileparts(mfilename('fullpath')));
if ~isfolder(OUT_DIR), mkdir(OUT_DIR); end

% 5941 post: the side camera dropped 18 frames (29763 vs bottom 29781 = logger pulses), so up to 30
% frames are tolerated; the BOTTOM video (= logger = imaging frames) is the reference for every time
sessions = mini1p_pair_sessions(VIDEO_ROOT, mice, DAY_NO, BOTTOM_DIR, 30);
[sessions.autoFile] = deal('');
for k = 1:numel(sessions)
    ph = 'pre';
    if strcmp(sessions(k).preset.phase, 'Post-treatment'), ph = 'post'; end
    f = fullfile(LOGGER_DIR, sprintf('DeliveryTimes_%s_d2_%s_LOGGER.mat', sessions(k).mouse, ph));
    if isfile(f)
        sessions(k).preset.deliveryFile = f;
        L = load(f, 'numFrames', 'deliveryFrames');
        sessions(k).label = sprintf('%s   [logger: %d deliveries, %d frames vs video %d]', ...
            sessions(k).label, numel(L.deliveryFrames), round(L.numFrames), sessions(k).nFrames);
        if sessions(k).nSide ~= sessions(k).nFrames
            % camera_drop_locator.py: 5941 post lost all 18 side frames at once at ~170 s (before the
            % first block), so from there side frame k - 18 = bottom frame k: offset -18 frames
            dropped = sessions(k).nFrames - sessions(k).nSide;
            sessions(k).preset.syncOff = sprintf('%.3f', -dropped / FPS_TRUE);
            sessions(k).label = sprintf('%s   [side camera %d frames short: side offset %s s]', sessions(k).label, ...
                dropped, sessions(k).preset.syncOff);
        end
    else
        sessions(k).label = [sessions(k).label '   [NO logger file: use mode 3]'];
    end
    % mode 4: YOU score the deliveries (and behaviours); the logger deliveries are the grey reference
    % rows (logger_reference_for_scorer.py), j / k jump from one logger delivery to the next
    % modes 1-3: the same logger deliveries as a read-only grey reference row (preset.refFile)
    fr = fullfile(LOGGER_DIR, sprintf('LoggerReference_%s_d2_%s.mat', sessions(k).mouse, ph));
    if isfile(fr)
        sessions(k).autoFile = fr;
        sessions(k).preset.refFile = fr;
        sessions(k).label = [sessions(k).label '   [logger deliveries shown in grey: modes 3 and 4]'];
    end
end
fprintf('bottom videos: %s\n', BOTTOM_DIR);
fprintf('%d sessions paired (bottom + side, equal frame counts):\n', numel(sessions));
fprintf('  %s\n', sessions.label);
fprintf('Output: %s\n\n', OUT_DIR);

msgbox(sprintf(['mini1p Day-2 scoring  (times at %d Hz)\n\n' ...
    'MODE 3 (deliveries + behaviour): you tap the deliveries (keys 1-4) and score\n' ...
    'the behaviours; the logger deliveries are shown in the grey row LOGGER deliv,\n' ...
    'j / k jump to the next / previous logger delivery, y copies the one at the cursor.\n\n' ...
    'MODE 4 (compare with the logger): YOU tap the deliveries (keys 1-4 = this\n' ...
    'session''s order) and score the behaviours; the logger deliveries are the grey\n' ...
    'rows, j / k jump to the next / previous logger delivery, left click on a grey\n' ...
    'mark takes it, right click rejects it. Saved in the ..._reviewed folder.\n' ...
    'MODE 2 (behaviour only): the logger deliveries are already your marks.\n\n' ...
    'HOLD  a attending   s lick/bite   d guarding   f escape\n' ...
    '      g general movement (turn / steps)   h grooming\n' ...
    'TAP   w withdrawal   e flinch   (mode 2 also 1 and 2; mode 4: 1-4 = deliveries)\n' ...
    'SPACE pause   x save and quit (q does nothing)'], FPS_TRUE), 'Scoring keys', 'help');

score_AB_dual_view_mini1p(sessions, OUT_DIR, FPS_TRUE, [OUT_DIR '_reviewed']);
