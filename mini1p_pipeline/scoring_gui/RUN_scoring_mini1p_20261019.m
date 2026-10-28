%% RUN_scoring_mini1p_20261019.m
% Dual-view manual scoring of the 2026-10-19 mini1p sessions (4 mice x baseline + post).
% Same scorer, keys and outputs as mini2p (score_AB_dual_view); see
% score_AB_dual_view_mini1p.m for the four mini1p changes (30 Hz triggered rate, pairing,
% presets, mode-2 offset).
%
% Open this file in the MATLAB desktop and press F5 (needs a window + keyboard).
%
% Keys (same as mini2p / SBI-553):
%   HOLD  a attending   s lick/bite   d guarding   f escape/rear   u uncertain
%   HOLD  g general movement (turning / a few steps)   h grooming   (mini1p additions)
%   TAP   w withdrawal  e flinch
%   TAP   1-4 = this session's stimuli in the order given below (randomisation sheet)
%   SPACE pause   arrows step   . faster   , slower
%   z undo   BACKSPACE delete nearest   x save and quit (q does nothing)
%
% Videos are only READ from K:. Scoring files go to OUT_DIR on C:.

clear; clc; close all;

VIDEO_ROOT = 'K:\mini1p_pain_behavior';        % side view (original AVI, white background kept)
% bottom view: the scoring copies (paws made visible, same frames, 30 fps) of
% make_scoring_videos.py; the original AVIs in K:\...\bottom are used if they are missing
BOTTOM_DIR = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos_scoring\20261019\bottom_enhanced';
if ~isfolder(BOTTOM_DIR) || isempty(dir(fullfile(BOTTOM_DIR, '*.mp4')))
    BOTTOM_DIR = fullfile(VIDEO_ROOT, 'bottom');
end
OUT_DIR    = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261019';
% mode 4 (REVIEW AUTO): the automatic annotation of behavior_auto_annotate.py is read from
% AUTO_DIR and your corrected version is saved in REVIEW_DIR (your manual files in OUT_DIR
% are never touched)
AUTO_DIR   = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261019_auto';
REVIEW_DIR = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261019_reviewed';
FPS_TRUE   = 30;     % Inscopix frame rate = camera trigger rate (AVI header says 60)
DAY_NO     = 1;

% sex from the 2026-10-19 randomisation sheet (sbi mini1p 1019)
mice = struct('id', {'5941', '5937', '5939', '1191'}, 'sex', {'F', 'M', 'M', 'F'});
% Stimulus ORDER per session (randomisation sheet "sbi mini1p 1019"): keys 1-4 follow it,
% key 1 = the first stimulus block of that session. Session numbers as on the sheet.
ord = @(ph, ses, s) struct('phase', ph, 'session', ses, 'stim', {s});
mice(1).order = [ord('Baseline', 1, {'Mild touch', 'Pin prick', 'Light touch', 'Heat'}), ...
                 ord('Post-treatment', 7, {'Pin prick', 'Heat', 'Mild touch', 'Light touch'})];
mice(2).order = [ord('Baseline', 2, {'Heat', 'Mild touch', 'Pin prick', 'Light touch'}), ...
                 ord('Post-treatment', 8, {'Pin prick', 'Heat', 'Light touch', 'Mild touch'})];
mice(3).order = [ord('Baseline', 3, {'Heat', 'Light touch', 'Pin prick', 'Mild touch'}), ...
                 ord('Post-treatment', 9, {'Light touch', 'Mild touch', 'Pin prick', 'Heat'})];
mice(4).order = [ord('Baseline', 4, {'Pin prick', 'Mild touch', 'Heat', 'Light touch'}), ...
                 ord('Post-treatment', 10, {'Light touch', 'Pin prick', 'Heat', 'Mild touch'})];

addpath(fileparts(mfilename('fullpath')));
if ~isfolder(OUT_DIR), mkdir(OUT_DIR); end

sessions = mini1p_pair_sessions(VIDEO_ROOT, mice, DAY_NO, BOTTOM_DIR);
% the automatic annotation of each session, if it exists (same file name as a manual one)
[sessions.autoFile] = deal('');
for k = 1:numel(sessions)
    [~, stem] = fileparts(sessions(k).botFile);
    f = fullfile(AUTO_DIR, ['ScoringAB_' matlab.lang.makeValidName(stem) '.mat']);
    if isfile(f)
        sessions(k).autoFile = f;
        if ismember('refDFrames', who('-file', f))
            sessions(k).label = [sessions(k).label '   [YOURS + AUTO: mode 4, j = next disagreement]'];
        else
            sessions(k).label = [sessions(k).label '   [AUTO ready: mode 4]'];
        end
    end
end
fprintf('bottom videos: %s\n', BOTTOM_DIR);
fprintf('%d sessions paired (bottom + side, equal frame counts):\n', numel(sessions));
fprintf('  %s\n', sessions.label);
fprintf('Output: %s\nReview output (mode 4): %s\n\n', OUT_DIR, REVIEW_DIR);

msgbox(sprintf(['mini1p scoring  (times at %d Hz)\n\n' ...
    'TAP at contact:  keys 1-4 = THIS session''s stimulus order\n' ...
    '                 (shown in the session list and on the start screen)\n' ...
    'HOLD  a attending   s lick/bite   d guarding   f escape\n' ...
    '      g general movement (turn / steps)   h grooming\n' ...
    'TAP   w withdrawal  e flinch\n' ...
    'SPACE pause   x save and quit (q does nothing)\n\n' ...
    'MODE 4 (REVIEW AUTO): the automatic marks are loaded; they stay when you play\n' ...
    'through. HOLD a key to overwrite, HOLD c to erase, BACKSPACE deletes a tap,\n' ...
    'j / k jump to the next / previous place to check.\n' ...
    'Sessions you scored yourself show YOUR marks (editable) + AUTO marks (grey rows);\n' ...
    'y copies the AUTO mark at the cursor into yours.'], FPS_TRUE), 'Scoring keys', 'help');

score_AB_dual_view_mini1p(sessions, OUT_DIR, FPS_TRUE, REVIEW_DIR);
