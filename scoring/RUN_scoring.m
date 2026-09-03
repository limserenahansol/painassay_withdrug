%% RUN_scoring.m  -  open this file in MATLAB and press F5 (Run).
%
%  This is the only file you need to open. It checks your folders, then hands
%  over to score_AB_dual_view.m, which shows the bottom and side videos side by
%  side, with a live label track underneath, and scores the stimulus
%  deliveries and the six behaviours.
%
%  ---------------------------------------------------------------------------
%  STEP 1   Put this session's two videos in these folders, one video each:
%
%             ...\HEAL_mini1p_SBI553\videos\cameraA\    <- BOTTOM view
%             ...\HEAL_mini1p_SBI553\videos\cameraB\    <- SIDE view
%
%           You can leave several sessions in there; you pick which one from a
%           list when the script starts.
%
%  STEP 2   Press F5. First pick WHAT you are scoring:
%
%             1  STIMULUS ONLY   just the deliveries and their type
%             2  BEHAVIOUR ONLY  behaviours, importing a mode-1 delivery file
%             3  BOTH AT ONCE    one pass over the session
%
%           Doing 1 then 2 is slower but easier - marking contact frames is a
%           different kind of attention from watching behaviour. Then fill in
%           the dialog (there is deliberately NO treatment field - blind).
%
%  STEP 3   Press ENTER on the video window to start, then score:
%
%             HOLD   a attending          s licking or biting
%                    d guarding           f escape / rearing
%             TAP    w withdrawal         e flinch
%             TAP    1 2 3 4  = stimulus delivered, by type
%             HOLD   u  = "I cannot tell"  (excluded from classifier training)
%             SPACE pause   left/right step a frame   . faster   , slower
%             z undo last delivery/reflex mark        q stop and save
%
%           TO GO BACK AND FIX SOMETHING:
%             drag the seek bar at the bottom to jump anywhere in the
%             session (click the label track for fine seeking). Then
%             BACKSPACE deletes the one mark nearest the cursor, or just
%             play forward again with the right key held to overwrite a
%             held behaviour.
%
%           A scrolling track under the two videos shows the last 30 s of
%           everything you have marked, so you can see your own label history
%           and catch a stuck key or a missed delivery straight away.
%
%  STEP 4   Results land in  ...\HEAL_mini1p_SBI553\videos\output\
%
%  Stopped half way? Note the time it printed, run again, and put that number
%  in the "Start at time (s)" field.
%  ---------------------------------------------------------------------------

clear; clc; close all;

PROJECT = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553';
VID     = fullfile(PROJECT, 'videos');

% ---- the recording days -------------------------------------------------
%  Each row: label, BOTTOM folder, SIDE folder, OUTPUT folder, day number.
%
%  The output folder is stated explicitly, one per day. It used to be derived
%  from the video folder, and with Day 2 videos in videos\day2\bottom that
%  derivation would have produced videos\day2\output - which no analysis
%  script looks at. Naming it here keeps the scoring output and the analysis
%  input in the same place.
%
%  To add Day 3, copy a row. Rows whose folders hold no video are dropped
%  automatically, so an unused row is harmless.
DAYS = { ...
  'Day 1  (no drug)', fullfile(VID,'cameraA'),      fullfile(VID,'cameraB'), ...
                      fullfile(VID,'output'),       '1'; ...
  'Day 2  (drug)',    fullfile(VID,'day2','bottom'), fullfile(VID,'day2','side'), ...
                      fullfile(VID,'output_day2'),  '2'; ...
  'WT practice',      'C:\Users\hsollim\Research_Projects\08_HEAL_mini1p_SBI553\mousevideo_WT\bottom', ...
                      'C:\Users\hsollim\Research_Projects\08_HEAL_mini1p_SBI553\mousevideo_WT\side', ...
                      fullfile(VID,'output_practice'), '' ...
};

%% ---------------- checks, so you get a clear message not an error --------
addpath(fileparts(mfilename('fullpath')));

exts = {'*.avi', '*.mp4', '*.mov'};

fprintf('Looking for videos...\n\n');
avail = [];
for r = 1:size(DAYS, 1)
    nB = countVideos(DAYS{r,2}, exts);
    nS = countVideos(DAYS{r,3}, exts);
    fprintf('%-18s  bottom %2d   side %2d\n', DAYS{r,1}, nB, nS);
    if nB > 0 && nS > 0
        avail(end+1) = r;   %#ok<SAGROW>
    end
end
fprintf('\n');

if isempty(avail)
    fprintf(2, 'Nothing to score yet. Put one BOTTOM and one SIDE video in:\n');
    for r = 1:size(DAYS, 1)
        fprintf(2, '  %s\n    %s\n    %s\n', DAYS{r,1}, DAYS{r,2}, DAYS{r,3});
    end
    return;
end

if numel(avail) == 1
    pick = avail;
    fprintf('Only %s has videos - using it.\n', DAYS{pick,1});
else
    labels = cell(1, numel(avail));
    for k = 1:numel(avail)
        labels{k} = sprintf('%s   ->   %s', DAYS{avail(k),1}, ...
                            DAYS{avail(k),4});
    end
    [k, tf] = listdlg('PromptString', 'Which recording day?', ...
                      'SelectionMode', 'single', 'ListSize', [560 140], ...
                      'ListString', labels, 'InitialValue', numel(avail));
    if ~tf, disp('Nothing selected. Exiting.'); return; end
    pick = avail(k);
end

bottomFolder = DAYS{pick,2};
sideFolder   = DAYS{pick,3};
outputFolder = DAYS{pick,4};
dayNumber    = DAYS{pick,5};

fprintf('\n%s\n  bottom : %s\n  side   : %s\n  output : %s\n\n', ...
        DAYS{pick,1}, bottomFolder, sideFolder, outputFolder);
fprintf('A list of videos will appear - pick the BOTTOM one first.\n');
score_AB_dual_view(bottomFolder, sideFolder, outputFolder, dayNumber);


%% ------------------------------------------------------------------------
function n = countVideos(folder, exts)
    n = 0;
    if ~exist(folder, 'dir')
        return;
    end
    for i = 1:numel(exts)
        n = n + numel(dir(fullfile(folder, exts{i})));
    end
end
