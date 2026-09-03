function score_AB_dual_view(bottomFolder, sideFolder, outputFolder, dayDefault)
% SCORE_AB_DUAL_VIEW  Score with BOTH cameras visible, plus a live label track.
%
%   Both videos are shown side by side in one window (bottom view left, side
%   view right) so a single keystroke can be based on both. Underneath them a
%   scrolling track shows the last 30 s of everything you have marked, so you
%   can see your own label history while you score and catch a stuck key or a
%   missed delivery immediately.
%
%   Frames are STREAMED, not buffered. A 50,000-frame DV session would need
%   ~52 GB of RAM to buffer, which is why score_A could not open these files.
%   Measured on the WT DV files: dual-stream playback runs at ~135 fps
%   (4.5x real time). The first seek after ENTER takes ~2 s; frame stepping
%   after that is 1-20 ms in both directions.
%
%   ------------------------------ THREE MODES -------------------------------
%   You pick one at the start.
%
%     1  STIMULUS ONLY   mark only the deliveries and their type.
%                        -> DeliveryTimes_*.mat / .csv, DeliveryCounts_*.csv
%     2  BEHAVIOUR ONLY  score only the behaviours; imports the delivery times
%                        from a previous mode-1 pass.
%     3  BOTH AT ONCE    deliveries and behaviours in a single pass.
%
%   Doing stimulus first (mode 1, then mode 2) is slower but easier: marking
%   contact frames is a different kind of attention from watching behaviour.
%   Mode 3 is one pass over a 28 min session instead of two.
%
%   ------------------------------- KEYS -------------------------------------
%   Same muscle memory as manual_scoring_video_multibehavior_batch2.m: SPACE
%   pauses, q stops, affective behaviours are held on the home row from "a".
%
%   ENTER   start
%
%   AFFECTIVE-MOTIVATIONAL  (HOLD the key for as long as the behaviour lasts)
%     a  paw attending                    b  no mouth contact
%     s  licking or biting                b  any mouth contact with the paw
%     d  sustained lifting / guarding      b  paw held up > 2 s, no weight-bearing
%     f  escape / rearing
%
%   REFLEXIVE  (tap once per event)  - the number row is MODE-DEPENDENT
%     mode 2 (BEHAVIOUR ONLY)   1  paw withdrawal     2  flinch / flick
%                               deliveries are imported, so the number row is
%                               free. w and e still work as aliases.
%     mode 1 and 3              w  paw withdrawal     e  flinch / flick
%                               1-4 are taken by the stimulus type, below.
%
%   STIMULUS DELIVERY  (tap at the moment of contact - this sets the epoch)
%   Modes 1 and 3 only.
%     1  stimulus 1   2  stimulus 2   3  stimulus 3   4  stimulus 4
%     0  delivery of UNKNOWN type  (avoid: it has no normalisation denominator)
%
%   UNCERTAIN  (HOLD)
%     u  "I cannot tell what this is"  -> flagged exclude_from_training = 1
%
%   CONTROL
%     SPACE  pause / resume        (p also works)
%     left / right   step 1 frame (while paused)
%     .  faster      ,  slower     (1.5x per press, 0.1x - 8x)
%     q  stop and save             (x also works)
%
%   GOING BACK TO FIX A MISTAKE
%     MOUSE      click or drag on the seek bar at the bottom to jump anywhere
%                in the session, or click the label track for fine seeking
%                within the last 30 s. Seeking auto-pauses, like a video
%                player. Playing forward again overwrites the held-key
%                behaviours for the frames you pass through.
%     BACKSPACE  delete the ONE mark nearest the cursor (reflex or delivery),
%                within +/- 0.4 s. Press again for the next one. Use this
%                rather than z after seeking - z only undoes the most recent
%                mark, which is not the one you came back for.
%     z          undo the most recent mark (fine while scoring forward)
%   --------------------------------------------------------------------------
%
%   LICKING AND BITING ARE ONE CATEGORY
%     They were separate in an earlier version. They are pooled here because
%     the two cannot be told apart reliably at this magnification - the tongue
%     is ~5 px in the side view (see qc/WT_RECORDING_QC.md). Scoring them
%     separately would have produced a distinction the video cannot support.
%
%   Outputs into <bottomFolder>\..\output\ :
%     DeliveryTimes_<vid>.csv / .mat  delivery frame, time, stimulus type
%     DeliveryCounts_<vid>.csv        n delivered per stimulus  <- denominator
%     RawScores_<vid>.csv             one row per delivery
%     Normalized_<vid>.csv            one row per stimulus type, per-stimulus rates
%     TrainingLabels_<vid>.csv        frame-level labels for a classifier
%     ScoringAB_<vid>.mat             everything
%     BehaviorTimeSeries / RasterPlot / NormalizedPerStim  .png
%
%   Scoring is BLIND: treatment is never entered or displayed.
%
%   Hansol Lim - HEAL mini1p / SBI-553

    clc; close all;

    root = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos';
    if nargin < 1 || isempty(bottomFolder), bottomFolder = fullfile(root, 'cameraA'); end
    if nargin < 2 || isempty(sideFolder),   sideFolder   = fullfile(root, 'cameraB'); end

    if ~exist(bottomFolder, 'dir'), error('No bottom folder: %s', bottomFolder); end
    if ~exist(sideFolder, 'dir'),   error('No side folder: %s',   sideFolder);   end

    % Output folder is now an explicit argument. It used to be derived as
    % <parent of bottomFolder>\output, which was fine while there was only
    % one day, but Day 2 videos live in videos\day2\bottom - and that would
    % have derived videos\day2\output while every analysis script and every
    % line of PIPELINE.md expects videos\output_day2. Worse, if the Day 2
    % videos had been dropped straight into cameraA the derivation would have
    % pointed at videos\output and written Day 2 scoring on top of Day 1.
    if nargin < 3 || isempty(outputFolder)
        outputFolder = fullfile(fileparts(bottomFolder), 'output');
    end
    % Pre-fill the "Day number" field so it cannot be mistyped session after
    % session. Still editable in the dialog.
    if nargin < 4, dayDefault = ''; end
    if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end
    probe = fullfile(outputFolder, 'test_write.txt');
    fid = fopen(probe, 'w');
    if fid == -1
        error('Cannot write to %s.', outputFolder);
    else
        fclose(fid); delete(probe);
    end

    bot = listVideos(bottomFolder);
    sid = listVideos(sideFolder);
    if isempty(bot), error('No videos in %s', bottomFolder); end
    if isempty(sid), error('No videos in %s', sideFolder);   end

    MODES = {'1  STIMULUS ONLY  - mark deliveries and their type', ...
             '2  BEHAVIOUR ONLY - score behaviours, import delivery times', ...
             '3  BOTH AT ONCE   - deliveries and behaviours in one pass'};

    keepGoing = true;
    while keepGoing
        [im, tf] = listdlg('PromptString', 'What are you scoring?', ...
                           'SelectionMode', 'single', 'ListSize', [520 110], ...
                           'ListString', MODES, 'InitialValue', 3);
        if ~tf, disp('Nothing selected. Exiting.'); break; end
        mode = im;

        [ib, tf] = listdlg('PromptString', 'BOTTOM view (camera A) video:', ...
                           'SelectionMode', 'single', 'ListSize', [520 320], ...
                           'ListString', {bot.name});
        if ~tf, disp('Nothing selected. Exiting.'); break; end

        guess = matchByName(bot(ib).name, {sid.name});
        [is, tf] = listdlg('PromptString', 'SIDE view (camera B) video:', ...
                           'SelectionMode', 'single', 'ListSize', [520 320], ...
                           'ListString', {sid.name}, 'InitialValue', guess);
        if ~tf, disp('Nothing selected. Exiting.'); break; end

        fprintf('\n===== mode %d =====\nbottom: %s\nside  : %s\n', ...
                mode, bot(ib).name, sid(is).name);
        scoreDual(fullfile(bot(ib).folder, bot(ib).name), ...
                  fullfile(sid(is).folder, sid(is).name), outputFolder, ...
                  mode, dayDefault);

        if strcmpi(questdlg('Another pass?', 'Continue', 'Yes', 'No', 'Yes'), 'No')
            keepGoing = false;
        end
    end
    disp('Done.');
end


%% ------------------------------------------------------------------ %%
function v = listVideos(folder)
    v = [dir(fullfile(folder, '*.avi')); ...
         dir(fullfile(folder, '*.mp4')); ...
         dir(fullfile(folder, '*.mov'))];
end


function k = matchByName(name, candidates)
% Pick the side file whose name shares the longest prefix with the bottom file,
% so "female1bottom0001 ..." preselects "female1side0001 ...".
    k = 1; best = -1;
    a = lower(name);
    for i = 1:numel(candidates)
        b = lower(candidates{i});
        n = min(numel(a), numel(b));
        j = 0;
        while j < n && a(j + 1) == b(j + 1), j = j + 1; end
        if j > best, best = j; k = i; end
    end
end


%% ------------------------------------------------------------------ %%
function scoreDual(botFile, sidFile, outputFolder, mode, dayDefault)
    if nargin < 5, dayDefault = ''; end

    global gBehav gPaused gStopped gStarted gTapQ gDelQ gUndo gStep gSpeedCmd ...
           gUncertain gStimKeys gSeekTo gCurIdx gDelNear ...
           gRelPending gRelClock gDragging gDragClock gSeekCommit;
    gBehav = 0; gPaused = false; gStopped = false; gStarted = false;
    gTapQ = []; gDelQ = []; gUndo = false; gStep = 0; gSpeedCmd = 0;
    gUncertain = false; gSeekTo = NaN; gCurIdx = 1; gDelNear = false;
    gDragging = false; gDragClock = tic; gSeekCommit = false;
    DRAG_WATCHDOG_S = 3.0;   % a drag this old must have lost its button-up
    gRelPending = false; gRelClock = tic;
    RELEASE_DEBOUNCE_S = 0.25;   % a release must survive this long to count

    % ---- FOUR affective behaviours; licking and biting are one category ----
    affNames = {'None', 'Paw attending', 'Licking or biting', ...
                'Sustained lifting / guarding', 'Escape / rearing'};
    affKeys  = {'a', 's', 'd', 'f'};
    refNames = {'Paw withdrawal', 'Flinch / flick'};
    nAff     = 4;
    defStim  = {'Light touch', 'Mild touch', 'Heat', 'Pin prick'};

    doStim  = (mode == 1 || mode == 3);
    doBehav = (mode == 2 || mode == 3);
    % number row means stimulus type only when deliveries are being marked
    gStimKeys = doStim;

    % ---- mode 2 needs the delivery times from a mode-1 pass ----
    impSec = []; impType = []; stimNames = defStim;
    if mode == 2
        [fA, pA] = uigetfile({'DeliveryTimes_*.mat', 'Delivery times'}, ...
            'Select the DeliveryTimes_*.mat from your STIMULUS pass', outputFolder);
        if isequal(fA, 0)
            warning('No delivery file selected - nothing to score against.');
            return;
        end
        A = load(fullfile(pA, fA));
        impSec = A.deliverySec(:);
        if isfield(A, 'deliveryType')
            impType = A.deliveryType(:);
            stimNames = reshape(A.stimNames(1:4), 1, []);
        else
            impType = zeros(numel(impSec), 1);
            warning(['That file has no stimulus types, so nothing can be ' ...
                     'normalised. Re-run the stimulus pass.']);
        end
        fprintf('Imported %d deliveries from %s\n', numel(impSec), fA);
    end

    META_KEYS = {'sessionNo', 'mouseID', 'sexID', 'dayNo', 'phase'};
    META_SLOT = [1 2 3 4 5];

    % ---- resume a part-scored session? -------------------------------------
    % Without this, restarting with "Start at time" set would save a file whose
    % earlier frames are all zero, silently throwing away the part you had
    % already scored. So: if a ScoringAB file for this video already exists,
    % offer to carry it forward and continue where it stopped.
    [~, vidStem] = fileparts(botFile);
    vidStem = matlab.lang.makeValidName(vidStem);
    priorPath = fullfile(outputFolder, ['ScoringAB_' vidStem '.mat']);
    prior = [];  resumeAt = 0;
    if exist(priorPath, 'file')
        P = load(priorPath);
        if isfield(P, 'score') && isfield(P, 'nUsed') && isfield(P, 'frameRate')
            gotTo = double(P.nUsed) / double(P.frameRate);
            q = questdlg(sprintf(['Existing scoring found for this video, ' ...
                    'stopped at %.1f s (%.1f min).\n\n' ...
                    'Continue from there and keep what you already scored?'], ...
                    gotTo, gotTo / 60), ...
                'Resume?', 'Resume', 'Start over', 'Resume');
            if strcmp(q, 'Resume')
                prior = P;
                resumeAt = gotTo;
                fprintf('Resuming: keeping %.1f s of existing scoring.\n', gotTo);
            elseif strcmp(q, 'Start over')
                fprintf(2, ['Starting over. The existing ScoringAB file will ' ...
                            'be OVERWRITTEN when you save.\n']);
            else
                disp('Cancelled.'); return;
            end
        end
    end

    % ---- session dialog ----
    prompts = {'Session number (from randomisation sheet)', ...
               'Mouse ID', 'Sex (M/F)', 'Day (1 or 2)', ...
               'Phase (Baseline / Post-treatment)', ...
               'SIDE minus BOTTOM sync offset (s, 0 if hardware-synced)', ...
               'Observation window per stimulus (s)  [TBC]', ...
               'Guarding minimum hold (s)  [>2 s]', ...
               'Start at time (s)  - to resume a part-scored video', ...
               'Playback speed (1 = real time)', ...
               'Key-press lag to correct in TRAINING labels only (ms)', ...
               'Live track window (s)'};
    defaults = {'', '', '', dayDefault, 'Baseline', '0', '30', '2', ...
                sprintf('%.1f', resumeAt), '1', '250', '30'};
    % carry the prior session's metadata across so it does not get retyped
    if ~isempty(prior)
        for k = 1:numel(META_KEYS)
            if isfield(prior, META_KEYS{k})
                v = strtrim(char(string(prior.(META_KEYS{k}))));
                if ~isempty(v), defaults{META_SLOT(k)} = v; end
            end
        end
    end
    if mode == 1
        prompts  = prompts([1 6 9 10 12]);
        defaults = defaults([1 6 9 10 12]);
    end
    % inputdlg returns a COLUMN cell, so stimNames loaded from a mode-1 .mat is
    % 4x1 while defaults is 1x12. Force a row before concatenating.
    stimNames = reshape(stimNames, 1, []);
    meta = inputdlg([prompts, ...
                     {'Stimulus 1 name (key 1)', 'Stimulus 2 name (key 2)', ...
                      'Stimulus 3 name (key 3)', 'Stimulus 4 name (key 4)'}], ...
                    sprintf('Mode %d - do NOT enter treatment', mode), 1, ...
                    [defaults, stimNames]);
    if isempty(meta), disp('Cancelled.'); return; end

    if mode == 1
        sessionNo = meta{1}; mouseID = ''; sexID = ''; dayNo = ''; phase = '';
        syncOff = str2double(meta{2}); obsWindow = 30; guardMin = 2;
        startSec = str2double(meta{3}); playSpeed = str2double(meta{4});
        lagMs = 0; trackWin = str2double(meta{5});
        stimNames = reshape(meta(6:9), 1, []);
    else
        sessionNo = meta{1}; mouseID = meta{2}; sexID = meta{3};
        dayNo = meta{4};     phase = meta{5};
        syncOff   = str2double(meta{6});
        obsWindow = str2double(meta{7});
        guardMin  = str2double(meta{8});
        startSec  = str2double(meta{9});
        playSpeed = str2double(meta{10});
        lagMs     = str2double(meta{11});
        trackWin  = str2double(meta{12});
        stimNames = reshape(meta(13:16), 1, []);
    end
    if isnan(syncOff),   syncOff = 0;   end
    if isnan(obsWindow) || obsWindow <= 0, obsWindow = 30; end
    if isnan(guardMin)  || guardMin  < 0,  guardMin  = 2;  end
    if isnan(startSec)  || startSec  < 0,  startSec  = 0;  end
    if isnan(playSpeed) || playSpeed <= 0, playSpeed = 1;  end
    if isnan(lagMs)     || lagMs     < 0,  lagMs     = 0;  end
    if isnan(trackWin)  || trackWin  < 5,  trackWin  = 30; end
    for s = 1:4
        if isempty(strtrim(stimNames{s})), stimNames{s} = defStim{s}; end
    end

    % ---- open both, stream ----
    vb = VideoReader(botFile);
    vs = VideoReader(sidFile);
    frameRate = vb.FrameRate;
    nB = floor(vb.Duration * frameRate);
    nS = floor(vs.Duration * vs.FrameRate);
    offFrames = round(syncOff * frameRate);
    if abs(vs.FrameRate - frameRate) > 0.05
        warning('Frame rates differ (bottom %.2f, side %.2f). Using bottom.', ...
                frameRate, vs.FrameRate);
    end
    numFrames = min(nB, nS - offFrames);
    if numFrames < 1
        error('Sync offset %.2f s leaves no overlapping frames.', syncOff);
    end
    fprintf('bottom %d frames, side %d, offset %d -> scoring %d frames\n', ...
            nB, nS, offFrames, numFrames);

    score  = zeros(numFrames, 1);
    uncert = false(numFrames, 1);
    idx0   = max(1, round(startSec * frameRate) + 1);

    % ---- carry the earlier work forward ----
    if ~isempty(prior)
        ps = double(prior.score(:));
        n  = min(numel(ps), numFrames);
        score(1:n) = ps(1:n);
        if isfield(prior, 'uncert')
            pu = logical(prior.uncert(:));
            m = min(numel(pu), numFrames);
            uncert(1:m) = pu(1:m);
        end
        if isfield(prior, 'reflexEvents') && ~isempty(prior.reflexEvents)
            gTapQ = prior.reflexEvents;
        end
        % deliveries: mode 2 already imported them, so only take the prior
        % ones when we are the pass that marks deliveries
        if doStim && isfield(prior, 'dFrames') && ~isempty(prior.dFrames)
            gDelQ = [double(prior.dFrames(:)), double(prior.dTypes(:))];
        end
        fprintf(['Carried forward: %d scored frame(s), %d reflex mark(s), ' ...
                 '%d delivery mark(s).\n'], sum(score > 0), size(gTapQ, 1), ...
                size(gDelQ, 1));
        if idx0 <= n
            fprintf(2, ['NOTE: you are restarting at %.1f s but %.1f s was ' ...
                        'already scored.\n      Frames you play through again ' ...
                        'will be OVERWRITTEN.\n'], startSec, n / frameRate);
        end
    end

    % imported deliveries become pre-loaded marks in mode 2
    if mode == 2
        f = round((impSec + syncOff) * frameRate) + 1;
        ok = f >= 1 & f <= numFrames;
        if ~all(ok)
            warning('%d imported delivery time(s) fall outside the video.', ...
                    sum(~ok));
        end
        gDelQ = [f(ok), impType(ok)];
    end

    % ---- splash ----
    hFig = figure('Name', 'DUAL VIEW - bottom (left) + side (right)', ...
                  'NumberTitle', 'off', 'Color', 'k');
    set(hFig, 'WindowKeyPressFcn', @gKeyDown, 'WindowKeyReleaseFcn', @gKeyUp);
    axes('Position', [0 0 1 1], 'Color', 'k'); axis off; hold on;
    L = {sprintf('MODE %d', mode), ''};
    if doBehav
        L = [L, {'AFFECTIVE (HOLD)   a attending      s licking or biting', ...
                 '                   d guarding       f escape / rearing', ''}];
        if doStim
            % number row is taken by the stimulus type, so reflexes stay on w/e
            L = [L, {'REFLEXIVE (tap)    w withdrawal     e flinch', ''}];
        else
            L = [L, {'REFLEXIVE (tap)    1 withdrawal     2 flinch', ...
                     '                   (w and e also work)', ''}];
        end
        L = [L, {'UNCERTAIN (HOLD)   u   -> excluded from classifier training', ''}];
    end
    if doStim
        L = [L, {'DELIVERY (tap at contact)', ...
                 sprintf('   1 %s     2 %s', stimNames{1}, stimNames{2}), ...
                 sprintf('   3 %s     4 %s     0 unknown', stimNames{3}, stimNames{4}), ...
                 ''}];
    else
        L = [L, {sprintf('%d deliveries imported - shown as lines on the track', ...
                 size(gDelQ, 1)), ''}];
    end
    L = [L, {'SPACE pause   left/right step   . faster  , slower   z undo   q stop', ...
             'MOUSE: drag the seek bar to jump    BACKSPACE: delete mark at cursor', ...
             '', 'Press ENTER to start'}];
    for i = 1:numel(L)
        text(0.5, 0.95 - 0.055 * i, L{i}, 'Color', 'w', 'FontSize', 11.5, ...
             'FontName', 'Consolas', 'HorizontalAlignment', 'center', ...
             'Units', 'normalized');
    end
    drawnow;
    disp('Waiting for ENTER...');
    while ~gStarted
        pause(0.1);
        if ~ishandle(hFig), disp('Closed before start.'); return; end
    end

    % ---- layout: video, live label track, then a full-session seek bar ----
    clf(hFig); set(hFig, 'Color', 'k');
    axV = axes('Parent', hFig, 'Position', [0.02 0.38 0.96 0.58]);
    [fb, fs] = readPair(vb, vs, idx0, offFrames, frameRate);
    hImg = imshow(pairImage(fb, fs), 'Parent', axV);
    hTtl = title(axV, '', 'Color', 'w', 'FontSize', 10, 'FontName', 'Consolas');

    trackFrames = max(30, round(trackWin * frameRate));
    nRow = nAff + 2;                       % + withdrawal row + flinch row
    axT = axes('Parent', hFig, 'Position', [0.06 0.17 0.92 0.19], 'Color', 'k');
    hTrack = image(axT, 'CData', ones(nRow, trackFrames, 3));
    set(axT, 'XLim', [0.5 trackFrames + 0.5], 'YLim', [0.5 nRow + 0.5], ...
             'YDir', 'reverse', 'XTick', [], 'YColor', 'w', 'XColor', 'w', ...
             'YTick', 1:nRow, 'YTickLabel', ...
             [{'attend', 'lick/bite', 'guard', 'escape'}, {'withdraw', 'flinch'}], ...
             'FontSize', 8, 'TickLength', [0 0], 'Box', 'on');
    ylabel(axT, '');

    % full-session seek bar: click or drag anywhere on it to jump there
    SEEKH = 40;                                  % rows in the seek strip image
    axS = axes('Parent', hFig, 'Position', [0.06 0.075 0.92 0.075], 'Color', 'k');
    hSeek = image(axS, 'CData', zeros(SEEKH, 1000, 3));
    set(axS, 'XLim', [0.5 1000.5], 'YLim', [0.5 SEEKH + 0.5], 'YTick', [], ...
             'XColor', 'w', 'YColor', 'w', 'FontSize', 8, ...
             'TickLength', [0 0], 'Box', 'on');
    nt = 6;
    set(axS, 'XTick', linspace(1, 1000, nt), 'XTickLabel', ...
        arrayfun(@(f) sprintf('%.0f', f / frameRate), ...
                 linspace(1, numFrames, nt), 'UniformOutput', false));
    xlabel(axS, ['whole session (s)  -  CLICK or DRAG here to jump  ' ...
                 '|  click the track above for fine seek  |  ' ...
                 'BACKSPACE deletes marks at the cursor'], ...
           'Color', 'w', 'FontSize', 8);

    % both are draggable; the callbacks convert an x position into a frame
    set(hSeek,  'ButtonDownFcn', @(s, e) seekDown(axS, 'session', ...
                                                  numFrames, trackFrames));
    set(axS,    'ButtonDownFcn', @(s, e) seekDown(axS, 'session', ...
                                                  numFrames, trackFrames));
    set(hTrack, 'ButtonDownFcn', @(s, e) seekDown(axT, 'track', ...
                                                  numFrames, trackFrames));

    % 'Enable','inactive' so a stray click on the status bar cannot steal
    % keyboard focus from the figure and silently kill the key callbacks
    hSub = uicontrol('Style', 'text', 'Enable', 'inactive', ...
                     'Parent', hFig, 'Units', 'normalized', ...
                     'Position', [0 0 1 0.05], 'BackgroundColor', 'k', ...
                     'ForegroundColor', [0.7 1 0.7], 'FontName', 'Consolas', ...
                     'FontSize', 9, 'HorizontalAlignment', 'center', 'String', '');
    drawnow;

    affRGB = [0.20 0.45 0.75;      % attending
              0.85 0.35 0.15;      % licking or biting
              0.55 0.30 0.70;      % guarding
              0.20 0.65 0.35];     % escape / rearing
    refRGB = [0.90 0.75 0.10;      % withdrawal
              0.55 0.55 0.55];     % flinch

    idx   = idx0;
    seqOK = true;
    tick  = 0;
    while ishandle(hFig) && ~gStopped && idx <= numFrames
        tStart = tic;

        if gSpeedCmd ~= 0
            playSpeed = max(0.1, min(8, playSpeed * (1.5 ^ gSpeedCmd)));
            gSpeedCmd = 0;
        end

        gCurIdx = idx;                 % the seek callbacks need to know this

        % Watchdog: if a drag has been "active" for longer than any real
        % click-and-drag, its button-up was lost. Detach the motion callback
        % before it can freeze the window with repeated cold seeks.
        if gDragging && toc(gDragClock) > DRAG_WATCHDOG_S
            releaseDrag(hFig);
            fprintf('  (drag watchdog: released a stuck seek)\n');
        end

        % A KeyRelease only ends the hold once it has survived the debounce
        % in WALL-CLOCK time without a new key-down. Auto-repeat is a
        % real-time phenomenon, so debouncing in loop iterations would be
        % wrong at 4x playback. See gKeyUp for why this exists at all.
        if gRelPending && toc(gRelClock) >= RELEASE_DEBOUNCE_S
            gBehav = 0; gRelPending = false;
        end

        % stamp pending taps with the current frame
        if ~isempty(gTapQ), gTapQ(gTapQ(:, 1) == 0, 1) = idx; end
        if doStim && ~isempty(gDelQ), gDelQ(gDelQ(:, 1) == 0, 1) = idx; end
        if gUndo, undoLast(doStim); gUndo = false; end
        if gDelNear
            [gTapQ, gDelQ] = deleteNear(gTapQ, gDelQ, idx, ...
                                        round(0.40 * frameRate), doStim, ...
                                        frameRate, stimNames, refNames);
            gDelNear = false;
        end

        if doBehav
            score(idx)  = gBehav;
            uncert(idx) = gUncertain;
        end

        % ---- fetch frames ----
        if gSeekCommit
            % the mouse button has been released: move for real. While the
            % drag is still in progress we only move the cursor on the seek
            % bar, because a cold read() costs ~2 s and doing one per mouse
            % movement is what used to hang the window.
            idx = min(max(round(gSeekTo), 1), numFrames);
            gSeekTo = NaN; gSeekCommit = false;
            gStep = 0;
            [fb, fs] = readPair(vb, vs, idx, offFrames, frameRate);
            seqOK = false;
        elseif gDragging
            % preview only - no frame read
            set(hSeek, 'CData', seekImage(score, gDelQ, ...
                min(max(round(gSeekTo), 1), numFrames), numFrames, ...
                SEEKH, 1000, nAff, affRGB));
            set(hTtl, 'String', sprintf('SEEK to %6.1f s   (release to jump)', ...
                (min(max(round(gSeekTo), 1), numFrames) - 1) / frameRate));
            drawnow limitrate;
            pause(0.02);
            continue;
        elseif gPaused && gStep ~= 0
            idx = min(max(idx + gStep, 1), numFrames);
            gStep = 0;
            [fb, fs] = readPair(vb, vs, idx, offFrames, frameRate);
            seqOK = false;
        elseif ~gPaused
            if ~seqOK
                [fb, fs] = readPair(vb, vs, idx, offFrames, frameRate);
                seqOK = true;
            elseif hasFrame(vb) && hasFrame(vs)
                fb = readFrame(vb); fs = readFrame(vs);
            else
                break;
            end
        end

        % ---- draw video ----
        set(hImg, 'CData', pairImage(fb, fs));
        if isempty(gDelQ)
            lastD = 'last: -';
        else
            lastD = sprintf('last: %s', typeName(gDelQ(end, 2), stimNames));
        end
        set(hTtl, 'String', sprintf( ...
            '%s  %6.1f s  (frame %d/%d)  %.2fx   |   NOW: %s   |   %s', ...
            tern(gPaused, 'PAUSED ', 'PLAYING'), (idx - 1) / frameRate, ...
            idx, numFrames, playSpeed, affNames{gBehav + 1}, lastD));

        % ---- draw the live label track (cheap: every 4th frame) ----
        tick = tick + 1;
        if mod(tick, 4) == 0 || gPaused || gStep ~= 0
            set(hTrack, 'CData', trackImage(score, gTapQ, gDelQ, idx, ...
                trackFrames, nAff, nRow, affRGB, refRGB));
        end
        % ---- seek bar: cheaper still, it only has to track the cursor ----
        if mod(tick, 15) == 0 || gPaused
            set(hSeek, 'CData', seekImage(score, gDelQ, idx, numFrames, ...
                SEEKH, 1000, nAff, affRGB));
        end

        set(hSub, 'String', sprintf('deliveries  %s        reflex events %d', ...
            countString(gDelQ, stimNames), size(gTapQ, 1)));
        drawnow limitrate;

        if gPaused
            pause(0.03);
        else
            idx = idx + 1;
            pause(max(0, 1 / (frameRate * playSpeed) - toc(tStart)));
        end
    end

    nUsed = min(idx - 1, numFrames);
    if ~isempty(prior)
        % Never shrink below what was carried forward. Without this, resuming
        % at 500 s and stopping at 600 s would truncate the file to 600 s and
        % throw away everything the earlier pass had scored after that.
        nUsed = max(nUsed, min(double(prior.nUsed), numFrames));
    end
    if nUsed < 1, disp('Nothing scored.'); return; end
    score    = score(1:nUsed);
    uncert   = uncert(1:nUsed);
    timeAxis = (0:nUsed - 1) / frameRate;
    if ~isempty(gTapQ), gTapQ = gTapQ(gTapQ(:, 1) > 0 & gTapQ(:, 1) <= nUsed, :); end
    if ~isempty(gDelQ), gDelQ = gDelQ(gDelQ(:, 1) > 0 & gDelQ(:, 1) <= nUsed, :); end

    if isempty(gDelQ)
        warning('No deliveries - nothing can be normalised. Not saving.');
        return;
    end

    [~, vidName] = fileparts(botFile);
    vidName = matlab.lang.makeValidName(vidName);

    [dFrames, ord] = sort(gDelQ(:, 1));
    dTypes  = gDelQ(ord, 2);

    % Two taps can land on the SAME frame when playing fast, which would give
    % that delivery a zero-length observation window and NaN percentages.
    % Collapse them and say so rather than writing NaN into the results.
    dup = [false; diff(dFrames) == 0];
    if any(dup)
        warning(['%d delivery mark(s) landed on an already-marked frame ' ...
                 '(double tap at %.1fx?). Keeping the first of each.'], ...
                 sum(dup), playSpeed);
        dFrames = dFrames(~dup);
        dTypes  = dTypes(~dup);
    end
    % Flag suspiciously close pairs too - these are usually a double tap, but
    % they are kept because they can be real back-to-back deliveries.
    close = find(diff(dFrames) < round(1.0 * frameRate));
    if ~isempty(close)
        fprintf(2, ['  NOTE: %d delivery pair(s) less than 1 s apart ' ...
                    '(at %s s). Check these.\n'], numel(close), ...
                strjoin(arrayfun(@(f) sprintf('%.1f', (f - 1) / frameRate), ...
                        dFrames(close), 'UniformOutput', false), ', '));
    end

    dSec    = (dFrames - 1) / frameRate;
    reflexEvents = gTapQ;

    if doStim
        writeDeliveries(outputFolder, vidName, sessionNo, dFrames, dTypes, ...
                        dSec, stimNames, frameRate, nUsed);
    end

    if ~doBehav
        fprintf('\nStimulus pass complete: %d deliveries.\n', numel(dFrames));
        fprintf(['Next: run again in mode 2 and select\n' ...
                 '   DeliveryTimes_%s.mat\n'], vidName);
        fprintf('Scored to %.1f s of %.1f s.\n', nUsed / frameRate, ...
                numFrames / frameRate);
        return;
    end

    holdQC(score, frameRate, affNames, guardMin);

    [T, winSec, rep] = perDelivery(score, reflexEvents, dFrames, dTypes, ...
                                   frameRate, obsWindow, guardMin, nUsed);
    writeRawScores(outputFolder, vidName, sessionNo, dayNo, mouseID, sexID, ...
                   phase, dTypes, rep, winSec, T, stimNames, guardMin);
    writeNormalized(outputFolder, vidName, sessionNo, dayNo, mouseID, sexID, ...
                    phase, dTypes, winSec, T, stimNames);
    writeTrainingLabels(outputFolder, vidName, score, uncert, timeAxis, ...
                        reflexEvents, dFrames, dTypes, stimNames, affNames, ...
                        nUsed, frameRate, lagMs);
    makeFigures(outputFolder, vidName, score, timeAxis, reflexEvents, dFrames, ...
                dTypes, T, winSec, stimNames, affNames, sessionNo, mouseID, ...
                nUsed, nAff, affRGB);

    matPath = fullfile(outputFolder, ['ScoringAB_' vidName '.mat']);
    save(matPath, 'score', 'uncert', 'timeAxis', 'affNames', 'affKeys', ...
         'refNames', 'reflexEvents', 'dFrames', 'dSec', 'dTypes', 'stimNames', ...
         'rep', 'T', 'winSec', 'syncOff', 'offFrames', 'frameRate', 'nUsed', ...
         'obsWindow', 'guardMin', 'lagMs', 'mode', 'sessionNo', 'mouseID', ...
         'sexID', 'dayNo', 'phase', 'botFile', 'sidFile');
    fprintf('MAT written: %s\n', matPath);
    fprintf('\nScored to %.1f s of %.1f s.\n', nUsed / frameRate, ...
            numFrames / frameRate);
    if nUsed < numFrames
        fprintf(['Stopped early. To resume, run again and set "Start at time" ' ...
                 'to %.1f s.\n'], nUsed / frameRate);
    end
end


%% ---------------- the live label track ---------------------------- %%
function img = trackImage(score, tapQ, delQ, idx, W, nAff, nRow, affRGB, refRGB)
% Scrolling raster of the last W frames. Right edge is "now".
% Built as an RGB image and pushed with one set(CData) call, so it costs
% O(nRow x W) per update and does not slow the video down.

    img = ones(nRow, W, 3);
    i0  = max(1, idx - W + 1);
    seg = score(i0:idx);
    off = W - numel(seg);                       % blank left pad near t = 0

    % affective behaviour rows
    for b = 1:nAff
        on = false(1, W);
        on(off + 1:end) = (seg(:)' == b);
        for ch = 1:3
            row = ones(1, W);
            row(on) = affRGB(b, ch);
            img(b, :, ch) = row;
        end
    end

    % reflex event rows - widen each tap to 3 px so a single frame is visible
    if ~isempty(tapQ)
        for r = 1:2
            f = tapQ(tapQ(:, 2) == r, 1);
            f = f(f >= i0 & f <= idx);
            col = off + (f - i0 + 1);
            for c = col(:)'
                lo = max(1, c - 1); hi = min(W, c + 1);
                for ch = 1:3
                    img(nAff + r, lo:hi, ch) = refRGB(r, ch);
                end
            end
        end
    end

    % deliveries - a black line down every row
    if ~isempty(delQ)
        f = delQ(delQ(:, 1) > 0, 1);
        f = f(f >= i0 & f <= idx);
        col = off + (f - i0 + 1);
        for c = col(:)'
            lo = max(1, c - 1); hi = min(W, c + 1);
            img(:, lo:hi, :) = 0;
        end
    end

    % faint gridline every second-ish, and a light band for the pad
    if off > 0
        img(:, 1:off, :) = img(:, 1:off, :) * 0.90;
    end
end


%% ---------------- mouse seeking (click / drag) --------------------- %%
function seekDown(ax, kind, numFrames, trackFrames)
% Mouse pressed on the seek bar or on the label track. Auto-pause (a video
% player does the same) and follow the mouse until the button is released.
%
%   WHY THE gDragging FLAG AND THE WATCHDOG EXIST
%   If the button is released outside the figure, or over a component that
%   swallows the event, WindowButtonUpFcn never fires. The motion callback
%   then stays attached and every stray mouse movement requests a new seek.
%   Each seek is a cold read(), which can take ~2 s, so the window appears to
%   freeze permanently and moving the mouse keeps it frozen. That is exactly
%   the hang this guards against: seekMove refuses to do anything unless
%   gDragging is set, and releaseDrag is called from the main loop as soon as
%   the drag looks stale.
    global gSeekAx gSeekKind gSeekN gSeekW gPaused gDragging gDragClock;
    gSeekAx = ax; gSeekKind = kind;
    gSeekN = numFrames; gSeekW = trackFrames;
    gPaused = true;
    gDragging = true;
    gDragClock = tic;
    fig = ancestor(ax, 'figure');
    set(fig, 'WindowButtonMotionFcn', @(s, e) seekMove(), ...
             'WindowButtonUpFcn',     @(s, e) releaseDrag(fig));
    seekMove();
end


function seekMove()
    global gSeekAx gSeekKind gSeekN gSeekW gSeekTo gCurIdx gDragging;
    if ~gDragging, return; end                      % not an active drag
    if isempty(gSeekAx) || ~isgraphics(gSeekAx), return; end
    cp = get(gSeekAx, 'CurrentPoint');
    x  = cp(1, 1);
    switch gSeekKind
        case 'session'
            % the strip is 1000 columns wide and spans the whole recording
            f = round((x - 1) / 999 * (gSeekN - 1)) + 1;
        case 'track'
            % trackImage puts frame gCurIdx in the LAST column, so
            % column c corresponds to frame gCurIdx + c - trackFrames
            f = gCurIdx + round(x) - gSeekW;
        otherwise
            return;
    end
    f = min(max(f, 1), gSeekN);
    if f ~= gCurIdx                    % never re-seek to where we already are
        gSeekTo = f;
    end
end


function releaseDrag(fig)
% End the drag and, critically, detach the motion callback.
    global gSeekAx gDragging gSeekTo gSeekCommit;
    if isgraphics(fig)
        set(fig, 'WindowButtonMotionFcn', '', 'WindowButtonUpFcn', '');
    end
    gDragging = false;
    gSeekAx = [];
    % commit whatever the last position under the mouse was. Reading frames
    % on every mouse move would mean a cold read (~2 s) per pixel dragged.
    if ~isnan(gSeekTo), gSeekCommit = true; end
end


function img = seekImage(score, delQ, idx, numFrames, H, W, nAff, affRGB)
% Thin overview strip for the whole session: where you are, where the
% deliveries are, and which behaviour is scored where.
    img = ones(H, W, 3) * 0.12;                     % dark background
    col = @(f) min(max(round((f - 1) / max(numFrames - 1, 1) * (W - 1)) + 1, 1), W);

    % scored behaviour, collapsed to one colour per column
    n = min(numel(score), numFrames);
    if n > 1
        edges = round(linspace(1, n, W + 1));
        for c = 1:W
            seg = score(edges(c):max(edges(c), edges(c + 1) - 1));
            b = max(seg);                            % show the "strongest" code
            if b >= 1 && b <= nAff
                for ch = 1:3
                    img(1:H - 8, c, ch) = affRGB(b, ch);
                end
            end
        end
    end

    % delivery ticks along the bottom
    if ~isempty(delQ)
        f = delQ(delQ(:, 1) > 0, 1);
        for c = arrayfun(col, f(:)')
            img(H - 7:H, max(1, c - 1):min(W, c + 1), :) = 1;
        end
    end

    % current position: a full-height red cursor
    c = col(idx);
    img(:, max(1, c - 1):min(W, c + 1), 1) = 1;
    img(:, max(1, c - 1):min(W, c + 1), 2) = 0.1;
    img(:, max(1, c - 1):min(W, c + 1), 3) = 0.1;
end


function [tapQ, delQ] = deleteNear(tapQ, delQ, idx, win, doStim, frameRate, ...
                                   stimNames, refNames)
% BACKSPACE: remove the ONE mark nearest the cursor, within +/- win frames.
%
%   "z" only undoes the most recent mark, which is useless once you have
%   seeked backwards to fix an earlier mistake. This is the tool for that:
%   park the cursor on the bad mark and press BACKSPACE.
%
%   Deliberately deletes only the SINGLE nearest mark, not everything in the
%   window. In real sessions marks come as close as 0.9 s apart, so a
%   delete-all-in-window rule would take out the neighbour too. Press again
%   to remove the next one.

    [dTap, rTap] = nearestMark(tapQ, idx);
    if doStim
        [dDel, rDel] = nearestMark(delQ, idx);
    else
        dDel = inf; rDel = 0;
    end

    if min(dTap, dDel) > win
        bestKind = 0; bestRow = 0;
    elseif dTap <= dDel
        bestKind = 1; bestRow = rTap;
    else
        bestKind = 2; bestRow = rDel;
    end

    switch bestKind
        case 1
            fprintf('  deleted reflex "%s" at %.2f s\n', ...
                    refNames{tapQ(bestRow, 2)}, ...
                    (tapQ(bestRow, 1) - 1) / frameRate);
            tapQ(bestRow, :) = [];
        case 2
            fprintf('  deleted delivery "%s" at %.2f s\n', ...
                    typeName(delQ(bestRow, 2), stimNames), ...
                    (delQ(bestRow, 1) - 1) / frameRate);
            delQ(bestRow, :) = [];
        otherwise
            fprintf('  no mark within %.2f s of %.2f s\n', ...
                    win / frameRate, (idx - 1) / frameRate);
    end
end


function [dist, row] = nearestMark(Q, idx)
% Distance in frames from idx to the closest already-stamped mark in Q.
    dist = inf; row = 0;
    if isempty(Q), return; end
    d = abs(Q(:, 1) - idx);
    d(Q(:, 1) <= 0) = inf;          % marks not yet stamped with a frame
    [dist, row] = min(d);
end


%% ------------------------------------------------------------------ %%
function [fb, fs] = readPair(vb, vs, idx, offFrames, frameRate)
% Random-access read of an aligned pair. Used at start and while stepping.
    fb = read(vb, idx);
    js = max(1, idx + offFrames);
    fs = read(vs, js);
    vb.CurrentTime = min(idx / frameRate, vb.Duration - 1 / frameRate);
    vs.CurrentTime = min(js / vs.FrameRate, vs.Duration - 1 / vs.FrameRate);
end


function im = pairImage(fb, fs)
% Bottom left, side right, matched height, thin green divider.
    if size(fb, 1) ~= size(fs, 1)
        fs = imresize(fs, [size(fb, 1) NaN]);
    end
    if size(fb, 3) == 1, fb = repmat(fb, 1, 1, 3); end
    if size(fs, 3) == 1, fs = repmat(fs, 1, 1, 3); end
    bar = zeros(size(fb, 1), 6, 3, class(fb));
    bar(:, :, 2) = intmax(class(fb));
    im = [fb, bar, fs];
end


function undoLast(doStim)
% Remove whichever mark was made most recently.
    global gTapQ gDelQ;
    lastTap = 0; lastDel = 0;
    if ~isempty(gTapQ), lastTap = gTapQ(end, 1); end
    if doStim && ~isempty(gDelQ), lastDel = gDelQ(end, 1); end
    if doStim && lastDel >= lastTap && ~isempty(gDelQ)
        fprintf('  undo: delivery at frame %d\n', gDelQ(end, 1));
        gDelQ(end, :) = [];
    elseif ~isempty(gTapQ)
        fprintf('  undo: reflex event at frame %d\n', gTapQ(end, 1));
        gTapQ(end, :) = [];
    end
end


function nm = typeName(code, stimNames)
    if code >= 1 && code <= 4, nm = stimNames{code}; else, nm = 'UNKNOWN'; end
end


function str = countString(delQ, stimNames)
    if isempty(delQ), t = []; else, t = delQ(:, 2); end
    parts = cell(1, 4);
    for s = 1:4
        nm = stimNames{s};
        parts{s} = sprintf('%s %d', nm(1:min(4, end)), sum(t == s));
    end
    str = strjoin(parts, '   ');
    nU = sum(t == 0);
    if nU > 0, str = sprintf('%s   UNK %d', str, nU); end
end


function out = tern(c, a, b)
    if c, out = a; else, out = b; end
end


%% ------------------------------------------------------------------ %%
function [T, winSec, rep] = perDelivery(score, tapQ, dFrames, dTypes, ...
                                        frameRate, obsWindow, guardMin, nUsed)
% T columns: withdrawal, flinch, attC, attD, lbC, lbD, grdC, grdD, escC, escD
    obsFrames   = round(obsWindow * frameRate);
    guardFrames = round(guardMin  * frameRate);
    nStim  = numel(dFrames);
    T      = zeros(nStim, 10);
    winSec = zeros(nStim, 1);
    rep    = zeros(nStim, 1);
    seen   = zeros(1, 5);

    for k = 1:nStim
        slot = dTypes(k);
        if slot < 1 || slot > 4, slot = 5; end
        seen(slot) = seen(slot) + 1;
        rep(k) = seen(slot);

        f0 = dFrames(k);
        f1 = min(f0 + obsFrames - 1, nUsed);
        if k < nStim, f1 = min(f1, dFrames(k + 1) - 1); end
        win = f0:f1;
        winSec(k) = numel(win) / frameRate;

        ev = [];
        if ~isempty(tapQ)
            ev = tapQ(tapQ(:, 1) >= f0 & tapQ(:, 1) <= f1, 2);
        end
        s = score(win);
        [attC, attD] = episodeStats(s, 1, frameRate, 0);
        [lbC,  lbD]  = episodeStats(s, 2, frameRate, 0);
        [grdC, grdD] = episodeStats(s, 3, frameRate, guardFrames);
        [escC, escD] = episodeStats(s, 4, frameRate, 0);

        T(k, :) = [double(any(ev == 1)), sum(ev == 2), ...
                   attC, attD, lbC, lbD, grdC, grdD, escC, escD];
    end
end


function holdQC(score, frameRate, affNames, guardMin)
% Did the affective keys actually get HELD? Print a verdict before saving.
%
%   This check exists because it went wrong once. In female1/female2 the
%   lick/bite "bouts" had a median length of 2-3 frames separated by 2-4 frame
%   gaps: partly a code bug (a KeyRelease fired during Windows key auto-repeat,
%   now debounced) and partly the keys being tapped instead of held. Every
%   duration measure was unusable and the > 2 s guarding rule discarded 31 of
%   32 bouts. Neither was obvious from the figures.
%
%   A held key gives bouts of seconds. If the medians below are a few
%   hundred milliseconds, the durations in this file are not usable and the
%   session needs re-scoring.

    fprintf('\n  ---- hold check: were the a/s/d/f keys held? ----\n');
    fprintf('  %-30s %7s %9s %8s %8s\n', ...
            'behaviour', 'bouts', 'median', 'longest', '<=3 frm');
    bad = {};
    for b = 1:4
        bin = (score(:) == b);
        d = diff([0; bin; 0]);
        st = find(d == 1); en = find(d == -1) - 1;
        if isempty(st)
            fprintf('  %-30s %7d %9s %8s %8s\n', affNames{b + 1}, 0, ...
                    '-', '-', '-');
            continue;
        end
        L = (en - st + 1) / frameRate;
        shortFrac = mean((en - st + 1) <= 3);
        fprintf('  %-30s %7d %8.2fs %7.2fs %7.0f%%\n', ...
                affNames{b + 1}, numel(L), median(L), max(L), 100 * shortFrac);
        if median(L) < 0.5 || shortFrac > 0.3
            bad{end + 1} = affNames{b + 1};  %#ok<AGROW>
        end
    end

    if isempty(bad)
        fprintf('  -> looks like real holds. Counts AND durations are usable.\n');
        return;
    end

    if guardMin > 0
        % durations are still being relied on somewhere, so this is a problem
        fprintf(2, ['\n  *** WARNING: %s look TAPPED, not held. ***\n' ...
                    '  Bouts of a few hundred ms cannot be real episodes.\n' ...
                    '  The DURATION measures in this file are not usable, and the\n' ...
                    '  > %.1f s guarding rule will discard almost every bout.\n' ...
                    '  Either HOLD the keys for as long as the behaviour lasts,\n' ...
                    '  or - if you only want COUNTS - set "Guarding minimum\n' ...
                    '  hold" to 0 so the duration rule stops filtering.\n'], ...
                strjoin(bad, ', '), guardMin);
    else
        % guardMin = 0 means the scorer has chosen count-only scoring
        fprintf(['  -> tapped, and the guarding minimum is 0, so this is\n' ...
                 '     COUNT-ONLY scoring. That is a valid choice. Two things\n' ...
                 '     follow and both belong in the methods:\n' ...
                 '       1. duration columns here are meaningless - report\n' ...
                 '          counts and rates only, never %% of time.\n' ...
                 '       2. the "> 2 s" guarding criterion is now YOUR\n' ...
                 '          judgement at the keyboard, not a rule the code\n' ...
                 '          applies. Only press d when the paw really stays up.\n']);
    end
end


function [nEp, totSec] = episodeStats(s, code, frameRate, minFrames)
% Episodes shorter than minFrames are discarded (the guarding > 2 s threshold).
    binary = (s(:) == code);
    d  = diff([0; binary; 0]);
    st = find(d == 1); en = find(d == -1) - 1;
    keep = (en - st + 1) >= max(minFrames, 1);
    st = st(keep); en = en(keep);
    nEp    = numel(st);
    totSec = sum(en - st + 1) / frameRate;
end


%% ------------------------------------------------------------------ %%
function writeDeliveries(out, vid, sess, dFrames, dTypes, dSec, stimNames, fr, nUsed)
    p = fullfile(out, ['DeliveryTimes_' vid '.csv']);
    fid = fopen(p, 'w');
    fprintf(fid, 'Session,Delivery,Stim code,Stimulus,Frame,Time_s\n');
    for j = 1:numel(dFrames)
        fprintf(fid, '%s,%d,%d,%s,%d,%.4f\n', sess, j, dTypes(j), ...
                typeName(dTypes(j), stimNames), dFrames(j), dSec(j));
    end
    fclose(fid);

    nPerStim = zeros(1, 4);
    for s = 1:4, nPerStim(s) = sum(dTypes == s); end
    nUnknown = sum(dTypes == 0);

    c = fullfile(out, ['DeliveryCounts_' vid '.csv']);
    fid = fopen(c, 'w');
    fprintf(fid, 'Session,Stim code,Stimulus,n_delivered\n');
    for s = 1:4
        fprintf(fid, '%s,%d,%s,%d\n', sess, s, stimNames{s}, nPerStim(s));
    end
    if nUnknown > 0, fprintf(fid, '%s,0,UNKNOWN,%d\n', sess, nUnknown); end
    fclose(fid);

    fprintf('\n  deliveries per stimulus (normalisation denominator):\n');
    for s = 1:4, fprintf('    %-14s %d\n', stimNames{s}, nPerStim(s)); end
    if nUnknown > 0
        warning('%d delivery/deliveries are UNKNOWN and cannot be normalised.', ...
                nUnknown);
    end
    if any(nPerStim == 0)
        warning('Never delivered: %s', strjoin(stimNames(nPerStim == 0), ', '));
    end

    deliveryFrames = dFrames; deliverySec = dSec; deliveryType = dTypes;
    frameRate = fr; numFrames = nUsed; sessionNo = sess; vidName = vid;
    save(fullfile(out, ['DeliveryTimes_' vid '.mat']), ...
         'deliveryFrames', 'deliverySec', 'deliveryType', 'stimNames', ...
         'nPerStim', 'nUnknown', 'frameRate', 'numFrames', 'sessionNo', 'vidName');
    fprintf('  wrote %s\n  wrote %s\n', p, c);
end


function writeRawScores(out, vid, sess, day, mouse, sex, phase, dTypes, rep, ...
                        winSec, T, stimNames, guardMin)
    p = fullfile(out, ['RawScores_' vid '.csv']);
    fid = fopen(p, 'w');
    fprintf(fid, ['Session,Day,Mouse ID,Sex,Phase,Treatment,Trial,Stim code,' ...
                  'Stimulus,Rep,Obs window (s),Withdrawal (0/1),Flinch count,' ...
                  'Attending count,Attending dur (s),' ...
                  'Lick/bite count,Lick/bite dur (s),' ...
                  'Guarding count,Guarding dur (s),' ...
                  'Escape/rear count,Escape/rear dur (s),Scorer,Video file,Notes\n']);
    for k = 1:numel(dTypes)
        fprintf(fid, ['%s,%s,%s,%s,%s,%s,%d,%d,%s,%d,%.2f,' ...
                      '%d,%d,%d,%.2f,%d,%.2f,%d,%.2f,%d,%.2f,%s,%s,%s\n'], ...
            sess, day, mouse, sex, phase, 'BLIND', k, dTypes(k), ...
            typeName(dTypes(k), stimNames), rep(k), winSec(k), ...
            T(k, 1), T(k, 2), T(k, 3), T(k, 4), T(k, 5), T(k, 6), ...
            T(k, 7), T(k, 8), T(k, 9), T(k, 10), ...
            '', [vid '.avi'], sprintf('dual view; guard min %.1fs', guardMin));
    end
    fclose(fid);
    fprintf('  wrote %s\n', p);
end


function writeNormalized(out, vid, sess, day, mouse, sex, phase, dTypes, ...
                         winSec, T, stimNames)
%  per_stim = total / n_delivered        -> counts
%  pct_time = total dur / observed time  -> durations (robust to truncated windows)
    p = fullfile(out, ['Normalized_' vid '.csv']);
    fid = fopen(p, 'w');
    fprintf(fid, ['Session,Day,Mouse ID,Sex,Phase,Treatment,Stim code,Stimulus,' ...
                  'n_delivered,obs_time_s,withdrawal_n,withdrawal_rate,' ...
                  'flinch_n,flinch_per_stim,attending_n,attending_per_stim,' ...
                  'attending_dur_s,attending_s_per_stim,attending_pct_time,' ...
                  'lickbite_n,lickbite_per_stim,lickbite_dur_s,' ...
                  'lickbite_s_per_stim,lickbite_pct_time,' ...
                  'guarding_n,guarding_per_stim,guarding_dur_s,' ...
                  'guarding_s_per_stim,guarding_pct_time,' ...
                  'escape_n,escape_per_stim,escape_dur_s,' ...
                  'escape_s_per_stim,escape_pct_time\n']);

    fprintf('\n  ---- normalised summary ----\n');
    fprintf('  %-14s %4s %8s %9s %11s %9s\n', ...
            'stimulus', 'n', 'obs(s)', 'withdr.', 'lick/bite', 'guard');
    for s = 1:4
        m = (dTypes == s); n = sum(m);
        fprintf(fid, '%s,%s,%s,%s,%s,%s,%d,%s', ...
                sess, day, mouse, sex, phase, 'BLIND', s, stimNames{s});
        if n == 0
            fprintf(fid, ',0,0.00');
            fprintf(fid, repmat(',NA', 1, 24));
            fprintf(fid, '\n');
            continue;
        end
        obsT = sum(winSec(m)); c = sum(T(m, :), 1);
        if obsT <= 0
            warning(['%s has %d delivery/deliveries but zero observed time. ' ...
                     'Percentages cannot be computed for it.'], stimNames{s}, n);
        end
        pct = @(x) 100 * x / obsT;          % obsT > 0 after the dedupe above
        fprintf(fid, ',%d,%.2f', n, obsT);
        fprintf(fid, ',%.0f,%.4f', c(1), c(1) / n);
        fprintf(fid, ',%.0f,%.4f', c(2), c(2) / n);
        fprintf(fid, ',%.0f,%.4f,%.2f,%.4f,%.2f', ...
                c(3), c(3) / n, c(4), c(4) / n, pct(c(4)));
        fprintf(fid, ',%.0f,%.4f,%.2f,%.4f,%.2f', ...
                c(5), c(5) / n, c(6), c(6) / n, pct(c(6)));
        fprintf(fid, ',%.0f,%.4f,%.2f,%.4f,%.2f', ...
                c(7), c(7) / n, c(8), c(8) / n, pct(c(8)));
        fprintf(fid, ',%.0f,%.4f,%.2f,%.4f,%.2f', ...
                c(9), c(9) / n, c(10), c(10) / n, pct(c(10)));
        fprintf(fid, '\n');
        fprintf('  %-14s %4d %8.1f %9.2f %10.1f%% %8.1f%%\n', ...
                stimNames{s}, n, obsT, c(1) / n, pct(c(6)), pct(c(8)));
    end
    nU = sum(dTypes == 0);
    if nU > 0
        fprintf(fid, '%s,%s,%s,%s,%s,%s,0,UNKNOWN,%d,%.2f', ...
                sess, day, mouse, sex, phase, 'BLIND', nU, sum(winSec(dTypes == 0)));
        fprintf(fid, repmat(',NA', 1, 24));
        fprintf(fid, '\n');
    end
    fclose(fid);
    fprintf('  wrote %s\n', p);
end


function writeTrainingLabels(out, vid, score, uncert, timeAxis, tapQ, dFrames, ...
                             dTypes, stimNames, affNames, nUsed, frameRate, lagMs)
% Frame-level labels for training an automatic classifier later.
%
%   Two label columns on purpose:
%     affective_code     exactly as scored - use for the behavioural stats
%     affective_code_ml  shifted EARLIER by the key-press lag - use for training
%
%   A human presses the key a few hundred ms after the behaviour starts, and
%   releases it a few hundred ms after it ends. Left uncorrected that lag
%   teaches the classifier the behaviour begins later than it does, which caps
%   its achievable accuracy. Shifting only the ML copy leaves the scored
%   measures untouched.
%
%   exclude_from_training = 1 on frames held with 'u', and on frames whose
%   shifted label ran off the end of the video.

    lagFrames = round(lagMs / 1000 * frameRate);
    p = fullfile(out, ['TrainingLabels_' vid '.csv']);
    fid = fopen(p, 'w');
    fprintf(fid, ['frame,time_s,affective_code,affective_label,' ...
                  'affective_code_ml,affective_label_ml,' ...
                  'withdrawal,flinch,withdrawal_ml,flinch_ml,' ...
                  'stim_index,stim_code,stimulus,sec_since_delivery,' ...
                  'uncertain,exclude_from_training\n']);

    wV = zeros(nUsed, 1); fV = zeros(nUsed, 1);
    if ~isempty(tapQ)
        wV(tapQ(tapQ(:, 2) == 1, 1)) = 1;
        fV(tapQ(tapQ(:, 2) == 2, 1)) = 1;
    end

    pad = zeros(lagFrames, 1);
    if lagFrames > 0 && lagFrames < nUsed
        scoreML = [score(1 + lagFrames:end); pad];
        wML     = [wV(1 + lagFrames:end);    pad];
        fML     = [fV(1 + lagFrames:end);    pad];
        offEnd  = [false(nUsed - lagFrames, 1); true(lagFrames, 1)];
    else
        scoreML = score; wML = wV; fML = fV; offEnd = false(nUsed, 1);
    end

    dt = 1 / frameRate;
    for f = 1:nUsed
        si = sum(dFrames <= f);
        if si >= 1
            sc = dTypes(si); sn = typeName(sc, stimNames);
            since = (f - dFrames(si)) * dt;
        else
            sc = -1; sn = 'pre-stimulus'; since = NaN;
        end
        excl = double(uncert(f) || offEnd(f));
        fprintf(fid, '%d,%.4f,%d,%s,%d,%s,%d,%d,%d,%d,%d,%d,%s,%.4f,%d,%d\n', ...
                f, timeAxis(f), ...
                score(f),   affNames{score(f) + 1}, ...
                scoreML(f), affNames{scoreML(f) + 1}, ...
                wV(f), fV(f), wML(f), fML(f), ...
                si, sc, sn, since, double(uncert(f)), excl);
    end
    fclose(fid);
    fprintf('  wrote %s\n', p);
    fprintf(['    lag correction %d ms (%d frames); %d frame(s) excluded ' ...
             'from training (%.1f%%)\n'], ...
            lagMs, lagFrames, sum(uncert | offEnd), ...
            100 * mean(uncert | offEnd));
end


%% ------------------------------------------------------------------ %%
function makeFigures(out, vid, score, timeAxis, tapQ, dFrames, dTypes, T, ...
                     winSec, stimNames, affNames, sess, mouse, nUsed, nAff, affRGB)
    stimC = [0.20 0.45 0.75; 0.85 0.55 0.15; 0.75 0.25 0.25; 0.25 0.55 0.35];

    f1 = figure('Name', 'Behaviour time series', 'NumberTitle', 'off', ...
                'Visible', 'off');
    hold on;
    for b = 1:nAff
        m = (score == b);
        if any(m)
            scatter(timeAxis(m), score(m), 20, affRGB(b, :), 'filled', ...
                    'DisplayName', affNames{b + 1});
        end
    end
    for k = 1:numel(dFrames)
        st = dTypes(k);
        if st >= 1 && st <= 4, col = stimC(st, :); else, col = [0 0 0]; end
        xline(timeAxis(dFrames(k)), '--', sprintf('%d', st), 'Color', col, ...
              'LineWidth', 1.1, 'HandleVisibility', 'off');
    end
    if ~isempty(tapQ)
        plot(timeAxis(tapQ(:, 1)), repmat(nAff + 0.6, size(tapQ, 1), 1), 'v', ...
             'MarkerSize', 5, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none', ...
             'HandleVisibility', 'off');
    end
    ylim([0 nAff + 1.2]); yticks(1:nAff); yticklabels(affNames(2:nAff + 1));
    xlabel('Time (s)');
    title('Affective behaviour + reflex events (dashed = delivery, number = stim)');
    legend('Location', 'eastoutside'); grid on;
    saveas(f1, fullfile(out, ['BehaviorTimeSeries_' vid '.png'])); close(f1);

    f2 = figure('Name', 'Raster', 'NumberTitle', 'off', 'Visible', 'off');
    for b = 1:nAff
        subplot(nAff, 1, b);
        bin = (score == b);
        d = diff([0; bin(:); 0]);
        sI = find(d == 1); eI = find(d == -1) - 1;
        n = min(numel(sI), numel(eI));
        for j = 1:n
            line([timeAxis(sI(j)) timeAxis(min(eI(j), nUsed))], [j j], ...
                 'LineWidth', 2, 'Color', affRGB(b, :)); hold on;
        end
        for k = 1:numel(dFrames)
            st = dTypes(k);
            if st >= 1 && st <= 4, col = stimC(st, :); else, col = [0 0 0]; end
            xline(timeAxis(dFrames(k)), '--', 'Color', col);
        end
        xlabel('Time (s)'); ylabel('Ep'); title(affNames{b + 1}); grid on;
    end
    saveas(f2, fullfile(out, ['RasterPlot_' vid '.png'])); close(f2);

    f3 = figure('Name', 'Normalised per stimulus', 'NumberTitle', 'off', ...
                'Visible', 'off');
    lbl = cell(1, 4); wr = nan(1, 4); lb = nan(1, 4); gd = nan(1, 4); nn = zeros(1, 4);
    for s = 1:4
        m = (dTypes == s); lbl{s} = stimNames{s}; nn(s) = sum(m);
        if nn(s) == 0, continue; end
        obsT = sum(winSec(m)); c = sum(T(m, :), 1);
        wr(s) = c(1) / nn(s);
        lb(s) = 100 * c(6) / obsT;
        gd(s) = 100 * c(8) / obsT;
    end
    subplot(1, 3, 1); bar(wr); ylim([0 1]);
    set(gca, 'XTickLabel', lbl, 'XTickLabelRotation', 30);
    ylabel('withdrawals / stimulus'); title('Withdrawal rate'); grid on;
    subplot(1, 3, 2); bar(lb);
    set(gca, 'XTickLabel', lbl, 'XTickLabelRotation', 30);
    ylabel('% of observed time'); title('Licking or biting'); grid on;
    subplot(1, 3, 3); bar(gd);
    set(gca, 'XTickLabel', lbl, 'XTickLabelRotation', 30);
    ylabel('% of observed time'); title('Guarding'); grid on;
    sgtitle(sprintf('Session %s  mouse %s   -   n delivered: %s', sess, mouse, ...
        strjoin(arrayfun(@(x) sprintf('%d', x), nn, 'UniformOutput', false), ' / ')));
    saveas(f3, fullfile(out, ['NormalizedPerStim_' vid '.png'])); close(f3);
end


%% ---------------------------- key callbacks ------------------------ %%
function gKeyDown(~, event)
% The number row is MODE-DEPENDENT, because it cannot mean two things at once:
%
%   mode 1 (stimulus) and mode 3 (both)   1-4 = stimulus type, 0 = unknown
%                                         reflexes are on w and e
%   mode 2 (behaviour only)               deliveries are imported, so the
%                                         number row is free:
%                                         1 = withdrawal, 2 = flinch
%                                         (w and e still work as aliases)
%
% gStimKeys carries which map is active; scoreDual sets it from the mode.

    global gBehav gPaused gStopped gStarted gTapQ gDelQ gUndo gStep gSpeedCmd ...
           gUncertain gStimKeys gDelNear gRelPending;
    switch lower(event.Key)
        case 'return'
            if ~gStarted, gStarted = true; disp('Playing...'); end
        % ---- delete every mark at the cursor (for fixing after a seek) ----
        case {'backspace', 'delete'}, gDelNear = true;
        % ---- affective: HOLD (home row) ----
        % a new key-down cancels any pending release, which is what collapses
        % keyboard auto-repeat back into one continuous hold
        case 'a', gBehav = 1; gRelPending = false;   % paw attending
        case 's', gBehav = 2; gRelPending = false;   % licking or biting
        case 'd', gBehav = 3; gRelPending = false;   % sustained lift / guarding
        case 'f', gBehav = 4; gRelPending = false;   % escape / rearing
        % ---- uncertain: HOLD ----
        case 'u', gUncertain = true;
        % ---- reflexive: TAP - always available on w / e ----
        case 'w', gTapQ(end + 1, :) = [0 1];
        case 'e', gTapQ(end + 1, :) = [0 2];
        % ---- number row: stimulus type, or reflexes in behaviour-only mode ----
        case '1'
            if gStimKeys, gDelQ(end + 1, :) = [0 1];
            else,         gTapQ(end + 1, :) = [0 1];   % withdrawal
            end
        case '2'
            if gStimKeys, gDelQ(end + 1, :) = [0 2];
            else,         gTapQ(end + 1, :) = [0 2];   % flinch
            end
        case '3'
            if gStimKeys, gDelQ(end + 1, :) = [0 3]; end
        case '4'
            if gStimKeys, gDelQ(end + 1, :) = [0 4]; end
        case '0'
            if gStimKeys, gDelQ(end + 1, :) = [0 0]; end
        % ---- control ----
        case {'space', 'p'}, gPaused = ~gPaused;
        case 'leftarrow',  gStep = -1;
        case 'rightarrow', gStep =  1;
        case {'comma', 'subtract', 'hyphen'},      gSpeedCmd = -1;
        case {'period', 'add', 'equal', 'equals'}, gSpeedCmd =  1;
        case 'z', gUndo = true;
        case {'q', 'x'}, gStopped = true; disp('Stopped by user.');
    end
end


function gKeyUp(~, event)
% DO NOT clear the behaviour here.
%
%   Windows keyboard auto-repeat makes MATLAB fire spurious KeyRelease events
%   while a key is still physically held. Clearing on release therefore chops
%   one real episode into a train of 2-3 frame fragments separated by 2-4
%   frame gaps - measured in female1/female2, where lick/bite bouts had a
%   median length of 2-3 frames and a median GAP of 4 frames. Every duration
%   measure was destroyed and the > 2 s guarding rule discarded 31 of 32
%   bouts.
%
%   Instead we only FLAG the release. The main loop clears the behaviour once
%   the release has survived RELEASE_DEBOUNCE_TICKS loop iterations without a
%   new key-down for the same key. Auto-repeat cancels the pending release, a
%   real finger lift does not.
    global gUncertain gRelPending gRelClock;
    switch lower(event.Key)
        case {'a', 's', 'd', 'f'}
            gRelPending = true;
            gRelClock = tic;          % wall clock: auto-repeat is real-time
        case 'u'
            gUncertain = false;
    end
end
