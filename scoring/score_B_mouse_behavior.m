function score_B_mouse_behavior(folderPath)
% SCORE_B_MOUSE_BEHAVIOR  Camera B — score the 7 pain behaviours, per stimulus.
%
%   *** OBSOLETE - DO NOT USE. Its output no longer matches the scoring book. ***
%
%   Use  score_AB_dual_view.m  in MODE 2 instead. Mode 2 does exactly what this
%   file did - score behaviours against an existing set of delivery times -
%   but with both cameras visible and a live label track.
%
%   Why this file must not be used any more:
%     * it scores SEVEN behaviours with licking and biting SEPARATE. They are
%       now one pooled category, because the two cannot be told apart at this
%       magnification (the tongue is ~5 px; see qc/WT_RECORDING_QC.md).
%     * its keys are the old map: taps 1/2 for the reflexes, holds a-g for the
%       affective behaviours. The current map is w/e for reflexes, a-f for
%       affective, and 1-4 for the stimulus type.
%     * its RawScores CSV has "Licking count/dur" and "Biting count/dur"
%       columns, which no longer exist in Behavioural_scoring_book.xlsx. Pasting
%       its output into the book will silently misalign every column after
%       "Attending dur".
%
%   It is kept only as a record of the earlier two-pass workflow.
%
%   Stimulus epochs are NOT guessed here. They are imported from the camera A
%   pass (score_A_stimulus_delivery.m) together with the STIMULUS TYPE of each
%   delivery, optionally shifted by a camera sync offset. Reflexive and
%   affective behaviours stay separate — no combined score.
%
%   WHY THE TYPE MATTERS
%     Mice do not receive the same number of each stimulus (e.g. 10 pin pricks
%     for one mouse, 11 for another). Raw totals are therefore not comparable
%     between animals. This script writes a normalised table in which every
%     measure is divided by the number of that stimulus actually delivered, and
%     every duration is ALSO expressed as a fraction of the time actually
%     observed — because a window can be cut short when the next stimulus
%     arrives early, so n_delivered alone is not a sufficient denominator.
%
%   THE SEVEN BEHAVIOURS
%     reflexive   tap  1  Paw withdrawal          brief, < ~1 s
%                 tap  2  Flinch / flick          brief shake at contact
%     affective   hold a  Paw attending           orients to / inspects the paw
%                 hold s  Licking                 rhythmic tongue on the paw
%                 hold d  Biting                  jaw closes / grasps the paw
%                 hold f  Sustained lift / guard  paw held up > 2 s, no weight-bearing
%                 hold g  Escape / rearing        locomotor escape or rear
%
%   LICKING vs BITING - operational rule
%     Licking  rhythmic tongue protrusion against the paw, head steady
%     Biting   jaw closure on the paw, usually with a head jerk or pull
%     If you cannot tell them apart, score LICKING. Biting is therefore the
%     conservative, specific measure, and licking+biting pooled stays exact.
%
%   ENTER start   p pause/resume   z undo last tap   q stop
%
%   Outputs into <folderPath>\output\ :
%     RawScores_<vid>.csv            one row per DELIVERY, Raw_scores column order
%     Normalized_<vid>.csv           one row per STIMULUS TYPE, per-stimulus rates
%     ScoringB_<vid>.mat             frame-wise score, reflexEvents, epochs, metadata
%     BehaviorTimeSeries_<vid>.png   affective trace + reflex marks + delivery lines
%     RasterPlot_<vid>.png           one raster per affective behaviour
%     TrainingLabels_<vid>.csv       frame-level labels for classifier training
%
%   Scoring is BLIND: treatment is never entered or displayed.
%
%   Hansol Lim - HEAL mini1p / SBI-553

    clc; close all;

    if nargin < 1
        folderPath = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos\cameraB';
    end
    outputFolder = fullfile(folderPath, 'output');
    if ~exist(folderPath, 'dir'), error('Folder does not exist: %s', folderPath); end
    if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

    testFile = fullfile(outputFolder, 'test_write.txt');
    fid = fopen(testFile, 'w');
    if fid == -1
        error('Cannot write to %s.', outputFolder);
    else
        fclose(fid); delete(testFile);
    end

    videoFiles = [dir(fullfile(folderPath, '*.mp4')); ...
                  dir(fullfile(folderPath, '*.avi')); ...
                  dir(fullfile(folderPath, '*.mov'))];
    if isempty(videoFiles), error('No videos in %s', folderPath); end
    videoNames = {videoFiles.name};

    keepGoing = true;
    while keepGoing
        [indx, tf] = listdlg('PromptString', 'CAMERA B - select video:', ...
                             'SelectionMode', 'single', 'ListSize', [420 320], ...
                             'ListString', videoNames);
        if ~tf, disp('Nothing selected. Exiting.'); break; end
        v = videoFiles(indx);
        fprintf('\n===== Camera B: %s =====\n', v.name);
        scoreBehaviour(fullfile(v.folder, v.name), outputFolder);
        if strcmpi(questdlg('Another video?', 'Continue', 'Yes', 'No', 'Yes'), 'No')
            keepGoing = false;
        end
    end
    disp('Camera B pass complete.');
end


%% ------------------------------------------------------------------ %%
function scoreBehaviour(videoFile, outputFolder)

    global bBehav bPaused bStopped bStarted bTapQ bUndo;
    bBehav = 0; bPaused = false; bStopped = false; bStarted = false;
    bTapQ = []; bUndo = false;

    affNames = {'None', 'Paw attending', 'Licking', 'Biting', ...
                'Sustained lifting / guarding', 'Escape / rearing'};
    refNames = {'Paw withdrawal', 'Flinch / flick'};

    % ---------- import the camera A delivery times AND types ----------
    [fA, pA] = uigetfile({'DeliveryTimes_*.mat', 'Camera A delivery file'}, ...
                         'Select the CAMERA A DeliveryTimes_*.mat for this session');
    if isequal(fA, 0)
        warning('No camera A file selected - skipping this video.');
        return;
    end
    A = load(fullfile(pA, fA));
    deliverySecA = A.deliverySec(:);

    % Stimulus types are required for normalisation. Tolerate an old camera A
    % file (no type field) but say loudly what is lost.
    if isfield(A, 'deliveryType')
        deliveryTypeA = A.deliveryType(:);
        stimNames     = A.stimNames;
    else
        deliveryTypeA = zeros(numel(deliverySecA), 1);
        stimNames     = {'Stim 1', 'Stim 2', 'Stim 3', 'Stim 4', 'UNKNOWN'};
        warning(['This camera A file has no stimulus types (old version of ' ...
                 'score_A). Per-stimulus normalisation will NOT be possible. ' ...
                 'Re-run score_A_stimulus_delivery.m on camera A.']);
    end
    fprintf('Imported %d delivery times from %s\n', numel(deliverySecA), fA);
    for s = 1:4
        fprintf('   %-14s %d delivered\n', stimNames{s}, sum(deliveryTypeA == s));
    end
    if any(deliveryTypeA == 0)
        warning('%d delivery/deliveries are UNKNOWN type and cannot be normalised.', ...
                sum(deliveryTypeA == 0));
    end

    meta = inputdlg( ...
        {'Session number', 'Mouse ID', 'Sex (M/F)', 'Day (1 or 2)', ...
         'Phase (Baseline / Post-treatment)', ...
         'Camera B minus camera A offset (s, 0 if hardware-synced)', ...
         'Observation window per stimulus (s)  [TBC]', ...
         'Guarding minimum hold (s)  [>2 s]', ...
         'Playback speed (1 = real time)'}, ...
        'Session info - do NOT enter treatment', 1, ...
        {char(string(A.sessionNo)), '', '', '', 'Baseline', '0', '30', '2', '1'});
    if isempty(meta), disp('Cancelled.'); return; end
    sessionNo = meta{1}; mouseID = meta{2}; sexID = meta{3};
    dayNo = meta{4};     phase   = meta{5};
    syncOffset = str2double(meta{6});
    obsWindow  = str2double(meta{7});
    guardMin   = str2double(meta{8});
    playSpeed  = str2double(meta{9});
    if isnan(syncOffset), syncOffset = 0;  end
    if isnan(obsWindow) || obsWindow <= 0, obsWindow = 30; end
    if isnan(guardMin) || guardMin < 0,    guardMin  = 2;  end   % sustained lifting > 2 s
    if isnan(playSpeed) || playSpeed <= 0, playSpeed = 1;  end

    vidObj    = VideoReader(videoFile);
    frameRate = vidObj.FrameRate;
    numFrames = floor(vidObj.Duration * frameRate);
    score     = zeros(numFrames, 1);

    % delivery times on camera B's clock - keep types aligned when filtering
    deliverySecB   = deliverySecA + syncOffset;
    deliveryFrameB = round(deliverySecB * frameRate) + 1;
    inRange        = deliveryFrameB >= 1 & deliveryFrameB <= numFrames;
    if ~all(inRange)
        warning(['%d of %d delivery times fall outside camera B. ' ...
                 'Check the sync offset.'], sum(~inRange), numel(inRange));
    end
    deliveryFrameB = deliveryFrameB(inRange);
    deliverySecB   = deliverySecB(inRange);
    deliveryTypeB  = deliveryTypeA(inRange);

    hFig = figure('Name', 'CAMERA B - mouse behaviour', 'NumberTitle', 'off');
    set(hFig, 'WindowKeyPressFcn', @bKeyDown, 'WindowKeyReleaseFcn', @bKeyUp);
    imshow(zeros(480, 640));
    text(320, 172, 'HOLD  a attending   s licking   d biting', ...
        'Color', 'w', 'FontSize', 11, 'HorizontalAlignment', 'center');
    text(320, 198, '      f guarding    g escape / rearing', ...
        'Color', 'w', 'FontSize', 11, 'HorizontalAlignment', 'center');
    text(320, 232, 'TAP   1 withdrawal   2 flinch', ...
        'Color', 'w', 'FontSize', 11, 'HorizontalAlignment', 'center');
    text(320, 262, 'p pause   z undo   q stop', ...
        'Color', 'w', 'FontSize', 11, 'HorizontalAlignment', 'center');
    text(320, 306, sprintf('%d stimulus epochs imported from camera A', ...
        numel(deliveryFrameB)), 'Color', [0.6 1 0.6], 'FontSize', 11, ...
        'HorizontalAlignment', 'center');
    text(320, 330, 'Press ENTER to start', ...
        'Color', 'y', 'FontSize', 14, 'HorizontalAlignment', 'center');
    drawnow;

    disp('Waiting for ENTER...');
    while ~bStarted
        pause(0.1);
        if ~ishandle(hFig), disp('Closed before start.'); return; end
    end

    frame = readFrame(vidObj);
    hImg  = imshow(frame);
    hTtl  = title('');
    drawnow;

    idx = 1;
    frameDelay = 1 / (frameRate * playSpeed);

    while hasFrame(vidObj) && ishandle(hFig) && ~bStopped && idx <= numFrames
        if ~bPaused
            tStart = tic;
            frame  = readFrame(vidObj);
            score(idx) = bBehav;

            if ~isempty(bTapQ)
                bTapQ(bTapQ(:, 1) == 0, 1) = idx;
            end
            if bUndo
                if ~isempty(bTapQ)
                    bTapQ(end, :) = [];
                    disp('Undo: removed last reflex event.');
                end
                bUndo = false;
            end

            % show WHICH stimulus we are inside, not just the running count
            curStim = sum(deliveryFrameB <= idx);
            if curStim >= 1
                curName = typeName(deliveryTypeB(curStim), stimNames);
            else
                curName = 'pre-stimulus';
            end
            set(hImg, 'CData', frame);
            set(hTtl, 'String', sprintf( ...
                'frame %d/%d | %.1fx | stim %d/%d = %s | now: %s | events: %d', ...
                idx, numFrames, playSpeed, curStim, numel(deliveryFrameB), ...
                curName, affNames{bBehav + 1}, size(bTapQ, 1)));
            drawnow;

            idx = idx + 1;
            pause(max(0, frameDelay - toc(tStart)));
        else
            set(hTtl, 'String', sprintf('PAUSED at frame %d/%d  (p to resume)', ...
                idx, numFrames));
            drawnow;
            pause(0.05);
        end
    end

    nUsed    = min(idx - 1, numFrames);
    score    = score(1:nUsed);
    timeAxis = (0:nUsed - 1) / frameRate;
    if ~isempty(bTapQ), bTapQ = bTapQ(bTapQ(:, 1) > 0 & bTapQ(:, 1) <= nUsed, :); end

    [~, vidName] = fileparts(videoFile);

    %% ---------------- per-delivery table ----------------
    obsFrames   = round(obsWindow * frameRate);
    guardFrames = round(guardMin * frameRate);
    nStim = numel(deliveryFrameB);

    % rep = which repeat of THIS stimulus type this delivery is (1, 2, 3, ...)
    rep = zeros(nStim, 1);
    seen = zeros(1, 5);
    for k = 1:nStim
        slot = deliveryTypeB(k);
        if slot < 1 || slot > 4, slot = 5; end
        seen(slot) = seen(slot) + 1;
        rep(k) = seen(slot);
    end

    T = zeros(nStim, 12);      % numeric per-delivery measures
    winSecAll = zeros(nStim, 1);

    for k = 1:nStim
        f0 = deliveryFrameB(k);
        f1 = min(f0 + obsFrames - 1, nUsed);
        if k < nStim, f1 = min(f1, deliveryFrameB(k + 1) - 1); end
        win = f0:f1;
        winSecAll(k) = numel(win) / frameRate;

        ev = [];
        if ~isempty(bTapQ)
            ev = bTapQ(bTapQ(:, 1) >= f0 & bTapQ(:, 1) <= f1, 2);
        end
        withdrawal = double(any(ev == 1));
        flinchN    = sum(ev == 2);

        s = score(win);
        [attC, attD] = episodeStats(s, 1, frameRate, 0);
        [lckC, lckD] = episodeStats(s, 2, frameRate, 0);
        [bitC, bitD] = episodeStats(s, 3, frameRate, 0);
        [grdC, grdD] = episodeStats(s, 4, frameRate, guardFrames); % min hold applied
        [escC, escD] = episodeStats(s, 5, frameRate, 0);

        T(k, :) = [withdrawal, flinchN, attC, attD, lckC, lckD, ...
                   bitC, bitD, grdC, grdD, escC, escD];
    end

    %% ---------------- CSV for Raw_scores (one row per delivery) ----------------
    csvPath = fullfile(outputFolder, ['RawScores_' vidName '.csv']);
    fid = fopen(csvPath, 'w');
    fprintf(fid, ['Session,Day,Mouse ID,Sex,Phase,Treatment,Trial,Stim code,Stimulus,' ...
                  'Rep,Obs window (s),Withdrawal (0/1),Flinch count,' ...
                  'Attending count,Attending dur (s),Licking count,Licking dur (s),' ...
                  'Biting count,Biting dur (s),Guarding count,Guarding dur (s),' ...
                  'Escape/rear count,Escape/rear dur (s),Scorer,Video file,Notes\n']);
    for k = 1:nStim
        fprintf(fid, ['%s,%s,%s,%s,%s,%s,%d,%d,%s,%d,%.2f,' ...
                      '%d,%d,%d,%.2f,%d,%.2f,%d,%.2f,%d,%.2f,%d,%.2f,%s,%s,%s\n'], ...
            sessionNo, dayNo, mouseID, sexID, phase, 'BLIND', ...
            k, deliveryTypeB(k), typeName(deliveryTypeB(k), stimNames), ...
            rep(k), winSecAll(k), ...
            T(k, 1), T(k, 2), T(k, 3), T(k, 4), T(k, 5), T(k, 6), ...
            T(k, 7), T(k, 8), T(k, 9), T(k, 10), T(k, 11), T(k, 12), ...
            '', [vidName '.avi'], sprintf('guard min %.1fs', guardMin));
    end
    fclose(fid);
    fprintf('Per-delivery CSV written: %s\n', csvPath);

    %% ---------------- normalised per-stimulus-type table ----------------
    %  Two denominators, because they answer different questions and a window
    %  can be truncated by an early next stimulus:
    %    per_stim  = total / n_delivered        -> "how much per stimulus given"
    %    pct_time  = total dur / observed time  -> "what fraction of the time"
    normPath = fullfile(outputFolder, ['Normalized_' vidName '.csv']);
    fid = fopen(normPath, 'w');
    fprintf(fid, ['Session,Day,Mouse ID,Sex,Phase,Treatment,Stim code,Stimulus,' ...
                  'n_delivered,obs_time_s,' ...
                  'withdrawal_n,withdrawal_rate,' ...
                  'flinch_n,flinch_per_stim,' ...
                  'attending_n,attending_per_stim,attending_dur_s,' ...
                  'attending_s_per_stim,attending_pct_time,' ...
                  'licking_dur_s,licking_s_per_stim,licking_pct_time,' ...
                  'biting_dur_s,biting_s_per_stim,biting_pct_time,' ...
                  'lickbite_dur_s,lickbite_s_per_stim,lickbite_pct_time,' ...
                  'guarding_n,guarding_per_stim,guarding_dur_s,' ...
                  'guarding_s_per_stim,guarding_pct_time,' ...
                  'escape_n,escape_per_stim,escape_dur_s,' ...
                  'escape_s_per_stim,escape_pct_time\n']);

    normRows = cell(0, 1);
    for s = 1:4
        m = (deliveryTypeB == s);
        n = sum(m);
        if n == 0
            fprintf(fid, '%s,%s,%s,%s,%s,%s,%d,%s,0,0.00', ...
                sessionNo, dayNo, mouseID, sexID, phase, 'BLIND', s, stimNames{s});
            fprintf(fid, repmat(',NA', 1, 28));
            fprintf(fid, '\n');
            warning('%s was never delivered - no normalised values.', stimNames{s});
            continue;
        end
        obsT = sum(winSecAll(m));
        c    = sum(T(m, :), 1);          % column sums for this stimulus type

        wN = c(1);  flN = c(2);
        attN = c(3);  attD = c(4);
        lckD = c(6);  bitD = c(8);  lbD = lckD + bitD;
        grdN = c(9);  grdD = c(10);
        escN = c(11); escD = c(12);

        vals = [n, obsT, ...
                wN,   wN / n, ...
                flN,  flN / n, ...
                attN, attN / n, attD, attD / n, 100 * attD / obsT, ...
                lckD, lckD / n, 100 * lckD / obsT, ...
                bitD, bitD / n, 100 * bitD / obsT, ...
                lbD,  lbD  / n, 100 * lbD  / obsT, ...
                grdN, grdN / n, grdD, grdD / n, 100 * grdD / obsT, ...
                escN, escN / n, escD, escD / n, 100 * escD / obsT];

        fprintf(fid, '%s,%s,%s,%s,%s,%s,%d,%s', ...
            sessionNo, dayNo, mouseID, sexID, phase, 'BLIND', s, stimNames{s});
        fprintf(fid, ',%d,%.2f', vals(1), vals(2));
        fprintf(fid, ',%.0f,%.4f', vals(3), vals(4));
        fprintf(fid, ',%.0f,%.4f', vals(5), vals(6));
        fprintf(fid, ',%.0f,%.4f,%.2f,%.4f,%.2f', vals(7:11));
        fprintf(fid, ',%.2f,%.4f,%.2f', vals(12:14));
        fprintf(fid, ',%.2f,%.4f,%.2f', vals(15:17));
        fprintf(fid, ',%.2f,%.4f,%.2f', vals(18:20));
        fprintf(fid, ',%.0f,%.4f,%.2f,%.4f,%.2f', vals(21:25));
        fprintf(fid, ',%.0f,%.4f,%.2f,%.4f,%.2f', vals(26:30));
        fprintf(fid, '\n');

        normRows{end + 1, 1} = struct('stim', stimNames{s}, 'n', n, ...
            'obsT', obsT, 'withdrawal_rate', wN / n, ...
            'flinch_per_stim', flN / n, 'lickbite_pct', 100 * lbD / obsT, ...
            'guard_pct', 100 * grdD / obsT); %#ok<AGROW>
    end

    nU = sum(deliveryTypeB == 0);
    if nU > 0
        fprintf(fid, '%s,%s,%s,%s,%s,%s,0,UNKNOWN,%d,%.2f', ...
            sessionNo, dayNo, mouseID, sexID, phase, 'BLIND', nU, ...
            sum(winSecAll(deliveryTypeB == 0)));
        fprintf(fid, repmat(',NA', 1, 28));
        fprintf(fid, '\n');
    end
    fclose(fid);
    fprintf('Normalised per-stimulus CSV written: %s\n', normPath);

    fprintf('\n  ---- normalised summary (this session) ----\n');
    fprintf('  %-14s %4s %8s %10s %10s %10s\n', ...
            'stimulus', 'n', 'obs (s)', 'withdr.', 'lick+bite', 'guard');
    for i = 1:numel(normRows)
        r = normRows{i};
        fprintf('  %-14s %4d %8.1f %9.2f  %8.1f%%  %8.1f%%\n', ...
            r.stim, r.n, r.obsT, r.withdrawal_rate, r.lickbite_pct, r.guard_pct);
    end

    %% ---------------- frame-level labels for a future classifier ----------------
    trainPath = fullfile(outputFolder, ['TrainingLabels_' vidName '.csv']);
    fid = fopen(trainPath, 'w');
    fprintf(fid, ['frame,time_s,affective_code,affective_label,withdrawal,flinch,' ...
                  'stim_index,stim_code,stimulus\n']);
    wVec = zeros(nUsed, 1); flVec = zeros(nUsed, 1);
    if ~isempty(bTapQ)
        wVec(bTapQ(bTapQ(:, 2) == 1, 1))  = 1;
        flVec(bTapQ(bTapQ(:, 2) == 2, 1)) = 1;
    end
    for f = 1:nUsed
        si = sum(deliveryFrameB <= f);
        if si >= 1
            sc = deliveryTypeB(si); sn = typeName(sc, stimNames);
        else
            sc = -1; sn = 'pre-stimulus';
        end
        fprintf(fid, '%d,%.4f,%d,%s,%d,%d,%d,%d,%s\n', f, timeAxis(f), score(f), ...
            affNames{score(f) + 1}, wVec(f), flVec(f), si, sc, sn);
    end
    fclose(fid);
    fprintf('Training labels written: %s\n', trainPath);

    %% ---------------- figures ----------------
    affColors  = lines(5);
    stimColors = [0.20 0.45 0.75; 0.85 0.55 0.15; 0.75 0.25 0.25; 0.25 0.55 0.35];

    figTS = figure('Name', 'Behaviour time series', 'NumberTitle', 'off');
    hold on;
    for b = 1:5
        m = (score == b);
        if any(m)
            scatter(timeAxis(m), score(m), 24, affColors(b, :), 'filled', ...
                    'DisplayName', affNames{b + 1});
        end
    end
    for k = 1:nStim
        st = deliveryTypeB(k);
        if st >= 1 && st <= 4, col = stimColors(st, :); else, col = [0 0 0]; end
        xline(timeAxis(deliveryFrameB(k)), '--', ...
              sprintf('%d', st), 'Color', col, 'LineWidth', 1.2, ...
              'HandleVisibility', 'off');
    end
    if ~isempty(bTapQ)
        plot(timeAxis(bTapQ(:, 1)), repmat(5.6, size(bTapQ, 1), 1), 'v', ...
             'MarkerSize', 6, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none', ...
             'HandleVisibility', 'off');
        text(0, 5.9, 'v = reflex event', 'FontSize', 8);
    end
    ylim([0 6.2]); yticks(1:5); yticklabels(affNames(2:6));
    xlabel('Time (s)');
    title('Affective behaviour + reflex events (dashed = delivery, number = stim code)');
    legend('Location', 'eastoutside'); grid on;
    saveas(figTS, fullfile(outputFolder, ['BehaviorTimeSeries_' vidName '.png']));

    figR = figure('Name', 'Raster', 'NumberTitle', 'off');
    for b = 1:5
        subplot(5, 1, b);
        binary = (score == b);
        dsc = diff([0; binary(:); 0]);
        sIdx = find(dsc == 1); eIdx = find(dsc == -1) - 1;
        n = min(numel(sIdx), numel(eIdx));
        sIdx = sIdx(1:n); eIdx = min(eIdx(1:n), nUsed);
        for j = 1:n
            line([timeAxis(sIdx(j)) timeAxis(eIdx(j))], [j j], ...
                 'LineWidth', 2, 'Color', affColors(b, :)); hold on;
        end
        for k = 1:nStim
            st = deliveryTypeB(k);
            if st >= 1 && st <= 4, col = stimColors(st, :); else, col = [0 0 0]; end
            xline(timeAxis(deliveryFrameB(k)), '--', 'Color', col);
        end
        xlabel('Time (s)'); ylabel('Episode'); title(affNames{b + 1}); grid on;
    end
    saveas(figR, fullfile(outputFolder, ['RasterPlot_' vidName '.png']));

    % per-stimulus normalised bar chart - the figure that is actually comparable
    figN = figure('Name', 'Normalised per stimulus', 'NumberTitle', 'off');
    lbl = cell(1, 4); wr = nan(1, 4); lb = nan(1, 4); gd = nan(1, 4); nn = zeros(1, 4);
    for s = 1:4
        m = (deliveryTypeB == s);
        lbl{s} = stimNames{s};
        nn(s) = sum(m);
        if nn(s) == 0, continue; end
        obsT = sum(winSecAll(m)); c = sum(T(m, :), 1);
        wr(s) = c(1) / nn(s);
        lb(s) = 100 * (c(6) + c(8)) / obsT;
        gd(s) = 100 * c(10) / obsT;
    end
    subplot(1, 3, 1); bar(wr); ylim([0 1]);
    set(gca, 'XTickLabel', lbl, 'XTickLabelRotation', 30);
    ylabel('withdrawals / stimulus'); title('Withdrawal rate'); grid on;
    subplot(1, 3, 2); bar(lb);
    set(gca, 'XTickLabel', lbl, 'XTickLabelRotation', 30);
    ylabel('% of observed time'); title('Licking + biting'); grid on;
    subplot(1, 3, 3); bar(gd);
    set(gca, 'XTickLabel', lbl, 'XTickLabelRotation', 30);
    ylabel('% of observed time'); title('Guarding'); grid on;
    sgtitle(sprintf('Session %s  mouse %s  -  n delivered: %s', ...
        sessionNo, mouseID, strjoin(arrayfun(@(x) sprintf('%d', x), nn, ...
        'UniformOutput', false), ' / ')));
    saveas(figN, fullfile(outputFolder, ['NormalizedPerStim_' vidName '.png']));

    %% ---------------- .mat ----------------
    reflexEvents = bTapQ;
    matPath = fullfile(outputFolder, ['ScoringB_' vidName '.mat']);
    save(matPath, 'score', 'timeAxis', 'affNames', 'refNames', 'reflexEvents', ...
         'deliveryFrameB', 'deliverySecB', 'deliveryTypeB', 'stimNames', ...
         'rep', 'T', 'winSecAll', 'syncOffset', 'frameRate', 'nUsed', ...
         'obsWindow', 'guardMin', 'sessionNo', 'mouseID', 'sexID', 'dayNo', ...
         'phase', 'vidName');
    fprintf('MAT written: %s\n', matPath);
end


%% ------------------------------------------------------------------ %%
function nm = typeName(code, stimNames)
    if code >= 1 && code <= 4
        nm = stimNames{code};
    else
        nm = 'UNKNOWN';
    end
end


%% ------------------------------------------------------------------ %%
function [nEp, totSec] = episodeStats(s, code, frameRate, minFrames)
% Episodes shorter than minFrames are discarded (used for the guarding threshold).
    binary = (s(:) == code);
    d = diff([0; binary; 0]);
    st = find(d == 1); en = find(d == -1) - 1;
    keep = (en - st + 1) >= max(minFrames, 1);
    st = st(keep); en = en(keep);
    nEp = numel(st);
    totSec = sum(en - st + 1) / frameRate;
end


%% ---------------------------- key callbacks ------------------------ %%
function bKeyDown(~, event)
    global bBehav bPaused bStopped bStarted bTapQ bUndo;
    switch lower(event.Key)
        case 'return'
            if ~bStarted, bStarted = true; disp('Playing...'); end
        case 'a', bBehav = 1;
        case 's', bBehav = 2;
        case 'd', bBehav = 3;
        case 'f', bBehav = 4;
        case 'g', bBehav = 5;
        case '1', bTapQ(end + 1, :) = [0 1];
        case '2', bTapQ(end + 1, :) = [0 2];
        case 'z', bUndo = true;
        case 'p', bPaused = ~bPaused;
        case 'q', bStopped = true; disp('Stopped.');
    end
end


function bKeyUp(~, event)
    global bBehav;
    switch lower(event.Key)
        case {'a', 's', 'd', 'f', 'g'}
            bBehav = 0;
    end
end
