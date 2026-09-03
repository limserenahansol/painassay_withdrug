function manual_scoring_pain_assay(folderPath)
% MANUAL_SCORING_PAIN_ASSAY  Blind manual scoring for the mini1p / SBI-553 pain assay.
%
%   Scores REFLEXIVE and AFFECTIVE-MOTIVATIONAL behaviours SEPARATELY, per the
%   Corder/Biafra criteria. There is deliberately no combined "pain score".
%
%   HELD keys  (duration + episode count)   TAP keys (event count)
%     a  Paw attending                        1  Paw withdrawal
%     s  Licking / biting                     2  Flinch / flick
%     d  Sustained lifting / guarding
%     f  Escape / rearing
%
%   m      mark the onset of the next stimulus (press once before each of the 4)
%   return start playback      space pause/resume      q stop
%   z      undo the last tap event or stimulus mark
%
%   Outputs, per video, into <folderPath>\output\ :
%     ManualScoringPain_<vid>.mat   frame-wise score, events, stimulus onsets
%     RawScores_<vid>.csv           one row per stimulus, matching the
%                                   Raw_scores sheet of Behavioural_scoring_book.xlsx
%     BehaviorTimeSeries_<vid>.png  and  RasterPlot_<vid>.png
%
%   Scoring is BLIND: treatment is never displayed or read. Join on Session /
%   Mouse ID afterwards using Stimulus_randomisation_mini1p.xlsx.
%
%   Hansol Lim - HEAL mini1p / SBI-553

    clc; close all;

    if nargin < 1
        folderPath = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos';
    end
    baseFolder   = folderPath;
    outputFolder = fullfile(baseFolder, 'output');

    if ~exist(baseFolder, 'dir')
        error('Base folder does not exist: %s', baseFolder);
    end
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end

    % Check write access
    testFile = fullfile(outputFolder, 'test_write.txt');
    fid = fopen(testFile, 'w');
    if fid == -1
        error('Cannot write to %s. Check permissions or drive status.', outputFolder);
    else
        fclose(fid); delete(testFile);
    end

    % Find video files
    videoFiles = [dir(fullfile(baseFolder, '*.mp4')); ...
                  dir(fullfile(baseFolder, '*.avi')); ...
                  dir(fullfile(baseFolder, '*.mov'))];
    if isempty(videoFiles)
        error('No video files found in %s', baseFolder);
    end
    videoNames = {videoFiles.name};

    processAnother = true;
    while processAnother
        [indx, tf] = listdlg('PromptString', 'Select a video to score:', ...
                             'SelectionMode', 'single', ...
                             'ListSize', [420 320], ...
                             'ListString', videoNames);
        if ~tf
            disp('No video selected. Exiting...');
            break;
        end
        selectedVideo = videoFiles(indx);
        videoPath = fullfile(selectedVideo.folder, selectedVideo.name);
        fprintf('\n========== Scoring: %s ==========\n', selectedVideo.name);

        scoreSinglePainVideo(videoPath, outputFolder);
        fprintf('Done with video: %s\n', selectedVideo.name);

        choice = questdlg('Score another video?', 'Continue', 'Yes', 'No', 'Yes');
        if strcmpi(choice, 'No')
            processAnother = false;
        end
    end
    disp('Pain assay scoring completed.');
end


%% ------------------------------------------------------------------ %%
function scoreSinglePainVideo(videoFile, outputFolder)

    global currentBehavior isPaused isStopped videoStarted ...
           tapQueue stimQueue undoRequest;

    currentBehavior = 0;     % 0 none, 1 attending, 2 lick/bite, 3 guarding, 4 escape/rear
    isPaused    = false;
    isStopped   = false;
    videoStarted = false;
    tapQueue    = [];        % frames at which a reflex event was tapped: [frame code]
    stimQueue   = [];        % frames at which a stimulus was marked
    undoRequest = false;

    affectiveNames = {'None', 'Paw attending', 'Licking / biting', ...
                      'Sustained lifting / guarding', 'Escape / rearing'};
    reflexNames    = {'Paw withdrawal', 'Flinch / flick'};

    % ---- ask for the session metadata (blind: no treatment field) ----
    meta = inputdlg( ...
        {'Session number (from randomisation sheet)', 'Mouse ID', 'Sex (M/F)', ...
         'Day (1 or 2)', 'Phase (Baseline / Post-treatment)', ...
         'Observation window per stimulus (s)  [TBC - Corder/Biafra]', ...
         'Playback speed (1 = real time)'}, ...
        'Session info - do NOT enter treatment', 1, ...
        {'', '', '', '', 'Baseline', '30', '1'});
    if isempty(meta)
        disp('Cancelled.'); return;
    end
    sessionNo = meta{1};  mouseID = meta{2};  sexID = meta{3};
    dayNo     = meta{4};  phase   = meta{5};
    obsWindow = str2double(meta{6});
    playbackSpeed = str2double(meta{7});
    if isnan(playbackSpeed) || playbackSpeed <= 0, playbackSpeed = 1; end
    if isnan(obsWindow) || obsWindow <= 0, obsWindow = 30; end

    % ---- video ----
    vidObj     = VideoReader(videoFile);
    frameRate  = vidObj.FrameRate;
    numFrames  = floor(vidObj.Duration * frameRate);
    score      = zeros(numFrames, 1);   % held affective behaviour per frame

    hFig = figure('Name', 'Pain assay - manual scoring', 'NumberTitle', 'off');
    set(hFig, 'WindowKeyPressFcn', @keyDownFcn, 'WindowKeyReleaseFcn', @keyUpFcn);
    imshow(zeros(480, 640));
    text(320, 190, 'HOLD  a attending   s lick/bite   d guarding   f escape/rear', ...
        'Color', 'w', 'FontSize', 11, 'HorizontalAlignment', 'center');
    text(320, 220, 'TAP   1 withdrawal   2 flinch        m mark stimulus onset', ...
        'Color', 'w', 'FontSize', 11, 'HorizontalAlignment', 'center');
    text(320, 250, 'space pause    z undo    q stop', ...
        'Color', 'w', 'FontSize', 11, 'HorizontalAlignment', 'center');
    text(320, 300, 'Press ENTER to start', ...
        'Color', 'y', 'FontSize', 14, 'HorizontalAlignment', 'center');
    drawnow;

    disp('Waiting for ENTER to start playback...');
    while ~videoStarted
        pause(0.1);
        if ~ishandle(hFig)
            disp('Figure closed before starting. Skipping...');
            return;
        end
    end

    frameIdx   = 1;
    frameDelay = 1 / (frameRate * playbackSpeed);

    frame = readFrame(vidObj);
    hImg  = imshow(frame);
    hTtl  = title('');
    drawnow;

    % ---- main loop ----
    while hasFrame(vidObj) && ishandle(hFig) && ~isStopped && frameIdx <= numFrames
        if ~isPaused
            tStart = tic;
            frame  = readFrame(vidObj);
            score(frameIdx) = currentBehavior;

            % stamp every pending tap / stimulus mark with this frame
            % (a fast double-tap can leave more than one waiting)
            if ~isempty(tapQueue)
                tapQueue(tapQueue(:, 1) == 0, 1) = frameIdx;
            end
            if ~isempty(stimQueue)
                stimQueue(stimQueue == 0) = frameIdx;
            end
            if undoRequest
                [tapQueue, stimQueue] = doUndo(tapQueue, stimQueue);
                undoRequest = false;
            end

            set(hImg, 'CData', frame);
            nStim = numel(stimQueue);
            set(hTtl, 'String', sprintf( ...
                'Frame %d/%d  |  %.1fx  |  stimulus %d/4  |  now: %s  |  events: %d', ...
                frameIdx, numFrames, playbackSpeed, nStim, ...
                affectiveNames{currentBehavior + 1}, size(tapQueue, 1)));
            drawnow;

            frameIdx = frameIdx + 1;
            elapsed  = toc(tStart);
            pause(max(0, frameDelay - elapsed));
        else
            set(hTtl, 'String', sprintf('PAUSED at frame %d/%d  (space to resume)', ...
                frameIdx, numFrames));
            drawnow;
            pause(0.05);
        end
    end

    score = score(1:min(frameIdx - 1, numFrames));
    nUsed = numel(score);
    timeAxis = (0:nUsed - 1) / frameRate;

    % drop any unresolved (frame 0) entries
    if ~isempty(tapQueue),  tapQueue  = tapQueue(tapQueue(:, 1) > 0, :);  end
    if ~isempty(stimQueue), stimQueue = stimQueue(stimQueue > 0);         end

    [~, vidName, ~] = fileparts(videoFile);

    % ---- warn if the stimulus count is not 4 ----
    if numel(stimQueue) ~= 4
        warning(['%d stimulus onsets marked, expected 4. ' ...
                 'Per-stimulus rows will still be written for what was marked.'], ...
                 numel(stimQueue));
    end

    %% ---------------- per-stimulus summary ----------------
    obsFrames = round(obsWindow * frameRate);
    nStim = numel(stimQueue);
    rows  = cell(nStim, 1);

    for k = 1:nStim
        f0 = stimQueue(k);
        f1 = min(f0 + obsFrames - 1, nUsed);
        if k < nStim
            f1 = min(f1, stimQueue(k + 1) - 1);   % never bleed into the next stimulus
        end
        win = f0:f1;

        ev = [];
        if ~isempty(tapQueue)
            ev = tapQueue(tapQueue(:, 1) >= f0 & tapQueue(:, 1) <= f1, 2);
        end
        withdrawal = double(any(ev == 1));
        flinchN    = sum(ev == 2);

        s = score(win);
        [attC, attD] = episodeStats(s, 1, frameRate);
        [lckC, lckD] = episodeStats(s, 2, frameRate);
        [~,    grdD] = episodeStats(s, 3, frameRate);
        [escC, escD] = episodeStats(s, 4, frameRate);

        rows{k} = { str2double(sessionNo), str2double(dayNo), mouseID, sexID, phase, ...
                    'BLIND', k, '', withdrawal, flinchN, ...
                    attC, attD, lckC, lckD, grdD, escC, escD, ...
                    '', [vidName '.mp4'], sprintf('window %.0f s', obsWindow) };
    end

    %% ---------------- CSV in Raw_scores column order ----------------
    csvPath = fullfile(outputFolder, ['RawScores_' vidName '.csv']);
    fid = fopen(csvPath, 'w');
    fprintf(fid, ['Session,Day,Mouse ID,Sex,Phase,Treatment,Stim position,Stimulus,' ...
                  'Withdrawal (0/1),Flinch count,Attending count,Attending dur (s),' ...
                  'Lick/bite count,Lick/bite dur (s),Guarding dur (s),' ...
                  'Escape/rear count,Escape/rear dur (s),Scorer,Video file,Notes\n']);
    for k = 1:nStim
        r = rows{k};
        fprintf(fid, '%g,%g,%s,%s,%s,%s,%d,%s,%d,%d,%d,%.2f,%d,%.2f,%.2f,%d,%.2f,%s,%s,%s\n', ...
            r{1}, r{2}, r{3}, r{4}, r{5}, r{6}, r{7}, r{8}, ...
            r{9}, r{10}, r{11}, r{12}, r{13}, r{14}, r{15}, r{16}, r{17}, ...
            r{18}, r{19}, r{20});
    end
    fclose(fid);
    fprintf('CSV written: %s\n', csvPath);
    fprintf('   Stimulus column is left blank - fill from the randomisation sheet.\n');

    %% ---------------- figures ----------------
    affColors = lines(4);

    figTS = figure('Name', 'Behaviour time series', 'NumberTitle', 'off');
    hold on;
    for b = 1:4
        idx = (score == b);
        if any(idx)
            scatter(timeAxis(idx), score(idx), 24, affColors(b, :), 'filled', ...
                    'DisplayName', affectiveNames{b + 1});
        end
    end
    for k = 1:nStim
        xline(timeAxis(stimQueue(k)), 'k--', sprintf('S%d', k), 'LineWidth', 1.2);
    end
    if ~isempty(tapQueue)
        for e = 1:size(tapQueue, 1)
            plot(timeAxis(tapQueue(e, 1)), 4.6, 'v', 'MarkerSize', 6, ...
                 'MarkerFaceColor', [0 0 0], 'MarkerEdgeColor', 'none', ...
                 'HandleVisibility', 'off');
        end
        text(0, 4.9, 'v = reflex event (1 withdrawal, 2 flinch)', 'FontSize', 8);
    end
    ylim([0 5.2]); yticks(1:4); yticklabels(affectiveNames(2:5));
    xlabel('Time (s)'); title('Affective behaviour + reflex events');
    legend('Location', 'eastoutside'); grid on;
    saveas(figTS, fullfile(outputFolder, ['BehaviorTimeSeries_' vidName '.png']));

    figR = figure('Name', 'Raster', 'NumberTitle', 'off');
    for b = 1:4
        subplot(4, 1, b);
        binary = (score == b);
        dsc    = diff([0; binary(:); 0]);
        sIdx   = find(dsc == 1);
        eIdx   = find(dsc == -1) - 1;
        n      = min(numel(sIdx), numel(eIdx));
        sIdx = sIdx(1:n); eIdx = min(eIdx(1:n), nUsed);
        for j = 1:n
            line([timeAxis(sIdx(j)) timeAxis(eIdx(j))], [j j], ...
                 'LineWidth', 2, 'Color', affColors(b, :));
            hold on;
        end
        for k = 1:nStim
            xline(timeAxis(stimQueue(k)), 'k--');
        end
        xlabel('Time (s)'); ylabel('Episode');
        title(affectiveNames{b + 1}); grid on;
    end
    saveas(figR, fullfile(outputFolder, ['RasterPlot_' vidName '.png']));

    %% ---------------- .mat ----------------
    stimulusOnsetFrames = stimQueue(:);
    reflexEvents        = tapQueue;       % [frame code], code 1 withdrawal, 2 flinch
    matPath = fullfile(outputFolder, ['ManualScoringPain_' vidName '.mat']);
    save(matPath, 'score', 'timeAxis', 'affectiveNames', 'reflexNames', ...
         'reflexEvents', 'stimulusOnsetFrames', 'frameRate', 'nUsed', ...
         'obsWindow', 'sessionNo', 'mouseID', 'sexID', 'dayNo', 'phase', 'rows');
    fprintf('MAT written: %s\n', matPath);
end


%% ------------------------------------------------------------------ %%
function [nEpisodes, totalSec] = episodeStats(s, code, frameRate)
    binary = (s(:) == code);
    d = diff([0; binary; 0]);
    nEpisodes = sum(d == 1);
    totalSec  = sum(binary) / frameRate;
end


function [tapQueue, stimQueue] = doUndo(tapQueue, stimQueue)
    lastTap  = 0; lastStim = 0;
    if ~isempty(tapQueue),  lastTap  = tapQueue(end, 1);  end
    if ~isempty(stimQueue), lastStim = stimQueue(end);    end
    if lastTap == 0 && lastStim == 0
        return;
    elseif lastTap >= lastStim
        tapQueue(end, :) = [];
        disp('Undo: removed last reflex event.');
    else
        stimQueue(end) = [];
        disp('Undo: removed last stimulus mark.');
    end
end


%% ---------------------------- key callbacks ------------------------ %%
function keyDownFcn(~, event)
    global currentBehavior isPaused isStopped videoStarted ...
           tapQueue stimQueue undoRequest;
    switch lower(event.Key)
        case 'return'
            if ~videoStarted
                videoStarted = true;
                disp('Starting playback...');
            end
        % held affective behaviours
        case 'a', currentBehavior = 1;
        case 's', currentBehavior = 2;
        case 'd', currentBehavior = 3;
        case 'f', currentBehavior = 4;
        % tapped reflex events - frame is stamped on the next processed frame
        case '1', tapQueue(end + 1, :) = [0 1];
        case '2', tapQueue(end + 1, :) = [0 2];
        % stimulus onset
        case 'm', stimQueue(end + 1) = 0;
        case 'z', undoRequest = true;
        case 'space', isPaused = ~isPaused;
        case 'q', isStopped = true; disp('Playback stopped by user.');
    end
end


function keyUpFcn(~, event)
    global currentBehavior;
    switch lower(event.Key)
        case {'a', 's', 'd', 'f'}
            currentBehavior = 0;
    end
end
