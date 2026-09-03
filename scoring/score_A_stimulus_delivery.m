function score_A_stimulus_delivery(folderPath)
% SCORE_A_STIMULUS_DELIVERY  Camera A — mark when each stimulus is delivered,
%                            AND which stimulus it was.
%
%   *** SUPERSEDED by score_AB_dual_view.m ***
%   Use score_AB_dual_view.m instead: it shows both cameras at once and scores
%   the deliveries and the six behaviours in a single pass. Two reasons this
%   file is kept rather than deleted:
%     1. it is the reference for the delivery-only pass, if you ever want to
%        re-mark deliveries without re-scoring behaviour;
%     2. it documents the original two-pass workflow.
%
%   KNOWN LIMITATION: this script buffers every frame into memory so that it
%   can step backwards while paused. A 50,000-frame DV session needs ~52 GB
%   and will not open. score_AB_dual_view.m streams instead.
%
%   Camera A watches the experimenter's hand / applicator. Two things are scored
%   here: the moment of contact, and the stimulus type. The type matters because
%   the number of deliveries per stimulus differs between mice (e.g. 10 pin
%   pricks for one mouse, 11 for another), so every downstream measure has to be
%   expressed per stimulus delivered. Without the type there is no denominator.
%
%   These timestamps become the reference clock for camera B, so this pass should
%   be frame-accurate: pause and step frame by frame.
%
%   ENTER  start                      p            pause / resume
%   1..4   mark delivery of stim 1-4  left / right  step 1 frame (while paused)
%   d      mark UNKNOWN stimulus      z            undo last mark
%   q      stop
%
%   Outputs into <folderPath>\output\ :
%     DeliveryTimes_<vid>.mat    deliveryFrames, deliverySec, deliveryType,
%                                stimNames, nPerStim, frameRate, sessionNo
%     DeliveryTimes_<vid>.csv    Session, Delivery, Stim code, Stimulus, Frame, Time_s
%     DeliveryCounts_<vid>.csv   Stim code, Stimulus, n_delivered
%                                <- this is the DENOMINATOR for normalisation
%
%   Feed the .mat into score_B_mouse_behavior.m.
%
%   Hansol Lim - HEAL mini1p / SBI-553

    clc; close all;

    if nargin < 1
        folderPath = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos\cameraA';
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
        [indx, tf] = listdlg('PromptString', 'CAMERA A - select video:', ...
                             'SelectionMode', 'single', 'ListSize', [420 320], ...
                             'ListString', videoNames);
        if ~tf, disp('Nothing selected. Exiting.'); break; end
        v = videoFiles(indx);
        fprintf('\n===== Camera A: %s =====\n', v.name);
        markDeliveries(fullfile(v.folder, v.name), outputFolder);
        if strcmpi(questdlg('Another video?', 'Continue', 'Yes', 'No', 'Yes'), 'No')
            keepGoing = false;
        end
    end
    disp('Camera A pass complete.');
end


%% ------------------------------------------------------------------ %%
function markDeliveries(videoFile, outputFolder)

    global aStarted aPaused aStopped aMarkType aUndo aStep;
    aStarted = false; aPaused = false; aStopped = false;
    aMarkType = -1; aUndo = false; aStep = 0;      % -1 = nothing pending

    % Stimulus names live here so there is ONE place to change them.
    % Keep the order identical to the randomisation sheet.
    defNames = {'Light touch', 'Mild touch', 'Heat', 'Pin prick'};

    meta = inputdlg({'Session number (from randomisation sheet)', ...
                     'Stimulus 1 name  (key 1)', ...
                     'Stimulus 2 name  (key 2)', ...
                     'Stimulus 3 name  (key 3)', ...
                     'Stimulus 4 name  (key 4)', ...
                     'Playback speed (1 = real time)'}, ...
                    'Camera A', 1, ...
                    [{''}, defNames, {'1'}]);
    if isempty(meta), disp('Cancelled.'); return; end
    sessionNo = meta{1};
    stimNames = meta(2:5);
    playSpeed = str2double(meta{6});
    if isnan(playSpeed) || playSpeed <= 0, playSpeed = 1; end
    for s = 1:4
        if isempty(strtrim(stimNames{s})), stimNames{s} = defNames{s}; end
    end
    stimNames{5} = 'UNKNOWN';        % code 0 -> slot 5, see typeName()

    vidObj    = VideoReader(videoFile);
    frameRate = vidObj.FrameRate;
    numFrames = floor(vidObj.Duration * frameRate);

    % read every frame once so we can step backwards while paused
    fprintf('Buffering %d frames...\n', numFrames);
    frames = cell(numFrames, 1);
    k = 0;
    while hasFrame(vidObj) && k < numFrames
        k = k + 1;
        frames{k} = readFrame(vidObj);
    end
    numFrames = k;
    fprintf('Buffered %d frames at %.2f fps.\n', numFrames, frameRate);

    hFig = figure('Name', 'CAMERA A - stimulus delivery', 'NumberTitle', 'off');
    set(hFig, 'WindowKeyPressFcn', @aKeyDown);
    imshow(frames{1});
    text(size(frames{1}, 2) / 2, 30, ...
        'ENTER start    p pause    \leftarrow \rightarrow step    z undo    q stop', ...
        'Color', 'y', 'FontSize', 11, 'HorizontalAlignment', 'center');
    text(size(frames{1}, 2) / 2, 56, ...
        sprintf('MARK:  1 %s    2 %s    3 %s    4 %s    d unknown', ...
                stimNames{1}, stimNames{2}, stimNames{3}, stimNames{4}), ...
        'Color', [0.6 1 0.6], 'FontSize', 10, 'HorizontalAlignment', 'center');
    drawnow;

    disp('Waiting for ENTER...');
    while ~aStarted
        pause(0.1);
        if ~ishandle(hFig), disp('Closed before start.'); return; end
    end

    hImg = imshow(frames{1});
    hTtl = title('');
    deliveryFrames = [];
    deliveryType   = [];        % 1..4, or 0 for UNKNOWN
    idx = 1;
    frameDelay = 1 / (frameRate * playSpeed);

    while ishandle(hFig) && ~aStopped && idx <= numFrames
        tStart = tic;

        if aMarkType >= 0
            deliveryFrames(end + 1) = idx;          %#ok<AGROW>
            deliveryType(end + 1)   = aMarkType;    %#ok<AGROW>
            fprintf('  delivery %d = %s  at frame %d (%.3f s)\n', ...
                numel(deliveryFrames), typeName(aMarkType, stimNames), ...
                idx, (idx - 1) / frameRate);
            aMarkType = -1;
        end
        if aUndo
            if ~isempty(deliveryFrames)
                fprintf('  undo: removed %s at frame %d\n', ...
                    typeName(deliveryType(end), stimNames), deliveryFrames(end));
                deliveryFrames(end) = [];
                deliveryType(end)   = [];
            end
            aUndo = false;
        end

        set(hImg, 'CData', frames{idx});
        set(hTtl, 'String', sprintf('%s  |  frame %d/%d  (%.3f s)  |  %s', ...
            ternary(aPaused, 'PAUSED', 'PLAYING'), idx, numFrames, ...
            (idx - 1) / frameRate, countString(deliveryType, stimNames)));
        drawnow;

        if aPaused
            if aStep ~= 0
                idx = min(max(idx + aStep, 1), numFrames);
                aStep = 0;
            end
            pause(0.03);
        else
            idx = idx + 1;
            pause(max(0, frameDelay - toc(tStart)));
        end
    end

    if isempty(deliveryFrames)
        warning('No deliveries marked. Nothing saved for %s.', videoFile);
        return;
    end

    % keep marks in time order, carrying the type along with them
    [deliveryFrames, ord] = sort(deliveryFrames(:));
    deliveryType   = deliveryType(:);
    deliveryType   = deliveryType(ord);
    deliverySec    = (deliveryFrames - 1) / frameRate;
    [~, vidName]   = fileparts(videoFile);

    % ---- the denominator: how many of each stimulus were actually given ----
    nPerStim = zeros(1, 4);
    for s = 1:4, nPerStim(s) = sum(deliveryType == s); end
    nUnknown = sum(deliveryType == 0);

    fprintf('\n  deliveries per stimulus (this is the normalisation denominator):\n');
    for s = 1:4
        fprintf('    %-14s %d\n', stimNames{s}, nPerStim(s));
    end
    if nUnknown > 0
        fprintf('    %-14s %d   <-- FIX THESE before scoring camera B\n', ...
                'UNKNOWN', nUnknown);
        warning(['%d delivery/deliveries have no stimulus type. They cannot be ' ...
                 'normalised. Re-score them or drop them explicitly.'], nUnknown);
    end
    if any(nPerStim == 0)
        warning('Stimulus type(s) with zero deliveries: %s', ...
                strjoin(stimNames(nPerStim == 0), ', '));
    end

    csvPath = fullfile(outputFolder, ['DeliveryTimes_' vidName '.csv']);
    fid = fopen(csvPath, 'w');
    fprintf(fid, 'Session,Delivery,Stim code,Stimulus,Frame,Time_s\n');
    for j = 1:numel(deliveryFrames)
        fprintf(fid, '%s,%d,%d,%s,%d,%.4f\n', sessionNo, j, deliveryType(j), ...
                typeName(deliveryType(j), stimNames), ...
                deliveryFrames(j), deliverySec(j));
    end
    fclose(fid);

    cntPath = fullfile(outputFolder, ['DeliveryCounts_' vidName '.csv']);
    fid = fopen(cntPath, 'w');
    fprintf(fid, 'Session,Stim code,Stimulus,n_delivered\n');
    for s = 1:4
        fprintf(fid, '%s,%d,%s,%d\n', sessionNo, s, stimNames{s}, nPerStim(s));
    end
    if nUnknown > 0
        fprintf(fid, '%s,0,UNKNOWN,%d\n', sessionNo, nUnknown);
    end
    fclose(fid);

    matPath = fullfile(outputFolder, ['DeliveryTimes_' vidName '.mat']);
    save(matPath, 'deliveryFrames', 'deliverySec', 'deliveryType', 'stimNames', ...
         'nPerStim', 'nUnknown', 'frameRate', 'numFrames', 'sessionNo', 'vidName');

    fprintf('\nSaved %d delivery times:\n   %s\n   %s\n   %s\n', ...
        numel(deliveryFrames), csvPath, cntPath, matPath);
end


%% ------------------------------------------------------------------ %%
function nm = typeName(code, stimNames)
    if code >= 1 && code <= 4
        nm = stimNames{code};
    else
        nm = stimNames{5};              % UNKNOWN
    end
end


function str = countString(deliveryType, stimNames)
    parts = cell(1, 4);
    for s = 1:4
        parts{s} = sprintf('%s %d', stimNames{s}(1:min(4, end)), ...
                           sum(deliveryType == s));
    end
    str = strjoin(parts, '  ');
    nU = sum(deliveryType == 0);
    if nU > 0, str = sprintf('%s  UNK %d', str, nU); end
end


function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end


%% ---------------------------- key callback ------------------------- %%
function aKeyDown(~, event)
    global aStarted aPaused aStopped aMarkType aUndo aStep;
    switch lower(event.Key)
        case 'return'
            if ~aStarted, aStarted = true; disp('Playing...'); end
        case '1',        aMarkType = 1;
        case '2',        aMarkType = 2;
        case '3',        aMarkType = 3;
        case '4',        aMarkType = 4;
        case 'd',        aMarkType = 0;      % unknown / could not tell
        case 'p',        aPaused = ~aPaused;
        case 'leftarrow',  aStep = -1;
        case 'rightarrow', aStep = 1;
        case 'z',        aUndo = true;
        case 'q',        aStopped = true; disp('Stopped.');
    end
end
