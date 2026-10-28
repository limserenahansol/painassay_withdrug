%% recover_from_open_windows.m - rescue what is left in scoring windows that were not saved (2026-10-21)
% Run in the scoring MATLAB AFTER the current session is finished (command window free).
% What can be recovered: only the HELD behaviours (attending, lick/bite, guarding, escape/rearing), from the
% whole-session bar at the bottom of each open window, at ~1 s resolution (1000 columns for ~1000 s).
% What cannot: withdrawal / flinch taps, movement and grooming - they were only in memory and were erased
% when the next session started.
% Writes scoring_mini1p\20261021_day2\RECOVERED_held_<frames>frames.mat (score per frame, approximate).
out = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring_mini1p\20261021_day2';
affRGB = [0.20 0.45 0.75; 0.85 0.35 0.15; 0.55 0.30 0.70; 0.20 0.65 0.35];
names = {'attending', 'lick/bite', 'guarding', 'escape/rearing'};
figs = findall(0, 'Type', 'figure', '-regexp', 'Name', '^DUAL VIEW');
fprintf('%d scoring window(s) open\n', numel(figs));
for k = 1:numel(figs)
    imgs = findall(figs(k), 'Type', 'image');
    seek = [];
    for i = 1:numel(imgs)
        c = imgs(i).CData;
        if ndims(c) == 3 && size(c, 2) == 1000 && size(c, 1) < 60, seek = c; end
    end
    t = findall(figs(k), 'Type', 'text');
    N = NaN;
    for i = 1:numel(t)
        s = string(t(i).String);
        tok = regexp(s, 'frame \d+/(\d+)', 'tokens', 'once');
        if ~isempty(tok), N = str2double(tok{1}); end
    end
    if isempty(seek) || isnan(N), fprintf('window %d: no session bar found\n', k); continue; end
    H = size(seek, 1); W = size(seek, 2);
    code = zeros(1, W);
    for c = 1:W
        px = squeeze(seek(floor((H - 8) / 2), c, :))';
        d = sum((affRGB - px) .^ 2, 2);
        [m, j] = min(d);
        if m < 1e-3, code(c) = j; end
    end
    edges = round(linspace(1, N, W + 1));
    score = zeros(N, 1);
    for c = 1:W
        score(edges(c):max(edges(c), edges(c + 1) - 1)) = code(c);
    end
    f = fullfile(out, sprintf('RECOVERED_held_%dframes.mat', N));
    note = 'approximate: held behaviours from the session bar (1000 columns); no taps, movement or grooming';
    save(f, 'score', 'N', 'names', 'note');
    fprintf('window %d: %d frames, recovered seconds - %s -> %s\n', k, N, ...
        strjoin(arrayfun(@(j) sprintf('%s %.0f', names{j}, sum(score == j) / 30), 1:4, 'UniformOutput', false), ', '), f);
end
