function cell_curation_gui(data_mat, mode)
%CELL_CURATION_GUI  Keep or delete every EXTRACT component by looking at it: a 1-2 min movie
%   of the whole field next to the max / mean projections, the selected cell zoomed, its
%   trace, and its automatic QC.
%
%   cell_curation_gui(data_mat)      data_mat from prepare_curation_data
%   cell_curation_gui(data_mat, 'selftest')   builds the window invisibly, exercises it, saves
%                                    to a temporary copy of the csv (for testing only)
%
%   LEFT   the whole field; background menu: movie dF/F, movie raw, max dF/F, mean, max
%          (background removed), Cn x PNR. Every component outlined + numbered:
%          green = keep, red = delete, dashed = not reviewed yet (colour = current keep value)
%   RIGHT  zoom on the selected component (same background), its whole trace with the
%          movie window shaded and a cursor that follows the movie, and its automatic QC
%   KEYS   k keep    d delete    -> / n next    <- / p previous    u next not reviewed
%          space play / pause    + / - speed    s save    (click a cell in the left image)
%   SAVE   writes keep (1/0) and reviewed (1/0) into the review csv; the csv before the
%          first save of the session is kept as <name>_before_gui_<time>.csv

if nargin < 2, mode = ''; end
selftest = strcmpi(mode, 'selftest');
D = load(data_mat);
csv = D.review_csv;
if selftest
    tmp = fullfile(tempdir, 'cell_curation_selftest.csv');
    copyfile(csv, tmp);
    csv = tmp;
end
R = readtable(csv, 'TextType', 'char');
C = height(R);
if ~ismember('reviewed', R.Properties.VariableNames), R.reviewed = zeros(C, 1); end
st = struct('cur', 1, 'playing', false, 'frame', 1, 'speed', 1, 'bg', 'movie dF/F', ...
            'dirty', false, 'backed_up', false);
nF = size(D.movie_dff, 3);
[Hs, Ws, ~] = size(D.movie_dff);
zoomhalf = max(30, round(4 * D.radius * D.sc));

fig = figure('Name', sprintf('Cell curation - %s', csv), 'NumberTitle', 'off', 'Color', 'w', ...
             'Position', [40 40 1650 900], 'KeyPressFcn', @onkey, 'CloseRequestFcn', @onclose, ...
             'Visible', tern(selftest, 'off', 'on'));
axF = axes('Parent', fig, 'Position', [0.01 0.12 0.60 0.83]);
hImg = imshow(zeros(Hs, Ws, 'uint8'), [0 255], 'Parent', axF);
hold(axF, 'on');
hOut = cell(C, 1); hNum = gobjects(C, 1);
for c = 1:C
    hOut{c} = gobjects(numel(D.outline{c}), 1);
    for b = 1:numel(D.outline{c})
        hOut{c}(b) = plot(axF, D.outline{c}{b}(:, 2), D.outline{c}{b}(:, 1), '-', 'LineWidth', 1.1, ...
                          'HitTest', 'off');
    end
    hNum(c) = text(axF, D.cen(c, 2) + 6, D.cen(c, 1) - 6, sprintf('%d', c), 'FontSize', 9, ...
                   'FontWeight', 'bold', 'HitTest', 'off');
end
set(hImg, 'ButtonDownFcn', @onclick);
hTitle = title(axF, '', 'FontWeight', 'normal');
axZ = axes('Parent', fig, 'Position', [0.63 0.52 0.17 0.42]);
hZ = imshow(zeros(2 * zoomhalf + 1, 'uint8'), [0 255], 'Parent', axZ);
hold(axZ, 'on');
hZo = plot(axZ, nan, nan, '-', 'LineWidth', 1.5);
axT = axes('Parent', fig, 'Position', [0.64 0.13 0.34 0.30]);
hInfo = uicontrol(fig, 'Style', 'text', 'Units', 'normalized', 'Position', [0.81 0.52 0.18 0.42], ...
                  'HorizontalAlignment', 'left', 'BackgroundColor', 'w', 'FontSize', 10, 'Enable', 'inactive');
hBg = uicontrol(fig, 'Style', 'popupmenu', 'Units', 'normalized', 'Position', [0.01 0.955 0.16 0.035], ...
    'String', {'movie dF/F', 'movie raw', 'max dF/F', 'mean', 'max (background removed)', 'Cn x PNR'}, ...
    'Callback', @onbg, 'FontSize', 10);
btn = {'Keep (k)', @(~, ~) decide(1); 'Delete (d)', @(~, ~) decide(0); '< Prev (p)', @(~, ~) go(-1); ...
       'Next (n) >', @(~, ~) go(1); 'Next not reviewed (u)', @(~, ~) nextopen(); 'Play / pause (space)', @(~, ~) toggle(); ...
       'Save (s)', @(~, ~) save_csv()};
for i = 1:size(btn, 1)
    uicontrol(fig, 'Style', 'pushbutton', 'String', btn{i, 1}, 'Units', 'normalized', ...
              'Position', [0.01 + (i - 1) * 0.087, 0.03, 0.083, 0.06], 'Callback', btn{i, 2}, ...
              'FontSize', 10, 'KeyPressFcn', @onkey);
end
hStat = uicontrol(fig, 'Style', 'text', 'Units', 'normalized', 'Position', [0.63 0.03 0.36 0.06], ...
                  'HorizontalAlignment', 'left', 'BackgroundColor', 'w', 'FontSize', 10, 'Enable', 'inactive');
tm = timer('ExecutionMode', 'fixedSpacing', 'Period', 0.067, 'TimerFcn', @tick, 'BusyMode', 'drop');
hCur = gobjects(1);                               % trace cursor, made by select()

redraw_all();
select(1);

if selftest
    decide(0); decide(1); nextopen(); toggle(); tick(); tick(); toggle();   % decide moves on by itself
    set(hBg, 'Value', 3); onbg(); set(hBg, 'Value', 1); onbg();
    save_csv();
    R2 = readtable(csv, 'TextType', 'char');
    assert(R2.keep(1) == 0 && R2.reviewed(1) == 1 && R2.keep(2) == 1 && R2.reviewed(2) == 1, ...
           'cell_curation_gui selftest: decisions not saved');
    st.dirty = false;
    onclose();
    fprintf('cell_curation_gui selftest PASSED (%d cells, %d movie frames)\n', C, nF);
end

    % ------------------------------------------------------------------------------
    function img = background(k)
        switch st.bg
            case 'movie dF/F', img = D.movie_dff(:, :, k);
            case 'movie raw', img = D.movie_raw(:, :, k);
            case 'max dF/F', img = to8(D.proj.max_dff);
            case 'mean', img = to8(D.proj.mean);
            case 'max (background removed)', img = to8(D.proj.max_bgsub);
            otherwise, img = to8(D.proj.cn_pnr);
        end
    end

    function redraw_all()
        for c2 = 1:C, style(c2); end
        show_frame();
        status();
    end

    function style(c2)
        if R.keep(c2) == 1, col = [0.1 0.85 0.2]; else, col = [0.95 0.15 0.15]; end
        ls = '-';
        if R.reviewed(c2) == 0, ls = '--'; end
        lw = 1.1;
        if c2 == st.cur, lw = 2.6; col = col * 0.6 + [0.4 0.4 0.4] * (c2 == st.cur); end
        set(hOut{c2}, 'Color', col, 'LineStyle', ls, 'LineWidth', lw);
        set(hNum(c2), 'Color', col);
    end

    function show_frame()
        k = min(max(st.frame, 1), nF);
        img = background(k);
        set(hImg, 'CData', img);
        c2 = st.cur;
        r0 = round(D.cen(c2, 1)); c0 = round(D.cen(c2, 2));
        rr = (r0 - zoomhalf):(r0 + zoomhalf); cc = (c0 - zoomhalf):(c0 + zoomhalf);
        z = zeros(numel(rr), numel(cc), 'uint8');
        okr = rr >= 1 & rr <= Hs; okc = cc >= 1 & cc <= Ws;
        z(okr, okc) = img(rr(okr), cc(okc));
        set(hZ, 'CData', z);
        if strncmp(st.bg, 'movie', 5)
            set(hTitle, 'String', sprintf('%s   t = %.1f s   (%gx)   movie %.0f-%.0f s', st.bg, ...
                D.movie_t(k), st.speed, D.win(1), D.win(2)));
            if isgraphics(hCur), set(hCur, 'XData', D.movie_t(k) * [1 1]); end
        else
            set(hTitle, 'String', sprintf('%s (whole recording)', st.bg));
        end
    end

    function select(c2)
        old = st.cur;
        st.cur = min(max(c2, 1), C);
        style(old); style(st.cur);
        % zoom outline
        r0 = round(D.cen(st.cur, 1)); c0 = round(D.cen(st.cur, 2));
        xs = []; ys = [];
        for b = 1:numel(D.outline{st.cur})
            o = D.outline{st.cur}{b};
            xs = [xs; o(:, 2) - c0 + zoomhalf + 1; nan]; %#ok<AGROW>
            ys = [ys; o(:, 1) - r0 + zoomhalf + 1; nan]; %#ok<AGROW>
        end
        set(hZo, 'XData', xs, 'YData', ys, 'Color', tern(R.keep(st.cur) == 1, [0.1 0.85 0.2], [0.95 0.15 0.15]));
        % trace
        cla(axT);
        x = D.Tb(:, st.cur);
        patch(axT, [D.win(1) D.win(2) D.win(2) D.win(1)], [min(x) min(x) max(x) max(x)], ...
              [0.92 0.92 1], 'EdgeColor', 'none');
        hold(axT, 'on');
        plot(axT, D.tt, x, 'k-', 'LineWidth', 0.5);
        hCur = plot(axT, [1 1] * D.movie_t(st.frame), [min(x) max(x)], 'r-', 'LineWidth', 1.2);
        xlim(axT, [0 D.tt(end)]);
        ylim(axT, [min(x) max(x)] + [-0.05 0.05] * max(eps, max(x) - min(x)));
        xlabel(axT, 'time (s)  (blue = movie window, red = movie now)'); ylabel(axT, 'dF/F (least squares)');
        title(axT, sprintf('#%d trace', st.cur), 'FontWeight', 'normal');
        box(axT, 'off');
        info();
        show_frame();
        status();
    end

    function info()
        c2 = st.cur;
        s = sprintf('#%d   of %d\n\nauto verdict: %s\n%s\n\nSNR %.1f   skew %.2f\nsize %.2fx   axis %.1f\nmovie at its events r %.2f\n\nnow: %s%s', ...
            c2, C, R.verdict{c2}, wrapstr(R.reasons{c2}), R.snr(c2), R.skew(c2), R.area_ratio(c2), ...
            R.axis_ratio(c2), getnum(R, 'event_image_corr', c2), tern(R.keep(c2) == 1, 'KEEP', 'DELETE'), ...
            tern(R.reviewed(c2) == 1, '  (reviewed)', '  (not reviewed yet)'));
        set(hInfo, 'String', s);
    end

    function status()
        set(hStat, 'String', sprintf(['reviewed %d / %d    keep %d    delete %d    %s'], ...
            nnz(R.reviewed), C, nnz(R.keep == 1), nnz(R.keep == 0), tern(st.dirty, 'UNSAVED changes (s)', 'saved')));
    end

    function decide(k)
        R.keep(st.cur) = k;
        R.reviewed(st.cur) = 1;
        st.dirty = true;
        style(st.cur);
        select(st.cur);
        go(1);
    end

    function go(dk)
        select(mod(st.cur - 1 + dk, C) + 1);
    end

    function nextopen()
        o = find(R.reviewed == 0);
        if isempty(o), return; end
        nx = o(find(o > st.cur, 1));
        if isempty(nx), nx = o(1); end
        select(nx);
    end

    function toggle()
        if st.playing
            stop(tm); st.playing = false;
        else
            if ~strncmp(st.bg, 'movie', 5), set(hBg, 'Value', 1); onbg(); end
            st.playing = true;
            if ~selftest, start(tm); end
        end
    end

    function tick(~, ~)
        if ~isvalid(fig), return; end
        st.frame = st.frame + st.speed;
        if st.frame > nF, st.frame = 1; end
        show_frame();
        drawnow limitrate;
    end

    function onbg(~, ~)
        s = get(hBg, 'String');
        st.bg = s{get(hBg, 'Value')};
        show_frame();
    end

    function onclick(~, ~)
        p = get(axF, 'CurrentPoint');
        d = hypot(D.cen(:, 1) - p(1, 2), D.cen(:, 2) - p(1, 1));
        [mn, j] = min(d);
        if mn < 3 * D.radius * D.sc + 10, select(j); end
    end

    function onkey(~, e)
        switch e.Key
            case 'k', decide(1);
            case 'd', decide(0);
            case {'rightarrow', 'n'}, go(1);
            case {'leftarrow', 'p'}, go(-1);
            case 'u', nextopen();
            case 'space', toggle();
            case {'add', 'equal'}, st.speed = min(8, st.speed * 2);
            case {'subtract', 'hyphen'}, st.speed = max(1, st.speed / 2);
            case 's', save_csv();
        end
    end

    function save_csv()
        if ~st.backed_up && ~selftest
            [p, n] = fileparts(csv);
            copyfile(csv, fullfile(p, sprintf('%s_before_gui_%s.csv', n, char(datetime('now', 'Format', 'yyyyMMdd_HHmmss')))));
            st.backed_up = true;
        end
        writetable(R, csv);
        st.dirty = false;
        status();
        fprintf('saved %s: keep %d, delete %d, reviewed %d of %d\n', csv, nnz(R.keep == 1), ...
                nnz(R.keep == 0), nnz(R.reviewed), C);
    end

    function onclose(~, ~)
        if st.dirty && ~selftest
            q = questdlg('Save your keep / delete decisions?', 'Unsaved', 'Save', 'Discard', 'Cancel', 'Save');
            if strcmp(q, 'Cancel') || isempty(q), return; end
            if strcmp(q, 'Save'), save_csv(); end
        end
        try, stop(tm); delete(tm); catch, end %#ok<NOCOM>
        delete(fig);
    end
end


function a = to8(x)
q = prctile(x(:), [1 99.7]);
a = uint8(255 * min(max((x - q(1)) / max(q(2) - q(1), eps), 0), 1));
end


function out = tern(c, a, b)
if c, out = a; else, out = b; end
end


function s = wrapstr(s)
if isempty(s), s = '(no flags)'; return; end
s = strrep(s, '; ', newline);
end


function v = getnum(R, f, c)
if ismember(f, R.Properties.VariableNames), v = R.(f)(c); else, v = NaN; end
end
