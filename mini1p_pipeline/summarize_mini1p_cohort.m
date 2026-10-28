function out = summarize_mini1p_cohort(out_root, mice)
%SUMMARIZE_MINI1P_COHORT  Collect every session's mini2p classification into two tables.
%
%   out = summarize_mini1p_cohort(out_root, mice)
%
%   out_root : the run_mini1p_day out_root (folders <mouse>_d<day>\5_analysis\<recording_id>)
%   mice     : struct array with fields id, sex, arm (as in the RUN script)
%
%   Writes into out_root
%     cohort_session_counts.csv  one row per session: mouse, sex, arm, phase, n cells, and
%                                for every event (pin, heat, light, mild, withdrawal,
%                                flinch, attending, lick/bite, guarding, escape) the number
%                                and % of cells activated (up) and inhibited (down)
%     cohort_cells_pre_post.csv  one row per cell of every mouse-day: for every event its
%                                class before (pre) and after the injection (post):
%                                up / down / none. Pre and post share one set of
%                                footprints, so a row is one neuron in both sessions.
%   Classes come from classify_pain_neurons (signrank of the mean peri-event signal in
%   [1, 3] s vs [-3, -1] s across trials, p < 0.05, >= 3 trials) - nothing is re-tested here.

events = {'PinPrick', 'Heat', 'LightTouch', 'MildTouch', 'Withdrawal', 'Flinch', ...
          'Attending', 'Lick_Bite', 'Guarding', 'Escape_Rear'};
L = dir(fullfile(out_root, '*_d*', '5_analysis', '*', 'neuron_pain_classification.csv'));
if isempty(L)
    error('summarize_mini1p_cohort:none', 'no neuron_pain_classification.csv under %s', out_root);
end
rows = {};
cells = containers.Map();
for i = 1:numel(L)
    rid = regexp(L(i).folder, '[^\\/]+$', 'match', 'once');
    tok = regexp(rid, '^(.+)_d(\d+)_(pre|post)$', 'tokens', 'once');
    if isempty(tok), continue; end
    mouse = tok{1}; day = str2double(tok{2}); phase = tok{3};
    j = find(strcmp({mice.id}, mouse), 1);
    sex = ''; arm = '';
    if ~isempty(j), sex = mice(j).sex; arm = mice(j).arm; end
    T = readtable(fullfile(L(i).folder, L(i).name), 'TextType', 'char');
    n = height(T);
    r = struct('recording_id', rid, 'mouse_id', mouse, 'day', day, 'sex', sex, 'arm', arm, ...
               'phase', phase, 'n_cells', n);
    key = sprintf('%s_d%d', mouse, day);
    if ~isKey(cells, key)
        cells(key) = table(T.orig_planB_id, 'VariableNames', {'cell_id'});
    end
    C = cells(key);
    for e = events
        up = col(T, [e{1} '_resp']);
        dn = col(T, [e{1} '_resp_down']);
        nt = col(T, [e{1} '_n_trials']);
        r.([e{1} '_trials']) = max([nt; 0]);
        r.([e{1} '_up']) = nnz(up);
        r.([e{1} '_down']) = nnz(dn);
        r.([e{1} '_up_pct']) = 100 * nnz(up) / max(n, 1);
        r.([e{1} '_down_pct']) = 100 * nnz(dn) / max(n, 1);
        cls = repmat({'none'}, n, 1);
        cls(logical(up)) = {'up'};
        cls(logical(dn)) = {'down'};
        if isempty(up), cls(:) = {'not_tested'}; end
        [ok, loc] = ismember(C.cell_id, T.orig_planB_id);
        v = repmat({''}, height(C), 1);
        v(ok) = cls(loc(ok));
        C.([e{1} '_' phase]) = v;
    end
    cells(key) = C;
    rows{end + 1} = r; %#ok<AGROW>
end
S = struct2table([rows{:}]);
S = sortrows(S, {'mouse_id', 'day', 'phase'}, {'ascend', 'ascend', 'descend'});
f1 = fullfile(out_root, 'cohort_session_counts.csv');
writetable(S, f1);
K = keys(cells);
P = table();
for k = 1:numel(K)
    C = cells(K{k});
    tok = regexp(K{k}, '^(.+)_d(\d+)$', 'tokens', 'once');
    j = find(strcmp({mice.id}, tok{1}), 1);
    arm = ''; if ~isempty(j), arm = mice(j).arm; end
    C = [table(repmat(K(k), height(C), 1), repmat({arm}, height(C), 1), ...
               'VariableNames', {'mouse_day', 'arm'}), C]; %#ok<AGROW>
    P = outerjoin_rows(P, C);
end
f2 = fullfile(out_root, 'cohort_cells_pre_post.csv');
writetable(P, f2);
fprintf('summarize_mini1p_cohort: %d sessions -> %s, %d cells -> %s\n', height(S), f1, height(P), f2);
out = struct('sessions', S, 'cells', P, 'files', {{f1, f2}});
end


function v = col(T, name)
if ismember(name, T.Properties.VariableNames)
    v = T.(name);
    if iscell(v), v = strcmpi(v, 'true') | strcmp(v, '1'); end
    v = double(v(:));
else
    v = zeros(0, 1);
end
end


function P = outerjoin_rows(P, C)
% Stack tables whose columns may differ (a session not yet scored has no post columns).
if isempty(P), P = C; return; end
for v = setdiff(C.Properties.VariableNames, P.Properties.VariableNames)
    P.(v{1}) = repmat({''}, height(P), 1);
end
for v = setdiff(P.Properties.VariableNames, C.Properties.VariableNames)
    C.(v{1}) = repmat({''}, height(C), 1);
end
P = [P; C(:, P.Properties.VariableNames)];
end
