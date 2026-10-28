function S = run_mini1p_day(cfg)
%RUN_MINI1P_DAY  The whole mini1p pipeline for one mouse-day (baseline 'pre' + injection
%   'post' session), stage by stage. Every stage writes its own files and is skipped when
%   they already exist, so the day can be run in pieces with your QC in between.
%
%   S = run_mini1p_day(cfg)
%
%   STAGES (cfg.stages, default {'mc', 'sweep'} - then look at the sweep sheet)
%     'mc'        inscopix_day: lossless .isxd -> h5 (pixel-exact, verified twice), NoRMCorre
%                 motion correction on one template for pre AND post, uint16 + gzip, the
%                 verified raw copy deleted. QC: <tag>_mc_qc.png, <rec>_mc_qc.png and the
%                 motion verdict printed here (largest shift, frames at the shift limit,
%                 frames that match the template badly).
%     'sweep'     extract_param_sweep: many EXTRACT settings on one active 5-min piece ->
%                 <tag>_sweep_sheet.png. YOU pick the setting (cfg.extract).
%     'extract'   extract_1p on the whole day with cfg.extract (struct of extract_1p fields,
%                 e.g. avg_cell_radius, T_min_snr, cellfind_min_snr, spatial_highpass_cutoff;
%                 'suggested' = the sweep's suggestion)
%     'review'    extract_cell_review: numbered map, one row per component, keep/drop table
%                 pre-filled. YOU edit keep in <tag>_cell_review.csv.
%     'keep'      apply_cell_keep -> <tag>_extract_kept.mat
%     'merge'     for each session: its ScoringAB file (found in cfg.scoring_dir by mouse +
%                 phase) + the kept cells -> analysis\<recording_id>\final_neuron_behavior.mat
%                 (merge_mini1p_scoring; camera frame n = Inscopix frame n, refused if the
%                 frame counts differ)
%     'classify'  mini2p classify_pain_neurons on each session: peri-event traces and, per
%                 cell and event, activated (up) / inhibited (down) / not, for pin, heat,
%                 light, mild, withdrawal, flinch, attending, lick/bite, guarding, escape
%     'all'       = every stage in order (stops with a message when a human step is missing)
%
%   cfg fields
%     mouse_id, day, sex, arm         arm = 'sbi553' | 'vehicle' (unblinded later is fine)
%     pre_isxd, post_isxd             raw IDAS movies
%     out_root                        e.g. C:\...\HEAL_mini1p_SBI553\mini1p_out\20261019
%     scoring_dir                     folder of the ScoringAB_*.mat files of this day
%     stages                          see above
%     extract                         struct | 'suggested' (default): pass-1 EXTRACT settings
%     extract_common                  struct of extract_1p fields used by BOTH seeded passes, e.g.
%                                     struct('detect_prebin', true, 'use_gpu', false)
%     extract_method                  ('seeded') pass 1 + Cn x PNR seeds without automatic
%                                     elimination (extract_1p_seeded, the 5941 recipe) | 'plain'
%     sweep                           struct of extract_param_sweep options (optional)
%     mc                              struct of extra inscopix_day fields (optional)
%     merge_opts                      struct passed to merge_mini1p_scoring (optional)
%     accept_mc_check                 (false) the extract stage refuses to run when the
%                                     motion QC says CHECK, until you set this after looking
%     toolbox_dir                     mini2p pipeline folder (classify_pain_neurons,
%                                     mini2p_toolbox_paths), default the mini2p folder

if ~isfield(cfg, 'stages') || isempty(cfg.stages), cfg.stages = {'mc', 'sweep'}; end
if ischar(cfg.stages), cfg.stages = {cfg.stages}; end
order = {'mc', 'sweep', 'extract', 'review', 'keep', 'merge', 'classify'};
if any(strcmp(cfg.stages, 'all')), cfg.stages = order; end
bad = setdiff(cfg.stages, order);
if ~isempty(bad), error('run_mini1p_day:stages', 'unknown stage(s): %s', strjoin(bad, ', ')); end
if ~isfield(cfg, 'extract') || isempty(cfg.extract), cfg.extract = 'suggested'; end
if ~isfield(cfg, 'sweep'), cfg.sweep = struct(); end
if ~isfield(cfg, 'mc'), cfg.mc = struct(); end
if ~isfield(cfg, 'toolbox_dir') || isempty(cfg.toolbox_dir)
    cfg.toolbox_dir = fullfile(getenv('USERPROFILE'), 'OneDrive', 'Desktop', 'cursor', 'pain_2plane_pipeline');
end
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, 'scoring_gui'));

tag = sprintf('%s_d%d', cfg.mouse_id, cfg.day);
D = struct();
D.root = fullfile(cfg.out_root, tag);
D.mc = fullfile(D.root, '1_movie');
D.sweep = fullfile(D.root, '2_sweep');
D.extract = fullfile(D.root, '3_extract');
D.review = fullfile(D.root, '4_cell_review');
D.analysis = fullfile(D.root, '5_analysis');
F = struct();
F.day_json = fullfile(D.mc, [tag '_inscopix.json']);
F.sweep_json = fullfile(D.sweep, [tag '_sweep.json']);
F.extract_mat = fullfile(D.extract, [tag '_extract.mat']);
F.review_csv = fullfile(D.review, [tag '_cell_review.csv']);
F.kept_mat = fullfile(D.extract, [tag '_extract_kept.mat']);
S = struct('tag', tag, 'dirs', D, 'files', F, 'done', {{}}, 'next', '');
if ~isfolder(D.root), mkdir(D.root); end
logf = fullfile(D.root, [tag '_run_log.txt']);
diary(logf);
cleanup = onCleanup(@() diary('off'));
fprintf('\n===== run_mini1p_day %s  (%s)  stages: %s =====\n', tag, ...
        char(datetime('now')), strjoin(cfg.stages, ', '));

for st = order
    if ~any(strcmp(cfg.stages, st{1})), continue; end
    switch st{1}
        case 'mc'
            if done_json(F.day_json)
                fprintf('[mc] already done: %s\n', F.day_json);
            else
                c = cfg.mc;
                c.mouse_id = cfg.mouse_id;
                c.day = cfg.day;
                c.out_dir = D.mc;
                % normally pre + post; one session alone is allowed (a test, or the other
                % session not copied yet) - then that day must be re-run when both exist,
                % so that pre and post share one template and one EXTRACT run
                assays = {'pre', 'post'};
                files = {get_field(cfg, 'pre_isxd'), get_field(cfg, 'post_isxd')};
                have = ~cellfun(@isempty, files);
                if ~any(have), error('run_mini1p_day:cfg', 'no pre_isxd / post_isxd given'); end
                cellfun(@need_file, files(have));
                if ~all(have)
                    warning('run_mini1p_day:one_session', ['only the %s session is processed; ' ...
                            're-run the day when both sessions exist'], assays{have});
                end
                c.recordings = struct('assay', assays(have), 'isxd', files(have));
                inscopix_day(c);
            end
            S.mc_verdict = mc_check(F.day_json);
        case 'sweep'
            need_file(F.day_json);
            if isfile(F.sweep_json)
                fprintf('[sweep] already done: %s\n', F.sweep_json);
            else
                c = cfg.sweep;
                c.day_json = F.day_json;
                c.out_dir = D.sweep;
                c.tag = tag;
                extract_param_sweep(c);
            end
            S.next = sprintf(['LOOK at %s (and %s), choose the setting, and run the next ' ...
                'stages with cfg.extract = struct(''avg_cell_radius'', .., ''T_min_snr'', ..) or ' ...
                '''suggested'''], fullfile(D.sweep, [tag '_sweep_sheet.png']), ...
                fullfile(D.sweep, [tag '_sweep_overview.png']));
        case 'extract'
            need_file(F.day_json);
            % QC gate: one footprint set for pre AND post only makes sense when both
            % sessions were registered to the same field of view.
            v = mc_check(F.day_json);
            bad = v(startsWith({v.verdict}, 'CHECK'));
            if ~isempty(bad) && ~(isfield(cfg, 'accept_mc_check') && cfg.accept_mc_check)
                error('run_mini1p_day:mc_qc', ['motion-correction QC says CHECK for %s. Look at ' ...
                      '%s and the per-session mc_qc.png. If pre and post are not the same field ' ...
                      'of view, they cannot share one EXTRACT run. Set cfg.accept_mc_check = true ' ...
                      'only after you have looked.'], strjoin({bad.recording_id}, ', '), ...
                      fullfile(D.mc, [tag '_mc_qc.png']));
            end
            if isfile(F.extract_mat)
                fprintf('[extract] already done: %s (delete it to re-run with new settings)\n', F.extract_mat);
            else
                c = extract_settings(cfg.extract, F.sweep_json);
                method = 'seeded';
                if isfield(cfg, 'extract_method') && ~isempty(cfg.extract_method), method = cfg.extract_method; end
                fprintf('[extract] method %s, pass-1 settings: %s\n', method, jsonencode(c));
                if strcmpi(method, 'plain')
                    c.day_json = F.day_json;
                    c.out_dir = D.extract;
                    extract_1p(c);
                else
                    % the 5941 recipe: pass 1 + Cn x PNR seeds, no automatic elimination
                    sc = struct('day_json', F.day_json, 'out_dir', D.extract, 'pass1', c);
                    % cfg.extract_common: extract_1p fields for BOTH passes (e.g. detect_prebin, use_gpu)
                    if isfield(cfg, 'extract_common') && isstruct(cfg.extract_common)
                        fc = fieldnames(cfg.extract_common);
                        for q = 1:numel(fc), sc.(fc{q}) = cfg.extract_common.(fc{q}); end
                    end
                    % cfg.extract_seed (seed_footprints options, e.g. k_sd, min_sep, extra_rc = your marks,
                    % extra_replace) and cfg.extract_pass1_mat (re-use a pass-1 result of the same movie)
                    if isfield(cfg, 'extract_seed') && isstruct(cfg.extract_seed), sc.seed = cfg.extract_seed; end
                    if isfield(cfg, 'extract_pass1_mat') && ~isempty(cfg.extract_pass1_mat), sc.pass1_mat = cfg.extract_pass1_mat; end
                    extract_1p_seeded(sc);
                end
            end
        case 'review'
            need_file(F.extract_mat);
            if isfile(F.review_csv)
                fprintf('[review] table exists (your decisions are kept): %s\n', F.review_csv);
            else
                if isfield(cfg, 'review_qc') && isstruct(cfg.review_qc)
                    extract_cell_review(F.extract_mat, D.review, struct('qc', cfg.review_qc));
                else
                    extract_cell_review(F.extract_mat, D.review);
                end
            end
            pm = fullfile(D.extract, [tag '_projections.mat']);
            gd = fullfile(D.review, [tag '_curation_data.mat']);
            if isfile(pm) && ~isfile(gd)
                prepare_curation_data(F.extract_mat, F.review_csv, pm, gd);
            end
            if isfile(gd)
                fprintf('[review] GUI: cell_curation_gui(''%s'')\n', gd);
            end
            S.next = sprintf(['LOOK at %s and the page PNGs, set keep = 1 / 0 in %s, then run ' ...
                'stages {''keep'', ''merge'', ''classify''}'], ...
                fullfile(D.review, [tag '_cells_map.png']), F.review_csv);
        case 'keep'
            need_file(F.review_csv);
            apply_cell_keep(F.extract_mat, F.review_csv, F.kept_mat);
        case 'merge'
            need_file(F.kept_mat);
            S.sessions = merge_sessions(cfg, F, D, tag);
        case 'classify'
            S.classified = classify_sessions(cfg, D, tag);
    end
    S.done{end + 1} = st{1};
end
if ~isempty(S.next), fprintf('\nNEXT: %s\n', S.next); end
fprintf('===== run_mini1p_day %s finished (%s) =====\n', tag, char(datetime('now')));
end


% =====================================================================================
function tf = done_json(f)
tf = false;
if ~isfile(f), return; end
js = jsondecode(fileread(f));
tf = isfield(js, 'status') && strcmp(js.status, 'ok');
end


function need_file(f)
if ~isfile(f)
    error('run_mini1p_day:missing', 'needed but missing: %s', f);
end
end


function v = mc_check(day_json)
% Motion verdict from every recording's shifts table.
js = jsondecode(fileread(day_json));
recs = js.recordings;
if ~iscell(recs), recs = num2cell(recs); end
mx = js.params.mc_max_shift_px;
v = struct('recording_id', {}, 'max_abs_shift_px', {}, 'n_at_limit', {}, ...
           'corr_after_median', {}, 'n_bad_frames', {}, 'verdict', {});
for r = 1:numel(recs)
    sh = readtable(relocate(recs{r}.mc_shifts_csv, fileparts(day_json)), 'TextType', 'char');
    a = hypot(sh.x_shift_px, sh.y_shift_px);
    ca = sh.corr_after;
    md = median(ca, 'omitnan');
    mad_ = 1.4826 * median(abs(ca - md), 'omitnan');
    nb = nnz(ca < md - max(5 * mad_, 0.05));
    lim = nnz(max(abs(sh.x_shift_px), abs(sh.y_shift_px)) >= 0.95 * mx);
    verdict = 'ok';
    if lim > 0, verdict = 'CHECK: shifts reach the limit (raise mc_max_shift_px)'; end
    if nb > 0.01 * height(sh), verdict = 'CHECK: >1% of frames match the template badly (see mc_qc.png)'; end
    v(end + 1) = struct('recording_id', recs{r}.recording_id, 'max_abs_shift_px', max(a), ...
        'n_at_limit', lim, 'corr_after_median', md, 'n_bad_frames', nb, 'verdict', verdict); %#ok<AGROW>
    fprintf('[mc QC] %-16s max shift %5.1f px, %d frames at the %g px limit, corr after %.3f, %d poorly matched frames -> %s\n', ...
            recs{r}.recording_id, max(a), lim, mx, md, nb, verdict);
end
end


function p = relocate(p, d)
if ~isfile(p)
    [~, nm, ext] = fileparts(p);
    p = fullfile(d, [nm ext]);
end
end


function c = extract_settings(spec, sweep_json)
if isstruct(spec)
    c = spec;
    return;
end
if ~strcmpi(spec, 'suggested')
    error('run_mini1p_day:extract', 'cfg.extract must be a struct or ''suggested''');
end
need_file(sweep_json);
js = jsondecode(fileread(sweep_json));
if ~isfield(js, 'suggested') || isempty(js.suggested)
    error('run_mini1p_day:extract', ['the sweep suggested nothing (every setting had > 25 %% ' ...
          'not_cell): choose cfg.extract by hand from the sweep sheet']);
end
c = js.suggested;
fprintf('[extract] using the sweep SUGGESTION (run %d) - check the sheet agrees\n', js.suggested_run);
end


function T = merge_sessions(cfg, F, D, tag)
phases = {'pre', 'Baseline'; 'post', 'Post-treatment'};
T = struct('recording_id', {}, 'scoring_mat', {}, 'analysis_dir', {}, 'n_cells', {});
L = dir(fullfile(cfg.scoring_dir, 'ScoringAB_*.mat'));
kr = load(F.kept_mat, 'recording_ids');
for p = 1:2
    rid = sprintf('%s_%s', tag, phases{p, 1});
    if ~ismember(rid, cellstr(string(kr.recording_ids)))
        fprintf('[merge] %s: not in this day''s EXTRACT run\n', rid);
        continue;
    end
    hit = {};
    for i = 1:numel(L)
        f = fullfile(L(i).folder, L(i).name);
        w = load(f, 'mouseID', 'phase');
        if isfield(w, 'mouseID') && strcmp(strtrim(char(string(w.mouseID))), cfg.mouse_id) && ...
                isfield(w, 'phase') && strcmpi(strtrim(char(string(w.phase))), phases{p, 2})
            hit{end + 1} = f; %#ok<AGROW>
        end
    end
    if isempty(hit)
        fprintf('[merge] %s: no ScoringAB file with mouse %s, phase %s in %s - not scored yet\n', ...
                rid, cfg.mouse_id, phases{p, 2}, cfg.scoring_dir);
        continue;
    end
    if numel(hit) > 1
        error('run_mini1p_day:scoring', '%s: %d scoring files match: %s', rid, numel(hit), strjoin(hit, ', '));
    end
    ad = fullfile(D.analysis, rid);
    mo = struct();
    if isfield(cfg, 'merge_opts'), mo = cfg.merge_opts; end
    o = merge_mini1p_scoring(F.kept_mat, rid, hit{1}, ad, mo);
    T(end + 1) = struct('recording_id', rid, 'scoring_mat', hit{1}, 'analysis_dir', ad, ...
                        'n_cells', o.n_cells); %#ok<AGROW>
end
end


function out = classify_sessions(cfg, D, tag)
addpath(cfg.toolbox_dir);
out = {};
for a = {'pre', 'post'}
    ad = fullfile(D.analysis, sprintf('%s_%s', tag, a{1}));
    if ~isfile(fullfile(ad, 'final_neuron_behavior.mat'))
        fprintf('[classify] %s: no final_neuron_behavior.mat (merge first)\n', ad);
        continue;
    end
    o = struct();
    o.pre_sec = 3;
    o.post_sec = 5;
    o.f0_start_sec = -3;
    o.f0_end_sec = -1;
    o.save_fig = true;
    o.extra_events = {'light_touch_vector', 'Light touch', [0.30 0.60 0.95]; ...
                      'mild_touch_vector',  'Mild touch',  [0.95 0.70 0.20]};
    o.extra_stim_fields = {'light_touch_vector', 'mild_touch_vector'};
    classify_pain_neurons(ad, o);
    out{end + 1} = ad; %#ok<AGROW>
end
end


function v = get_field(s, f)
v = '';
if isfield(s, f) && ~isempty(s.(f)), v = char(s.(f)); end
end
