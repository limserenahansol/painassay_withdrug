%% RUN_recover_day2_20261022.m  -  Day-2 recovery of ONE mouse (you, 2026-10-22: "the magenta x - I can see them on Day 2")
% The Day-1 footprints of the not-found neurons (ca_day2_recover.py -> analysis\crossday\<m>_recover_footprints.mat) are
% added to the Day-2 EXTRACT result and every trace is re-fitted jointly (rescue_footprints), then the Day-2 review
% numbers of every component are made again (extract_cell_review -> 4_cell_review_recover; that mouse's review rules).
%   matlab -batch "M='5939'; run('...\RUN_recover_day2_20261022.m')"
here = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_pipeline';
addpath(here);
OUT_ROOT = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_out\20261021';
tag = sprintf('%s_d2', M);
R = fullfile(OUT_ROOT, tag);
xm = fullfile(R, '3_extract', [tag '_extract.mat']);
rv = fullfile(R, '4_cell_review', [tag '_cell_review.csv']);
add = fullfile('C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis\crossday', [M '_recover_footprints.mat']);
t0 = tic;
try
    if isfile(fullfile(R, '3_extract', [tag '_extract_before_recover.mat']))
        error('recover:again', '%s was recovered already (backup exists) - restore it first', tag);
    end
    rescue_footprints(xm, rv, add);
    nw = fullfile(R, '4_cell_review_recover');
    if isfolder(nw), rmdir(nw, 's'); end
    if strcmp(M, '5941')
        extract_cell_review(xm, nw);
    else
        extract_cell_review(xm, nw, struct('qc', struct('piece_min_frac', 0.25, 'pieces_max', 4, 'axis_max', 4, 'area_max', 7)));
    end
    fprintf('#### recover %s done in %.1f min ####\n', tag, toc(t0) / 60);
catch err
    fprintf(2, '\n*** recover %s STOPPED after %.1f min: %s\n', tag, toc(t0) / 60, err.message);
    for k = 1:numel(err.stack), fprintf(2, '   %s line %d\n', err.stack(k).name, err.stack(k).line); end
    exit(1);
end
