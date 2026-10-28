% RUN_MATLAB_SELFTESTS  Every MATLAB module selftest, headless:
%   matlab -batch "run('<this folder>\run_matlab_selftests.m')"
% Any failing assert raises an error, so -batch exits non-zero.
% restoredefaultpath first: a saved path with ActSort shadows ~20 EXTRACT helpers.
restoredefaultpath;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
addpath(here);
cd(here);
t0 = tic; isxd_to_h5('selftest');        fprintf('isxd_to_h5 selftest PASS (%.0f s)\n', toc(t0));
t0 = tic; motion_correct_h5('selftest'); fprintf('motion_correct_h5 selftest PASS (%.0f s)\n', toc(t0));
t0 = tic; inscopix_day('selftest');      fprintf('inscopix_day selftest PASS (%.0f s)\n', toc(t0));
t0 = tic; extract_1p('selftest');        fprintf('extract_1p selftest PASS (%.0f s)\n', toc(t0));
