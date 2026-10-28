%% RUN_ca_partition_test_20261019.m  -  do EXTRACT's partition seams change the detected cells?
% The 3-min preview of 5941 SBI ran EXTRACT in 2 x 2 partitions (free RAM was low): a line of
% false candidates appeared along the vertical seam. Here the SAME movie and recipe run in ONE
% partition: detection on a movie binned by extract_1p itself (detect_prebin, time 4 x space 2),
% which fits EXTRACT in one partition; ca_partition_compare.py then matches the candidates.

addpath('C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_pipeline');
od = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\mini1p_out\20261019_preview3min\5941_d1_post';
mv = fullfile(od, '1_movie');
ex = extract_1p_seeded(struct('h5', fullfile(mv, '5941_d1_post_mc.h5'), ...
    'frames_csv', fullfile(mv, '5941_d1_post_raw_frames.csv'), 'mouse_id', '5941', 'day', 1, ...
    'out_dir', fullfile(od, '3_extract_prebin'), 'window_s', [15 180], 'detect_prebin', true, 'use_gpu', false));
rv = extract_cell_review(ex.files.mat, fullfile(od, '4_review_prebin'));
fprintf('PARTITION TEST DONE: %d candidates -> %d cell, %d check, %d not_cell\n', ex.n_cells, rv.n_cell, ...
        rv.n_check, rv.n_not_cell);
