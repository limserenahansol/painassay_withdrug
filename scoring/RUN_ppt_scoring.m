%% RUN_ppt_scoring.m  -  score the short PPT clip. Open and press F5.
%
%  This is the PPT-only twin of RUN_scoring.m. It points at the short clip
%  that make_ppt_clip.py --cut produced, NOT at your real sessions, so
%  nothing you do here can touch videos\output.
%
%  ---------------------------------------------------------------------------
%  THE THREE STEPS
%
%   1) in a terminal, cut the window you want:
%
%        cd C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis
%        python make_ppt_clip.py --cut --mouse female3 --stimulus "Pin prick"
%
%      add --start 1150 to choose the moment yourself, or leave it out and it
%      picks the busiest 30 s of that block.
%
%   2) run THIS file (F5). Pick mode 3, score the clip.
%
%   3) back in the terminal, render the slide video:
%
%        python make_ppt_clip.py --render-scored --zoom
%
%  ---------------------------------------------------------------------------
%  SCORING THIS CLIP
%
%    It is only ~30 s, so score it carefully - this is the one that goes on
%    the screen in front of the lab.
%
%    HOLD   a attending          s licking or biting
%           d guarding           f escape / rearing
%    TAP    w withdrawal         e flinch
%    TAP    1 2 3 4  = stimulus delivered, by type
%    SPACE pause   left/right step   , slower   . faster   z undo   q save
%
%    HOLD the a/s/d/f keys down for as long as the behaviour lasts. For the
%    demo the coloured bars want to have real width - a tapped key gives a
%    3-frame sliver that is invisible on a projector.
%
%    Scoring at 0.3-0.5x (press , a few times) makes the holds much easier
%    to place on a clip this short.
%  ---------------------------------------------------------------------------

clear; clc; close all;

CLIPDIR = 'C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\ppt_clips\score_me';
bottomFolder = fullfile(CLIPDIR, 'cameraA');
sideFolder   = fullfile(CLIPDIR, 'cameraB');

addpath(fileparts(mfilename('fullpath')));

exts = {'*.avi', '*.mp4', '*.mov'};
nBot = countVideos(bottomFolder, exts);
nSid = countVideos(sideFolder, exts);

fprintf('PPT clip folder\n  %s\n', CLIPDIR);
fprintf('  bottom %d video(s)\n  side   %d video(s)\n\n', nBot, nSid);

if nBot == 0 || nSid == 0
    fprintf(2, 'No clip to score yet.\n\n');
    fprintf(2, ['Cut one first, in a terminal:\n' ...
                '  cd C:\\Users\\hsollim\\Documents\\HEAL_mini1p_SBI553\\analysis\n' ...
                '  python make_ppt_clip.py --cut --mouse female3 ' ...
                '--stimulus "Pin prick"\n']);
    return;
end

fprintf('Starting. Pick mode 3, then HOLD the behaviour keys.\n');
fprintf('Results go to %s\\output\\\n\n', CLIPDIR);
score_AB_dual_view(bottomFolder, sideFolder);

fprintf('\nDone. Now render the slide video:\n');
fprintf('  cd C:\\Users\\hsollim\\Documents\\HEAL_mini1p_SBI553\\analysis\n');
fprintf('  python make_ppt_clip.py --render-scored --zoom\n');


%% ------------------------------------------------------------------------
function n = countVideos(folder, exts)
    n = 0;
    if ~exist(folder, 'dir')
        return;
    end
    for i = 1:numel(exts)
        n = n + numel(dir(fullfile(folder, exts{i})));
    end
end
