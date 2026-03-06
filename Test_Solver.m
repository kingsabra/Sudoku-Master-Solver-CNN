function Test_Solver
% Test_Solver  Entry point: sets up paths and runs the full pipeline (run_demo).
% Run from repository root. Train once with scripts/train_model, then run Test_Solver.

root = pwd;
addpath(fullfile(root, 'scripts'), fullfile(root, 'src', 'cnn'), ...
        fullfile(root, 'src', 'solver'), fullfile(root, 'utils'));

run(fullfile(root, 'scripts', 'run_demo.m'));
