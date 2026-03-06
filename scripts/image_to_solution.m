% image_to_solution  End-to-end: one image path -> detected board + solution (Phase 5).
% Loads Sudoku model (or base model), runs grid detection + batch CNN + backtracking solver.
% Usage: [board, solution, cells_81, Truth_board] = image_to_solution(root, image_path);
%   root: repository root. image_path: path to Sudoku image.
%   board: 9x9 detected (NaN = empty). solution: 9x9 solved.

function [board, solution, cells_81, Truth_board] = image_to_solution(root, image_path)

addpath(fullfile(root, 'src', 'cnn'), fullfile(root, 'src', 'solver'), fullfile(root, 'src', 'vision'), fullfile(root, 'utils'));

if nargin < 2
    image_path = [];
end
if nargin < 1 || isempty(root)
    root = pwd;
end

% Load model
modelPath = fullfile(root, 'models', 'MnistConv_sudoku.mat');
if ~isfile(modelPath)
    modelPath = fullfile(root, 'models', 'MnistConv.mat');
end
if ~isfile(modelPath)
    modelPath = fullfile(root, 'MnistConv.mat');
end
if ~isfile(modelPath)
    error('No trained model. Run scripts/train_model (and optionally train_sudoku_cells) first.');
end
load(modelPath, 'W1', 'W5', 'Wo');
if exist('X_sample', 'var')
    clear X_sample;
end

if isempty(image_path)
    fg = [dir(fullfile(root, 'images', '*.jpg')); dir(fullfile(root, 'images', '*.jpeg')); dir(fullfile(root, 'images', '*.png'))];
    if isempty(fg)
        error('No image path given and no .jpg/.jpeg/.png images in images/.');
    end
    image_path = fullfile(root, 'images', fg(1).name);
end

I = imread(image_path);
[cells_81, Truth_board, ~] = detect_grid(I, 28);

[preds, probs] = CNNForwardBatch(cells_81, W1, W5, Wo);
board = reshape(preds, 9, 9)';
CONFIDENCE_THRESH = 0.5;
max_probs = max(probs, [], 2);
for i = 1:9
    for j = 1:9
        k = (i-1)*9 + j;
        if max_probs(k) < CONFIDENCE_THRESH && Truth_board(i,j) > 0
            board(i,j) = Truth_board(i,j);
        end
    end
end

board_nan = board;
board_nan(board_nan == 0) = NaN;
solution = solveSudoku(board_nan);

end
