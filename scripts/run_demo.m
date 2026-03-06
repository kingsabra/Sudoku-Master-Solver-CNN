% run_demo  Run full pipeline: load trained CNN, recognize digits (batch), solve Sudoku.
% Phase 3: appends 81 cells + solution labels to data/sudoku_cells after each image.
% Phase 4: batch inference + confidence. Prefers models/MnistConv_sudoku.mat if present.

root = pwd;
addpath(fullfile(root, 'src', 'cnn'), fullfile(root, 'src', 'solver'), fullfile(root, 'utils'));

SAVE_CELLS_TO_DATASET = true;   % Phase 3: append (cell, solution digit) after each image
CONFIDENCE_THRESH = 0.5;        % Phase 4: below this max prob, optionally use OCR for that cell
CELL_SZ = 28;

% Prefer Sudoku-fine-tuned model
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
fprintf('Loading model from %s\n', modelPath);
load(modelPath);
if exist('X_sample', 'var')
    clear X_sample;
end

fg_fn = get_demo_files(root);
num_files = numel(fg_fn);

if num_files == 0
    error('No test images found. Add .jpg/.jpeg/.png images to images/.');
end

total_acc = [];
for idx = 1:num_files
    fprintf('\nProcessing file: %s\n', fg_fn{idx});

    I_FG = imread(fg_fn{idx});
    figure(1);
    imshow(I_FG);
    pause;

    Grayscale = rgb2gray(I_FG);
    BW = imbinarize(Grayscale, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.4);
    BW2 = bwareafilt(imcomplement(BW), 1, 'largest');
    label = bwlabel(BW2);
    [row, col] = find(label == 1);

    BW3 = BW(min(row):max(row), min(col):max(col));
    BW4 = BW2(min(row):max(row), min(col):max(col));
    BW5 = logical(imcomplement(BW3) - BW4);

    [width, height] = size(BW5);
    sub_width  = floor(width / 9);
    sub_height = floor(height / 9);
    sudokoMatrix = zeros(9, 9);

    Xi = 1;
    for r = 1:9
        Yi = 1;
        for c = 1:9
            BW6 = BW5(Xi:r*sub_height, Yi:c*sub_width);
            Yi = c * sub_width;
            l = ocr(BW6, 'CharacterSet', '0123456789', 'TextLayout', 'Block');
            T = double(l.Text);
            lenT = size(T);
            if lenT(2) ~= 0 && T(1) >= 49 && T(1) <= 57
                sudokoMatrix(r,c) = T(1) - 48;
            else
                sudokoMatrix(r,c) = 0;
            end
        end
        Xi = r * sub_height;
    end

    Truth_board = zeros(9);
    for i = 1:9
        for k = 1:9
            if sudokoMatrix(i,k) > 0
                Truth_board(i, k) = sudokoMatrix(i, k);
            end
        end
    end

    I_blur = imgaussfilt(Grayscale, 1);
    I_RbIn = double(I_blur <= 180);
    new = imresize(I_RbIn, [270 270]);
    cell_width = 30;
    cell_height = 30;

    Cell_positions = [];
    y_1 = 0;
    for i = 1:9
        y_2 = y_1 + cell_height;
        x_1 = 1;
        for j = 1:9
            x_2 = x_1 + cell_width;
            Cell_positions = [Cell_positions; x_1, x_2, y_1, y_2];
            x_1 = x_2;
        end
        y_1 = y_2;
    end

    % Build 28x28x81 cell stack (double [0,1]) for batch inference
    cells_81 = zeros(CELL_SZ, CELL_SZ, 81);
    for ic = 1:81
        cell_img = new(Cell_positions(ic,3)+1:Cell_positions(ic,4)-2, Cell_positions(ic,1)+1:Cell_positions(ic,2)-2);
        cell_img = imresize(cell_img, [CELL_SZ CELL_SZ]);
        cell_img = double(cell_img);
        if max(cell_img(:)) > min(cell_img(:))
            cell_img = (cell_img - min(cell_img(:))) / (max(cell_img(:)) - min(cell_img(:)));
        end
        cells_81(:, :, ic) = cell_img;
    end

    % Phase 4: batch inference + confidence
    [preds, probs] = CNNForwardBatch(cells_81, W1, W5, Wo);
    max_probs = max(probs, [], 2);

    board = reshape(preds, 9, 9)';
    % Confidence fallback: use OCR (Truth_board) where CNN is uncertain
    for i = 1:9
        for j = 1:9
            k = (i-1)*9 + j;
            if max_probs(k) < CONFIDENCE_THRESH && Truth_board(i,j) > 0
                board(i,j) = Truth_board(i,j);
            end
        end
    end

    similarity = 0;
    for i = 1:9
        for j = 1:9
            if board(i,j) == Truth_board(i,j)
                similarity = similarity + 1;
                if board(i,j) == 0
                    board(i,j) = NaN;
                end
            else
                board(i,j) = Truth_board(i,j);
            end
        end
    end

    acc = (similarity / 81) * 100;
    total_acc = [total_acc; acc];

    board_for_solver = board;
    board_for_solver(board_for_solver == 0) = NaN;
    Solution = solveSudoku(board_for_solver);

    % Phase 3: append (cells, solution) to dataset for continuous learning
    if SAVE_CELLS_TO_DATASET
        append_cells_to_dataset(root, cells_81, Solution);
    end

    disp('Detected board (filled with ground truth where different):');
    disp(board);
    disp('Solution:');
    disp(Solution);
    close all
end

if numel(total_acc) > 0
    Final_acc = sum(total_acc) / numel(total_acc);
    fprintf('\nFinal accuracy (mean over %d images): %.2f%%\n', numel(total_acc), Final_acc);
end

%% Local helper
function fg_fn = get_demo_files(root_dir)
    img_dir = fullfile(root_dir, 'images');
    fg_fn = {};
    exts = {'.jpg', '.jpeg', '.png'};
    files = [];
    for e = 1:numel(exts)
        files = [files; dir(fullfile(img_dir, ['*', exts{e}]))]; %#ok<AGROW>
    end
    if isempty(files)
        return;
    end
    % Sort for stable ordering
    [~, order] = sort({files.name});
    files = files(order);
    for i = 1:numel(files)
        fg_fn{i} = fullfile(img_dir, files(i).name);
    end
end
