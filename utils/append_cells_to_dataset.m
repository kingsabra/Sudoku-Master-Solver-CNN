function append_cells_to_dataset(root, X_cells_81, labels_81)
% append_cells_to_dataset  Append 81 cell images and labels to data/sudoku_cells/sudoku_cells.mat (Phase 3).
%   root: repository root path.
%   X_cells_81: [28 x 28 x 81] double (cell images from one Sudoku image).
%   labels_81: [9 x 9] or [81 x 1] — 0 = empty, 1–9 = digit. Use solution board so filled cells get correct labels.

labels_81 = labels_81(:);
if numel(labels_81) ~= 81
    error('labels_81 must have 81 elements');
end

data_dir = fullfile(root, 'data', 'sudoku_cells');
mat_path = fullfile(data_dir, 'sudoku_cells.mat');

if ~isfolder(data_dir)
    mkdir(data_dir);
end

if isfile(mat_path)
    ld = load(mat_path);
    X_old = ld.X_cells;
    Y_old = ld.Y_labels;
else
    X_old = [];
    Y_old = [];
end

% NaN in labels (solver empty) -> 0
labels_81(isnan(labels_81)) = 0;
labels_81 = double(labels_81);

if isempty(X_old)
    X_cells = X_cells_81;
    Y_labels = labels_81;
else
    X_cells = cat(3, X_old, X_cells_81);
    Y_labels = [Y_old; labels_81];
end

save(mat_path, 'X_cells', 'Y_labels');
fprintf('Appended 81 cells. Dataset size: %d\n', numel(Y_labels));
