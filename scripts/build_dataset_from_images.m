% build_dataset_from_images  Extract Sudoku cell images from photos in images/ and save to data/sudoku_cells.
% Labels come from OCR (ground truth). Optionally augment and add synthetic data.
% Run from repository root.

root = pwd;
addpath(fullfile(root, 'src', 'cnn'), fullfile(root, 'src', 'solver'), fullfile(root, 'utils'));

CELL_SZ = 28;
AUGMENT_PERCENT = 0.5;  % Phase 4: add augmented copies for this fraction of cells (0 = none)
img_dir = fullfile(root, 'images');
data_dir = fullfile(root, 'data', 'sudoku_cells');
if ~isfolder(data_dir)
    mkdir(data_dir);
end

exts = {'.jpg', '.jpeg', '.png'};
fg_imgs = [];
for e = 1:numel(exts)
    fg_imgs = [fg_imgs; dir(fullfile(img_dir, ['*', exts{e}]))]; %#ok<AGROW>
end
if isempty(fg_imgs)
    fprintf('No images found in images/. Add photos or run generate_synthetic_cells first.\n');
    X_cells = [];
    Y_labels = [];
    save(fullfile(data_dir, 'sudoku_cells.mat'), 'X_cells', 'Y_labels');
    return;
end

all_X = [];
all_Y = [];

% Stable ordering
[~, order] = sort({fg_imgs.name});
fg_imgs = fg_imgs(order);

for f = 1:numel(fg_imgs)
    fn = fullfile(img_dir, fg_imgs(f).name);
    fprintf('Processing %s\n', fn);
    I_FG = imread(fn);
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

    for ic = 1:81
        cell_img = new(Cell_positions(ic,3)+1:Cell_positions(ic,4)-2, Cell_positions(ic,1)+1:Cell_positions(ic,2)-2);
        cell_img = imresize(cell_img, [CELL_SZ CELL_SZ]);
        cell_img = double(cell_img);
        if max(cell_img(:)) > min(cell_img(:))
            cell_img = (cell_img - min(cell_img(:))) / (max(cell_img(:)) - min(cell_img(:)));
        end
        r = ceil(ic/9);
        c = ic - (r-1)*9;
        lab = sudokoMatrix(r, c);
        all_X = cat(3, all_X, cell_img);
        all_Y = [all_Y; lab];
        if AUGMENT_PERCENT > 0 && rand < AUGMENT_PERCENT
            aug = cell_augment(cell_img, true, true, true);
            all_X = cat(3, all_X, aug);
            all_Y = [all_Y; lab];
        end
    end
end

% Append to existing dataset if present
mat_path = fullfile(data_dir, 'sudoku_cells.mat');
if isfile(mat_path)
    ld = load(mat_path);
    if isfield(ld, 'X_cells') && isfield(ld, 'Y_labels') && ~isempty(ld.X_cells)
        all_X = cat(3, ld.X_cells, all_X);
        all_Y = [ld.Y_labels; all_Y];
    end
end

X_cells = all_X;
Y_labels = all_Y;
save(mat_path, 'X_cells', 'Y_labels');
fprintf('Saved %d cells to %s\n', numel(Y_labels), mat_path);
