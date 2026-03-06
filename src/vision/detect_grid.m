function [cells_81, Truth_board, new] = detect_grid(I, cell_sz)
% detect_grid  Extract 81 Sudoku cell images and OCR ground truth from a single image (Phase 5).
%   I: RGB or grayscale image (Sudoku photo).
%   cell_sz: output cell size (default 28).
%   cells_81: [cell_sz x cell_sz x 81] double in [0,1].
%   Truth_board: [9 x 9] from OCR (0 = empty).
%   new: [270 x 270] preprocessed grid image (for debugging/visualization).
%
% This wraps the current hand-crafted pipeline (crop, binarize, Hough-style grid).
% A learned grid detector could replace the body and keep this interface.

if nargin < 2
    cell_sz = 28;
end

if size(I, 3) == 3
    Grayscale = rgb2gray(I);
else
    Grayscale = I;
end

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

Truth_board = sudokoMatrix;

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

cells_81 = zeros(cell_sz, cell_sz, 81);
for ic = 1:81
    cell_img = new(Cell_positions(ic,3)+1:Cell_positions(ic,4)-2, Cell_positions(ic,1)+1:Cell_positions(ic,2)-2);
    cell_img = imresize(cell_img, [cell_sz cell_sz]);
    cell_img = double(cell_img);
    if max(cell_img(:)) > min(cell_img(:))
        cell_img = (cell_img - min(cell_img(:))) / (max(cell_img(:)) - min(cell_img(:)));
    end
    cells_81(:, :, ic) = cell_img;
end
