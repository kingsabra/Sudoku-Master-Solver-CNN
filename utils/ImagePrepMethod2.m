function sudoku = ImagePrepMethod2(filename)
% ImagePrepMethod2  Load Sudoku image, OCR to 9x9, optionally draw and solve.
% Requires addpath to src/solver and utils (for drawSudoku).

sudoku = imread(filename);
Grayscale = rgb2gray(sudoku);
BW = imbinarize(Grayscale, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.4);
BW2 = bwareafilt(imcomplement(BW), 1, 'largest');
label = bwlabel(BW2);
[row, col] = find(label == 1);

BW3 = BW(min(row):max(row), min(col):max(col));
BW4 = BW2(min(row):max(row), min(col):max(col));
BW5 = logical(imcomplement(BW3) - BW4);

[width, height] = size(BW5);
sub_width = floor(width/9);
sub_height = floor(height/9);
sudokoMatrix = zeros(9, 9);

Xi = 1;
for r = 1:9
    Yi = 1;
    for c = 1:9
        BW6 = BW5(Xi:r*sub_height, Yi:c*sub_width);
        Yi = c*sub_width;
        l = ocr(BW6, 'CharacterSet', '0123456789', 'TextLayout', 'Block');
        T = double(l.Text);
        lenT = size(T);
        if lenT(2) ~= 0
            switch T(1)
                case 49, sudokoMatrix(r,c) = 1;
                case 50, sudokoMatrix(r,c) = 2;
                case 51, sudokoMatrix(r,c) = 3;
                case 52, sudokoMatrix(r,c) = 4;
                case 53, sudokoMatrix(r,c) = 5;
                case 54, sudokoMatrix(r,c) = 6;
                case 55, sudokoMatrix(r,c) = 7;
                case 56, sudokoMatrix(r,c) = 8;
                case 57, sudokoMatrix(r,c) = 9;
                otherwise, sudokoMatrix(r,c) = 0;
            end
        else
            sudokoMatrix(r,c) = 0;
        end
    end
    Xi = r*sub_height;
end

c = [];
for i = 1:9
    for k = 1:9
        if sudokoMatrix(i,k) > 0
            c = [c; i, k, sudokoMatrix(i,k)];
        end
    end
end

A = NaN(9);
for i = 1:size(c,1)
    A(c(i,1), c(i,2)) = c(i,3);
end

drawSudoku(c);

y = solveSudoku(A);
sudoku = y;
