% generate_synthetic_cells  Generate synthetic Sudoku-style digit cell images (28x28) and save to data/sudoku_cells.
% Mimics printed/Sudoku look: dark digits on light background, optional noise/blur.
% Run from repository root.

root = pwd;
addpath(fullfile(root, 'utils'));

CELL_SZ = 28;
SAMPLES_PER_DIGIT = 80;   % 1-9
SAMPLES_EMPTY = 80;       % empty cells
data_dir = fullfile(root, 'data', 'sudoku_cells');
if ~isfolder(data_dir)
    mkdir(data_dir);
end

all_X = [];
all_Y = [];

for digit = 0:9
    n = SAMPLES_PER_DIGIT;
    if digit == 0
        n = SAMPLES_EMPTY;
    end
    for s = 1:n
        im = zeros(CELL_SZ, CELL_SZ);
        if digit > 0
            % Render digit in center (Sudoku-style: dark on light)
            hfig = figure('Visible', 'off', 'Color', 'white');
            ha = axes('Parent', hfig, 'Position', [0 0 1 1], 'Visible', 'off');
            text(ha, 0.5, 0.5, num2str(digit), 'FontSize', 22, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', [0 0 0]);
            xlim(ha, [0 1]);
            ylim(ha, [0 1]);
            F = getframe(hfig);
            close(hfig);
            patch = double(rgb2gray(F.cdata)) / 255;
            patch = 1 - patch;  % dark digit on light
            patch = imresize(patch, [CELL_SZ CELL_SZ]);
            im = patch;
        end
        % Normalize to [0,1]
        if max(im(:)) > min(im(:))
            im = (im - min(im(:))) / (max(im(:)) - min(im(:)));
        end
        % Augment: brightness, blur, small noise (like real photos)
        im = im * (0.85 + 0.3*rand);
        im = min(1, max(0, im + 0.05*(rand(CELL_SZ,CELL_SZ)-0.5)));
        if rand < 0.5
            im = imgaussfilt(im, 0.3 + 0.4*rand);
        end
        all_X = cat(3, all_X, im);
        all_Y = [all_Y; digit];
    end
end

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
fprintf('Generated %d synthetic cells. Total in dataset: %d\n', 9*SAMPLES_PER_DIGIT + SAMPLES_EMPTY, numel(Y_labels));
