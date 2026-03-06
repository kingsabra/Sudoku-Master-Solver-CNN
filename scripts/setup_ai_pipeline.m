% setup_ai_pipeline  One-shot setup for the AI learning pipeline (Phases 1–5).
% Run from repository root. Does: generate synthetic data, build dataset from images (if any),
% train base model (if needed), fine-tune on Sudoku cells.
%
% Prerequisites: MNIST files in MNIST/ for base training. images/ optional (.jpg/.jpeg/.png).

root = pwd;
addpath(fullfile(root, 'scripts'));

fprintf('=== Phase 1: Build Sudoku cell dataset ===\n');
generate_synthetic_cells;
imgs = [dir(fullfile(root, 'images', '*.jpg')); dir(fullfile(root, 'images', '*.jpeg')); dir(fullfile(root, 'images', '*.png'))];
if isfolder(fullfile(root, 'images')) && ~isempty(imgs)
    build_dataset_from_images;
else
    fprintf('No images found in images/; using synthetic data only.\n');
end

fprintf('\n=== Phase 2: Train base model (if needed) ===\n');
if ~isfile(fullfile(root, 'models', 'MnistConv.mat'))
    if isfile(fullfile(root, 'MNIST', 't10k-images.idx3-ubyte'))
        train_model;
    else
        fprintf('MNIST not found. Run train_model after placing MNIST files.\n');
    end
else
    fprintf('Base model exists.\n');
end

fprintf('\n=== Phase 2: Fine-tune on Sudoku cells ===\n');
train_sudoku_cells;

fprintf('\n=== Done. Run Test_Solver or scripts/run_demo to process images. ===\n');
