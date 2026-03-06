% train_model  Train CNN on MNIST once and save weights to models/MnistConv.mat.
% Run from repository root (e.g. cd Sudoku-Master-Solver-CNN; scripts/train_model).

root = pwd;
addpath(fullfile(root, 'src', 'cnn'));

mnistDir = fullfile(root, 'MNIST');
imgPath  = fullfile(mnistDir, 't10k-images.idx3-ubyte');
lblPath  = fullfile(mnistDir, 't10k-labels.idx1-ubyte');

if ~isfile(imgPath) || ~isfile(lblPath)
    error('MNIST files not found. Place t10k-images.idx3-ubyte and t10k-labels.idx1-ubyte in MNIST/');
end

fprintf('Loading MNIST...\n');
Images = LoadMNISTImg(imgPath);
Images = reshape(Images, 28, 28, []);
Labels = LoadMNISTLabel(lblPath);
Labels(Labels == 0) = 10;

rng_seed(1);

W1 = 1e-2 * randn([9 9 20]);
W5 = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(360 + 2000);
Wo = (2*rand(10, 100) - 1) * sqrt(6) / sqrt(10 + 100);

X = Images(:, :, 1:8000);
D = Labels(1:8000);

fprintf('Training CNN (4 epochs)...\n');
for epoch = 1:4
    fprintf('  Epoch %d\n', epoch);
    [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D);
end

modelDir = fullfile(root, 'models');
if ~isfolder(modelDir)
    mkdir(modelDir);
end
outPath = fullfile(modelDir, 'MnistConv.mat');
X_sample = Images(:, :, 1:100);  % for PlotFeatures
save(outPath, 'W1', 'W5', 'Wo', 'X_sample');
fprintf('Saved model to %s\n', outPath);
