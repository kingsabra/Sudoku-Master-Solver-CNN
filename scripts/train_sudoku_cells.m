% train_sudoku_cells  Fine-tune CNN on Sudoku cell dataset (Phase 2).
% Loads models/MnistConv.mat, trains on data/sudoku_cells/sudoku_cells.mat, saves models/MnistConv_sudoku.mat.
% Run from repository root. Run build_dataset_from_images and/or generate_synthetic_cells first.

root = pwd;
addpath(fullfile(root, 'src', 'cnn'), fullfile(root, 'utils'));

data_path = fullfile(root, 'data', 'sudoku_cells', 'sudoku_cells.mat');
if ~isfile(data_path)
    error('Run scripts/build_dataset_from_images and/or scripts/generate_synthetic_cells first.');
end

ld = load(data_path);
X = ld.X_cells;
D = ld.Y_labels;
N = numel(D);
if N < 50
    fprintf('Warning: only %d samples. Consider generate_synthetic_cells for more data.\n', N);
end

% Train/val split 80/20
rng(42);
idx = randperm(N);
nval = max(1, round(N * 0.2));
val_idx = idx(1:nval);
train_idx = idx(nval+1:end);
X_train = X(:, :, train_idx);
D_train = D(train_idx);
X_val = X(:, :, val_idx);
D_val = D(val_idx);

% Phase 4: data augmentation on training set (double with augmented copies)
addpath(fullfile(root, 'utils'));
n_train = size(X_train, 3);
X_aug = zeros(size(X_train,1), size(X_train,2), n_train);
for k = 1:n_train
    X_aug(:, :, k) = cell_augment(X_train(:, :, k), true, true, true);
end
X_train = cat(3, X_train, X_aug);
D_train = [D_train; D_train];
fprintf('After augmentation: %d training samples\n', numel(D_train));

% Load base model or init from scratch
model_path = fullfile(root, 'models', 'MnistConv.mat');
if isfile(model_path)
    fprintf('Loading base model from %s\n', model_path);
    load(model_path, 'W1', 'W5', 'Wo');
else
    fprintf('No base model; initializing weights.\n');
    rng_seed(1);
    W1 = 1e-2 * randn([9 9 20]);
    W5 = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(360 + 2000);
    Wo = (2*rand(10, 100) - 1) * sqrt(6) / sqrt(10 + 100);
end

% Fine-tune: small learning rate
alpha = 0.001;
beta = 0.95;
n_epochs = 10;

fprintf('Fine-tuning on %d samples (%d val)...\n', numel(D_train), numel(D_val));
for epoch = 1:n_epochs
    [W1, W5, Wo] = MnistConvFineTune(W1, W5, Wo, X_train, D_train, alpha, beta);
    % Validation accuracy
    acc = 0;
    D_val_10 = D_val;
    D_val_10(D_val_10 == 0) = 10;
    for k = 1:numel(D_val)
        x = X_val(:, :, k);
        y1 = Conv(x, W1);
        y2 = ReLU(y1);
        y3 = Pool(y2);
        y4 = reshape(y3, [], 1);
        v5 = W5 * y4;
        y5 = ReLU(v5);
        v = Wo * y5;
        y = Softmax(v);
        [~, pred] = max(y);
        if pred == D_val_10(k)
            acc = acc + 1;
        end
    end
    fprintf('  Epoch %d  val acc: %.2f%%\n', epoch, 100 * acc / numel(D_val));
end

out_path = fullfile(root, 'models', 'MnistConv_sudoku.mat');
if ~isfolder(fullfile(root, 'models'))
    mkdir(fullfile(root, 'models'));
end
save(out_path, 'W1', 'W5', 'Wo');
fprintf('Saved %s\n', out_path);
