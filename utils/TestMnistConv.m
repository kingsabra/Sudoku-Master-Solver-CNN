% TestMnistConv  Train CNN on MNIST and report test accuracy.
% Run from repo root with addpath('src/cnn', 'utils'). Prefer scripts/train_model for training.

root = pwd;
addpath(fullfile(root, 'src', 'cnn'));

Images = LoadMNISTImg(fullfile(root, 'MNIST', 't10k-images.idx3-ubyte'));
Images = reshape(Images, 28, 28, []);
Labels = LoadMNISTLabel(fullfile(root, 'MNIST', 't10k-labels.idx1-ubyte'));
Labels(Labels == 0) = 10;

rng_seed(1);

W1 = 1e-2*randn([9 9 20]);
W5 = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(360 + 2000);
Wo = (2*rand(10, 100) - 1) * sqrt(6) / sqrt(10 + 100);

X = Images(:, :, 1:8000);
D = Labels(1:8000);

for epoch = 1:72
    fprintf('Epoch %d\n', epoch);
    [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D);
end

save(fullfile(root, 'MnistConv.mat'), 'W1', 'W5', 'Wo');

X_test = Images(:, :, 8001:10000);
D_test = Labels(8001:10000);
acc = 0;
N = length(D_test);

for k = 1:N
    x = X_test(:, :, k);
    y1 = Conv(x, W1);
    y2 = ReLU(y1);
    y3 = Pool(y2);
    y4 = reshape(y3, [], 1);
    v5 = W5*y4;
    y5 = ReLU(v5);
    v = Wo*y5;
    y = Softmax(v);
    [~, pred] = max(y);
    if pred == D_test(k)
        acc = acc + 1;
    end
end

fprintf('Accuracy: %f\n', acc / N);
