% PlotFeatures  Load trained model and visualize CNN features (run with addpath to src/cnn and utils).
% Requires: MnistConv.mat in current dir or models/

root = pwd;
matPath = fullfile(root, 'models', 'MnistConv.mat');
if ~isfile(matPath)
    matPath = fullfile(root, 'MnistConv.mat');
end
load(matPath);
if exist('X_sample', 'var')
    X = X_sample;
end

k = 4;
x = X(:, :, k);
y1 = Conv(x, W1);
y2 = ReLU(y1);
y3 = Pool(y2);
y4 = reshape(y3, [], 1);
v5 = W5*y4;
y5 = ReLU(v5);
v = Wo*y5;
y = Softmax(v);

figure(1);
display_network(x(:));
title('Input Image')

convFilters = zeros(9*9, 20);
for i = 1:20
    filter = W1(:, :, i);
    convFilters(:, i) = filter(:);
end
figure(2);
display_network(convFilters);
title('Convolution Filters')

fList = zeros(20*20, 20);
for i = 1:20
    f = y1(:, :, i);
    fList(:, i) = f(:);
end
figure(3);
display_network(fList);
title('Features [convolution]')

fList = zeros(20*20, 20);
for i = 1:20
    f = y2(:, :, i);
    fList(:, i) = f(:);
end
figure(4);
display_network(fList);
title('Features [convolution + ReLU]')

fList = zeros(10*10, 20);
for i = 1:20
    f = y3(:, :, i);
    fList(:, i) = f(:);
end
figure(5);
display_network(fList);
title('Features [convolution + ReLU + MeanPool]')
