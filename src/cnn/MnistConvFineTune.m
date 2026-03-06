function [W1, W5, Wo] = MnistConvFineTune(W1, W5, Wo, X, D, alpha, beta)
% MnistConvFineTune  Train/fine-tune CNN with backpropagation (configurable learning rate).
%   X: [28 x 28 x N] cell images, D: [N x 1] labels 0=empty, 1-9=digit.
%   Labels 0 are mapped to class index 10 for the softmax target.
%   alpha, beta: learning rate and momentum (default 0.001, 0.95 for fine-tuning).

if nargin < 6, alpha = 0.001; end
if nargin < 7, beta = 0.95; end

% Map 0 -> 10 for target index
D = D(:);
D(D == 0) = 10;

momentum1 = zeros(size(W1));
momentum5 = zeros(size(W5));
momentumo = zeros(size(Wo));

N = length(D);
bsize = min(50, N);
blist = 1:bsize:(N - bsize + 1);
if isempty(blist)
    blist = 1;
    bsize = N;
end

for batch = 1:length(blist)
    dW1 = zeros(size(W1));
    dW5 = zeros(size(W5));
    dWo = zeros(size(Wo));

    begin = blist(batch);
    last = min(begin + bsize - 1, N);
    bsize_act = last - begin + 1;

    for k = begin:last
        x = X(:, :, k);
        y1 = Conv(x, W1);
        y2 = ReLU(y1);
        y3 = Pool(y2);
        y4 = reshape(y3, [], 1);
        v5 = W5 * y4;
        y5 = ReLU(v5);
        v  = Wo * y5;
        y  = Softmax(v);

        d = zeros(10, 1);
        d(D(k), 1) = 1;

        e = d - y;
        delta = e;
        e5 = Wo' * delta;
        delta5 = (y5 > 0) .* e5;
        e4 = W5' * delta5;
        e3 = reshape(e4, size(y3));
        e2 = zeros(size(y2));
        W3 = ones(size(y2)) / (2*2);

        for c = 1:20
            e2(:, :, c) = kron(e3(:, :, c), ones([2 2])) .* W3(:, :, c);
        end

        delta2 = (y2 > 0) .* e2;
        delta1_x = zeros(size(W1));
        for c = 1:20
            delta1_x(:, :, c) = conv2(x(:, :), rot90(delta2(:, :, c), 2), 'valid');
        end

        dW1 = dW1 + delta1_x;
        dW5 = dW5 + delta5 * y4';
        dWo = dWo + delta * y5';
    end

    dW1 = dW1 / bsize_act;
    dW5 = dW5 / bsize_act;
    dWo = dWo / bsize_act;

    momentum1 = alpha*dW1 + beta*momentum1;
    W1 = W1 + momentum1;
    momentum5 = alpha*dW5 + beta*momentum5;
    W5 = W5 + momentum5;
    momentumo = alpha*dWo + beta*momentumo;
    Wo = Wo + momentumo;
end
