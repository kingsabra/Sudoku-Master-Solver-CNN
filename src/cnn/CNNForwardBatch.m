function [preds, probs] = CNNForwardBatch(cells, W1, W5, Wo)
% CNNForwardBatch  Run CNN forward on 81 cell images at once (Phase 4 batch inference).
%   cells: [28 x 28 x 81] double.
%   preds: [81 x 1] predicted class 1-9 or 10 (empty); caller maps 10 -> 0 for board.
%   probs: [81 x 10] Softmax probabilities per cell.

N = size(cells, 3);
y4_stack = zeros(2000, N);

for k = 1:N
    x = cells(:, :, k);
    y1 = Conv(x, W1);
    y2 = ReLU(y1);
    y3 = Pool(y2);
    y4_stack(:, k) = reshape(y3, [], 1);
end

v5 = W5 * y4_stack;           % 100 x N
y5 = max(0, v5);
v = Wo * y5;                  % 10 x N

probs = zeros(10, N);
for k = 1:N
    probs(:, k) = Softmax(v(:, k));
end
probs = probs';

[~, preds] = max(probs, [], 2);   % 81 x 1, values 1-10
% Map 10 -> 0 for "empty" in board
preds(preds == 10) = 0;
