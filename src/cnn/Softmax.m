function y = Softmax(x)
% Softmax  Softmax over vector x.

ex = exp(x);
y = ex / sum(ex);
