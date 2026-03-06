function rng_seed(x)
% rng_seed  Set random seed for reproducibility (avoids shadowing built-in rng).
%   rng_seed(x) sets rand and randn seed to x.

randn('seed', x);
rand('seed', x);
