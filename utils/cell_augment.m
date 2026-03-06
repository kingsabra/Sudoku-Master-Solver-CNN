function Xout = cell_augment(X, do_rotate, do_brightness, do_blur)
% cell_augment  Augment a 28x28 cell image (or 28x28xN) for training.
%   X: 28x28 or 28x28xN double in [0,1].
%   do_rotate, do_brightness, do_blur: optional flags (default true).
%   Returns same size as X.

if nargin < 2, do_rotate = true; end
if nargin < 3, do_brightness = true; end
if nargin < 4, do_blur = true; end

Xout = X;
single = (ndims(X) == 2);
if single
    Xout = reshape(Xout, 28, 28, 1);
end
N = size(Xout, 3);

for i = 1:N
    im = Xout(:, :, i);
    if do_rotate
        angle = (rand - 0.5) * 20;  % ±10 deg
        im = imrotate(im, angle, 'bilinear', 'crop');
        im = im(1:28, 1:28);
    end
    if do_brightness
        g = 0.7 + 0.6 * rand;  % 0.7..1.3
        im = min(1, max(0, im * g));
    end
    if do_blur
        if rand < 0.5
            im = imgaussfilt(im, 0.5 + 0.5*rand);
        end
    end
    Xout(:, :, i) = im;
end

if single
    Xout = Xout(:, :, 1);
end
