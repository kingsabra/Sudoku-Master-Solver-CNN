function images = LoadMNISTImg(filename)
% LoadMNISTImg  Load MNIST images from IDX binary format.
%   images = LoadMNISTImg(filename) returns images as [rows*cols x numImages] double in [0,1].

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename]);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename]);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows   = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols   = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char=>unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images, [2 1 3]);
fclose(fp);

images = reshape(images, size(images,1)*size(images,2), size(images,3));
images = double(images) / 255;
