%loading the MNIST ROW binary data then 
Images = LoadMINSTImg('MNIST/t10k-images.idx3-ubyte');
%reshape the images to be 28 * 28 pixels
Images = reshape(Images, 28, 28, []);

%loading the assigned labels of the images
Labels = LoadMINSTLabel('MNIST/t10k-labels.idx1-ubyte');
Labels(Labels == 0) = 10;


rng(1);

%init weights
W1 = 1e-2*randn([9 9 20]);
W5 = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(360 + 2000);
Wo = (2*rand( 10, 100) - 1) * sqrt(6) / sqrt( 10 + 100);

%segemnting the data set (only first 8000)
X = Images(:, :, 1:8000);
%segmenting only the first 8000 labels
D = Labels(1:8000);

%displaying the training sample
%for i=1:length(D)
%    img = X(:, :, i);
%    imshow(img)
%end

%specifiyng a training option
for epoch = 1:72 
    epoch
    %passing the sole dataset each epoch 
    [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D);
end 


%saving the trained model
save('MnistConv.mat');

%segmenting the tetsing dataset 
X = Images(:, :, 8001:10000);
%segmneting the 
D = Labels(8001:10000);
acc = 0;
N = length(D);


%loop through all labels (images)
for k = 1:N
    %taking img by img and saving it to x
    %figure(9);
    x = X(:, :, k);
    %imshow(x)
    %pause
    %applying the CNN to the image
     y1 = Conv(x, W1);
    y2 = ReLU(y1);
    y3 = Pool(y2);
    y4 = reshape(y3, [], 1);
    v5 = W5*y4;
    y5 = ReLU(v5);
    v = Wo*y5;
    y = Softmax(v);
    
    % checking the index of valid cases with the orginal picture 
    [~, i] = max(y);
    if i == D(k)
        %update acc once a match is found
        acc = acc + 1;
    end
    
end
    %convert the acc to percentage
    acc = acc/N;
    %display acc
    fprintf('Accuracy is %f\n', acc);
    



















