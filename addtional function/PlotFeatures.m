%loding teh trained model
load('MnistConv.mat')

%deciding number to test (two)
k = 4;

% testing the network on only one number
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
%displaying input image only
display_network(x(:));
title('Input Images')

%init the filter matrix to be filler
convFilters = zeros(9*9, 20);
for i = 1:20
    %update each filter according the weight matrix  
    filter = W1(:, :, i);
    %saving teh filers in each colmn
    convFilters(:, i) = filter(:);
end

figure(2);
%display the ConvFilters
display_network(convFilters);
title('Convolution Filters')


fList = zeros(20*20, 20);
for i=1:20
    feature = y1(:, :, i);
    fList(:, i) = feature(:);
end

figure(3);
display_network(fList);
title('features [convolution]')

fList = zeros(20*20, 20);
for i=1:20
    feature = y2(:, :, i);
    fList(:, i) = feature(:);
end

figure(4);
display_network(fList);
title('features [convolution + ReLU]')

fList = zeros(10*10, 20);
for i=1:20
    feature = y3(:, :, i);
    fList(:, i) = feature(:);
end


figure(5);
display_network(fList);
title('features [convolution + ReLU+MeanPool]')
















