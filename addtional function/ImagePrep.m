function I = ImagePrep(filename)

% read and display
I_Orginal = imread(filename);
figure(1);
imshow(I_Orginal)
%pause(1)

%defining the the ground truth of the image
%Truth_board = ImagePrepMethod2(filename);
%Truth_board

%%%%
%load image
sudoku=imread(filename);

%rgb to gray
Grayscale=rgb2gray(sudoku);

%binarizing image
BW = imbinarize(Grayscale,'adaptive','ForegroundPolarity','dark','Sensitivity',0.4);

%extracting largest connected component
BW2 = bwareafilt(imcomplement(BW),1,'largest');

%labeling largest connected component
label=bwlabel(BW2);

%assigning label pixels coordinates to a matrix
[row,colom]=find(label==1);

%image cropping
BW3=BW(min(row):max(row),min(colom):max(colom));
BW4=BW2(min(row):max(row),min(colom):max(colom));
BW5=logical(imcomplement(BW3)-BW4);

%centroids of sudoku matrix
s = regionprops(imcomplement(BW2),'centroid');
centroids = cat(1, s.Centroid);

%image displaying with centroids
figure(12);
imshow(BW3);
hold on
plot(centroids(:,1),centroids(:,2),'b*')
hold off
[width,height] = size(BW5);%calculating number of coulomns and rows
sub_width= floor(width/9);%cell width
sub_height= floor(height/9);%cell length
sudokoMatrix = [];%matrix of sudoku

Xi=1;
for r = 1:9
    Yi=1;
    for c = 1:9
        BW6=BW5(Xi:r*sub_height,Yi:c*sub_width);
        Yi=c*sub_width;%shifting in width direction
        %OCR of each cell
        l=ocr(BW6, 'CharacterSet', '0123456789', 'TextLayout','Block');
         T=double(l.Text);
         lenT=size(T);
           if lenT(2)~=0
            switch T(1)
                case 49
                    sudokoMatrix(r,c)=1; 
                case 50
                    sudokoMatrix(r,c)=2;
                case 51
                    sudokoMatrix(r,c)=3;
                case 52
                    sudokoMatrix(r,c)=4;
                case 53
                    sudokoMatrix(r,c)=5;
                case 54
                    sudokoMatrix(r,c)=6;
                case 55
                    sudokoMatrix(r,c)=7;
                case 56
                    sudokoMatrix(r,c)=8;
                case 57 
                    sudokoMatrix(r,c)=9;
                otherwise
                    sudokoMatrix(r,c)=0;
             end
            end
          if lenT(2)==0
               sudokoMatrix(r,c)=0;
          end
    end
    Xi=r*sub_height;%shifting in length
end

clc;flag=0;
for i=1:9
    for k=1:9
        if sudokoMatrix(i,k)>0
            c=[i,k,sudokoMatrix(i,k)];
                flag=1; 
           break
           end
       end
     if(flag==1)
          break 
     end
end
 
for i=1:9
    for k=1:9
        if sudokoMatrix(i,k)>0
            b=[i,k,sudokoMatrix(i,k)];
            c=[b;c];
        end
    end
end

% clue puzzle (encypter)
c;

% the 9 X 9 presentation
Truth_board = zeros(9);

for i = 1:size(c,1)
    Truth_board(c(i,1),c(i,2)) = c(i,3);
end
Truth_board

%%%%




% convert to gray scale
I = rgb2gray(I_Orginal);

% applying gaussian blur
I_blur = imgaussfilt(I, 1);
%I_blur = edge(I, 'canny');
figure(2);
imshow(I_blur);
%pause(1)

%the binary inversion
I_RbIn = I_blur <= 180;
%I_RbIn = imbinarize(I_blur);
%I_RbIn = imbinarize(I_blur,'adaptive','ForegroundPolarity','dark','Sensitivity',0.45);
figure(3);
imshow(I_RbIn);


%PHT (Hou)
[H,theta,rho] = hough(I_RbIn);
figure(4);
imshow(imadjust(rescale(H)),[],...
       'XData',theta,...
       'YData',rho,...
       'InitialMagnification','fit');
xlabel('\theta (degrees)')
ylabel('\rho')
axis on
axis normal 
hold on
colormap(gca,hot)
% finding the peaks:
P = houghpeaks(H,20,'threshold',ceil(0.3*max(H(:))));
x = theta(P(:,2));
y = rho(P(:,1));
plot(x,y,'s','color','red');
% finding lines
lines = houghlines(I_RbIn,theta,rho,P,'FillGap',5,'MinLength',7);
%plotting lines on the image
figure(5); 
imshow(I_RbIn), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
% highlight the longest line segment
plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','red');


%write out the picture after the lines.
%imwrite(I_RbIn,'grid.jpg')

%the picture info:I_RbIn
figure(10)
new = imresize(I_RbIn, [270 270]);
imshow(new);

width_img2 = size(new, 2);
length_img2 = size(new, 1);

cell_width = floor(width_img2 / 9);
cell_height = floor(length_img2 / 9);

x_1 = 0;
x_2 = 0;
y_1 = 0;
y_2 = 0;

Cell_positions = [];

for i=1:9
    y_2 = y_1 + cell_height;
    x_1 = 1;
    for j=1:9
        x_2 = x_1 + cell_width;
        current_cell = [x_1, x_2, y_1, y_2];
        Cell_positions = [Cell_positions;current_cell];
        x_1 = x_2;
    end
    y_1 = y_2;
end

Cell_positions;


%elements by row
%Cell_positions(1,:)
All_cells = size(Cell_positions,1);
%empty = imread("empty_template.jpg")
%
all_numbers = [];

thersh = 34;



%%%%%
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
for epoch = 1:4 
    epoch
    %passing the sole dataset each epoch 
    [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D);
end 


%saving the trained model
save('MnistConv.mat');


%%%%
for i=1:All_cells
    figure(11)
    %cell = new(Cell_positions(i,3)+2:Cell_positions(i,4)-3, Cell_positions(i,1)+2:Cell_positions(i,2)-3);
    cell = new(Cell_positions(i,3)+1:Cell_positions(i,4)-2, Cell_positions(i,1)+1:Cell_positions(i,2)-2);
    
   %Cany = edge(cell, 'canny', [thresh_lower, thresh_upper] )
   % meanCell = mean(Cany)
    %norm2 = normxcorr2(cell,empty);
    %maxVale = max(norm2(:))
    %imwrite(cell,"empty_template.jpg")
    
    cell_center_box = new(Cell_positions(i,3)+7:Cell_positions(i,4)-6, Cell_positions(i,1)+7:Cell_positions(i,2)-6);
    pixelsum_1 = bwarea(cell_center_box);
    pixelsum_2 = sum(cell_center_box(:));
    
    if pixelsum_2 > thersh
        %it is a number 
        y1 = Conv(cell, W1);
        y2 = ReLU(y1);
        y3 = Pool(y2);
        y4 = reshape(y3, [], 1);
        v5 = W5*y4;
        y5 = ReLU(v5);
        v = Wo*y5;
        y = Softmax(v);
    
    % checking the index of valid cases with the orginal picture 
        [~, i] = max(y);
        all_numbers = [all_numbers; i];
        figure(18);
        imshow(cell);
     %map it as value of i
        %imshow(cell)
        %pause;
    else
        all_numbers = [all_numbers; 0];
        
    end 

    
end

%
board = zeros(9);
element = 1;
for i=1:9
    for j=1:9
        board(i,j) = all_numbers(element,1);
        element = element + 1;
    end
end

%board

%%%

%%accurcy measurement
similarity = 0;
Not_Similar_pos = [];
for i=1:9
    for j=1:9
        if board(i,j) == Truth_board(i,j)
            similarity = similarity + 1;
            if board(i,j) == 0;
                board(i,j) = NaN;
            end
            
        else
            Not_Similar_pos = [Not_Similar_pos; [i,j]];
            board(i,j) = Truth_board(i,j);
        end
    end
end

acc = (similarity/81)*100
board

%%%%
%finding the solution
%the backtracking solution
Solution = solveSudoku(board);
Solution

%%%%



%running it over all images:


 





end
