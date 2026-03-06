function sudoku = ImagePrepMethod2(filename)

%clear variables
%close all
%clc 




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
            c=[i,k,sudokoMatrix(i,k)]
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
            b=[i,k,sudokoMatrix(i,k)]
            c=[b;c]
        end
    end
end

% clue puzzle (encypter)
c

%% the 9 X 9 presentation
A = NaN(9);

for i = 1:size(c,1)
    A(c(i,1),c(i,2)) = c(i,3);
end
A
%%

%The board representation
drawSudoku2(c)


%the ground Truth
%type sudokuEngine2
%S = sudokuEngine2(c);
%drawSudoku(S);

%the backtracking solution
y = solveSudoku(A);
y




end
