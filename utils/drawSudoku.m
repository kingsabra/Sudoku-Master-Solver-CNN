function drawSudoku(B)
% drawSudoku  Draw the Sudoku board (9x9 or Nx3 clue list).
%   Copyright 2014 The MathWorks, Inc.

figure;
hold on;
axis off;
axis equal
rectangle('Position',[0 0 9 9],'LineWidth',3,'Clipping','off')
rectangle('Position',[3,0,3,9],'LineWidth',2)
rectangle('Position',[0,3,9,3],'LineWidth',2)
rectangle('Position',[0,1,9,1],'LineWidth',1)
rectangle('Position',[0,4,9,1],'LineWidth',1)
rectangle('Position',[0,7,9,1],'LineWidth',1)
rectangle('Position',[1,0,1,9],'LineWidth',1)
rectangle('Position',[4,0,1,9],'LineWidth',1)
rectangle('Position',[7,0,1,9],'LineWidth',1)

if size(B,2) == 9
    [SM,SN] = meshgrid(1:9);
    B = [SN(:),SM(:),B(:)];
end

for ii = 1:size(B,1)
    text(B(ii,2)-0.5, 9.5-B(ii,1), num2str(B(ii,3)));
end
hold off
