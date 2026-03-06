function [S,eflag] = sudokuEngine2(B)
% sudokuEngine2  Solve Sudoku via binary integer programming (intlinprog).
%   B: 9x9 matrix or Nx3 [row, col, value]. Returns solution S and exit flag.
%   Copyright 2014 The MathWorks, Inc.

if isequal(size(B),[9,9])
    [SM,SN] = meshgrid(1:9);
    B = [SN(:),SM(:),B(:)];
    [rrem,~] = find(B(:,3) == 0);
    B(rrem,:) = [];
end

if size(B,2) ~= 3 || length(size(B)) > 2
    error('The input matrix must be N-by-3 or 9-by-9')
end

if sum([any(B ~= round(B)),any(B < 1),any(B > 9)])
    error('Entries must be integers from 1 to 9')
end

N = 9^3;
M = 4*9^2;
Aeq = zeros(M,N);
beq = ones(M,1);
f = (1:N)';
lb = zeros(9,9,9);
ub = lb+1;

counter = 1;
for j = 1:9
    for k = 1:9
        Astuff = lb;
        Astuff(1:end,j,k) = 1;
        Aeq(counter,:) = Astuff(:)';
        counter = counter + 1;
    end
end

for i = 1:9
    for k = 1:9
        Astuff = lb;
        Astuff(i,1:end,k) = 1;
        Aeq(counter,:) = Astuff(:)';
        counter = counter + 1;
    end
end

for U = 0:3:6
    for V = 0:3:6
        for k = 1:9
            Astuff = lb;
            Astuff(U+(1:3),V+(1:3),k) = 1;
            Aeq(counter,:) = Astuff(:)';
            counter = counter + 1;
        end
    end
end

for i = 1:9
    for j = 1:9
        Astuff = lb;
        Astuff(i,j,1:end) = 1;
        Aeq(counter,:) = Astuff(:)';
        counter = counter + 1;
    end
end

for i = 1:size(B,1)
    lb(B(i,1),B(i,2),B(i,3)) = 1;
end

intcon = 1:N;
[x,~,eflag] = intlinprog(f,intcon,[],[],Aeq,beq,lb,ub);

if eflag > 0
    x = reshape(x,9,9,9);
    x = round(x);
    y = ones(size(x));
    for k = 2:9
        y(:,:,k) = k;
    end
    S = x.*y;
    S = sum(S,3);
else
    S = [];
end
