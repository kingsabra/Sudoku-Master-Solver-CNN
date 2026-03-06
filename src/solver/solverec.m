function [res, solved, noSolutionFound] = solverec(board, emptyInd, ind, solved)
% solverec  Recursive backtracking helper for solveSudoku.

if nargin < 4
    solved = false;
end

noSolutionFound = false;

if ind > numel(emptyInd)
    solved = true;
end

if solved
    res = board;
    return;
end

num  = emptyInd(ind);
col  = ceil(num / 9);
row  = num - (col - 1) * 9;

cd  = getCandidates(board, row, col);
ncd = numel(cd);

if ncd == 0
    noSolutionFound = true;
else
    for k = 1:ncd
        board(num) = cd(k);
        [res, solved, noSolutionFound] = solverec(board, emptyInd, ind+1, solved);
        if solved
            return;
        end
        if noSolutionFound
            board(num) = NaN;
        end
    end
    % All candidates tried and none led to a solution
    noSolutionFound = true;
end

if noSolutionFound
    board(num) = NaN;
    res = board;
    return;
end
end
