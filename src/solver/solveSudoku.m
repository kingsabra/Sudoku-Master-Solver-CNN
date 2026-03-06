function solvedBoard = solveSudoku(board)
% solveSudoku  Solve 9x9 Sudoku via recursive backtracking.
%   board: 9x9 with 1-9 for clues, NaN for empty. Returns filled 9x9.

emptyInd = find(isnan(board));
solvedBoard = solverec(board, emptyInd, 1);
