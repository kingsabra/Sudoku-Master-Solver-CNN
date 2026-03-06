function safe = checkSudoku(board, row, col, num)
% checkSudoku  Return true if placing num at (row,col) is valid.

subrow = board(row, :);
subcol = board(:, col);

subSquareRow = (1:3) + 3*(ceil(row/3)-1);
subSquareCol = (1:3) + 3*(ceil(col/3)-1);
subBoard = board(subSquareRow, subSquareCol);
subBoard = subBoard(:);

if any(subrow == num) || any(subcol == num) || any(subBoard == num)
    safe = false;
else
    safe = true;
end
