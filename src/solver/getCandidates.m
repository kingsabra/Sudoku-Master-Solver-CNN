function candidates = getCandidates(board, row, col)
% getCandidates  Return possible digits (1-9) for (row,col) given Sudoku constraints.

subrow = board(row, :);
subcol = board(:, col);

subSquareRow = (1:3) + 3*(ceil(row/3) - 1);
subSquareCol = (1:3) + 3*(ceil(col/3) - 1);
subBoard = board(subSquareRow, subSquareCol);
subBoard = subBoard(:);

refval = 1:9;
cdrow = setdiff(refval, subrow);
cdcol = setdiff(refval, subcol);
cdsqr = setdiff(refval, subBoard);

candidates = intersect(intersect(cdrow, cdcol), cdsqr);
