function safe = checkSudoku(board, row, col, num)
    subrow = board(row, :);
    subcol = board(:, col);

    subSquareRow = (1:3) + 3*(ceil(row/3)-1) ; 
    subSquareCol = (1:3) + 3*(ceil(col/3)-1) ;

    subBoard = board( subSquareRow , subSquareCol );
    subBoard = subBoard(:) ; % Reshape into column vector (easier comparison)

    % This whole block can be replaced with the line described below
    if any(subrow == num) || any(subcol == num) || any(any(subBoard == num))
        safe = false;
    else
        safe = true;
    end

    % Note that since we are dealing with boolean, the "IF" check above could
    % be avoided and simply written as :

    % safe = ~( any(subrow == num) || any(subcol == num) || any(any(subBoard == num)) ) ;
end