import numpy as np
from numpy import float64
from numpy.typing import NDArray

def jacobi_iteration_method(
    coefficient_matrix: NDArray[float64],
    constant_matrix: NDArray[float64],
    init_val: list[float],
    iterations: int,
) -> list[float]:

    rows1, cols1 = coefficient_matrix.shape
    rows2, cols2 = constant_matrix.shape
    
    if rows1 != cols1:
        msg = f"Coefficient matrix dimensions must be nxn but received {rows1}x{cols1}"
        raise ValueError(msg)

    if cols2 != 1:
        msg = f"Constant matrix must be nx1 but received {rows2}x{cols2}"
        raise ValueError(msg)

    if rows1 != rows2:
        msg = (
            "Coefficient and constant matrices dimensions must be nxn and nx1 but "
            f"received {rows1}x{cols1} and {rows2}x{cols2}"
        )
        raise ValueError(msg)

    if len(init_val) != rows1:
        msg = (
            "Number of initial values must be equal to number of rows in coefficient "
            f"matrix but received {len(init_val)} and {rows1}"
        )
        raise ValueError(msg)

    if iterations <= 0:
        raise ValueError("Iterations must be at least 1")

    table: NDArray[float64] = np.concatenate(
        (coefficient_matrix, constant_matrix), axis=1
    )
    
    rows, cols = table.shape

    strictly_diagonally_dominant(table)
    

    denominator = np.diag(coefficient_matrix)

    val_last = table[:, -1]

    masks = ~np.eye(coefficient_matrix.shape[0], dtype=bool)

    no_diagonals = coefficient_matrix[masks].reshape(-1, rows - 1)
    
    i_row, i_col = np.where(masks)
    ind = i_col.reshape(-1, rows - 1)

    for _ in range(iterations):
        arr = np.take(init_val, ind)
        sum_product_rows = np.sum((-1) * no_diagonals * arr, axis=1)
        new_val = (sum_product_rows + val_last) / denominator
        init_val = new_val

    return new_val.tolist()

def strictly_diagonally_dominant(table: NDArray[float64]) -> bool:

    rows, cols = table.shape

    is_diagonally_dominant = True

    for i in range(rows):
        total = 0
        for j in range(cols - 1):
            if i == j:
                continue
            else:
                total += table[i][j]

        if table[i][i] <= total:
            raise ValueError("Coefficient matrix is not strictly diagonally dominant")

    return is_diagonally_dominant

