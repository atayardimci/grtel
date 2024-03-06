from beartype import beartype
import numpy as np
import pandas as pd


@beartype
def _get_laplacian(W: np.matrix) -> np.matrix:
    D = np.zeros(W.shape)
    for i in range(len(W)):
        D[i, i] = W[i].sum()
    L = D - W
    return L


@beartype
def _get_lambda_and_cut_indicator(
    weight_matrix: pd.DataFrame
) -> tuple[float, pd.Series]:
    W = np.asmatrix(weight_matrix)
    L = _get_laplacian(W)
    eigen_values, eigen_vectors = np.linalg.eig(L)

    # order eigenvalues and eigenvectors
    indices = eigen_values.argsort()
    eigen_values = eigen_values[indices]
    eigen_vectors = eigen_vectors[:, indices]

    lmbda = eigen_values[1]
    eigen_vector = np.array(eigen_vectors[:, 1]).reshape(-1)

    cut_indicator = pd.Series(
        np.sign(eigen_vector),
        index=weight_matrix.index
    )

    return lmbda, cut_indicator


@beartype
def _portfolio_cut(weight_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut the portfolio into two portfolios."""
    _, cut_indicator = _get_lambda_and_cut_indicator(weight_matrix)

    weight_matrix_one, weight_matrix_two = (
        weight_matrix.loc[cut_indicator < 0, cut_indicator < 0],
        weight_matrix.loc[cut_indicator > 0, cut_indicator > 0]
    )
    return weight_matrix_one, weight_matrix_two


@beartype
def portfolio_cuts(
    weight_matrix: pd.DataFrame,
    max_size: int = 15
) -> list[pd.DataFrame]:
    """Cut the portfolio into smaller portfolios."""
    weight_matrix_one, weight_matrix_two = _portfolio_cut(weight_matrix)
    weight_matrices_one = (
        [weight_matrix_one]
        if len(weight_matrix_one) <= max_size
        else portfolio_cuts(weight_matrix_one, max_size)
    )
    weight_matrices_two = (
        [weight_matrix_two]
        if len(weight_matrix_two) <= max_size
        else portfolio_cuts(weight_matrix_two, max_size)
    )
    return weight_matrices_one + weight_matrices_two
