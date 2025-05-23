import numpy as _np
from scipy.sparse import issparse as _issparse
from scipy.sparse import eye as _eye
from tqdm import tqdm as _tqdm


from .types import (
    _AnnData,
    _NDArray,
    _UndefinedType,
    _UNDEFINED,
    _Iterable,
    _matrix,
    _csr_matrix,
    _coo_matrix,
    _dok_matrix,
    _lil_matrix,
)


# >>> Reshaping operations
def find_indices(lst1: _Iterable, lst2: _Iterable) -> _NDArray[_np.int_]:
    """Returns an array of indices where elements of lst1 appear in lst2, or -1 if not found.

    Example:
        ls1 = np.array([2,4,5,6,7])
        ls2 = np.array([5,4,7,3,2,0])
        iloc_2to1 = find_indices(ls1, ls2)
        # We have
        ls2[iloc_2to1] == np.array([2,4,5,0,7])
    """
    index_map = {
        val: idx for idx, val in enumerate(lst2)
    }  # Create a mapping for fast lookup
    return _np.array([index_map.get(ele, -1) for ele in lst1])


def rearrange_count_matrix(
    X: _csr_matrix | _coo_matrix | _np.ndarray,
    genes_X: _NDArray[_np.str_],
    genes_target: _NDArray[_np.str_],
) -> _csr_matrix:
    """Reshape X to match genes_target, setting absent gene counts to 0."""
    idx_subgenes = find_indices(lst1=genes_target, lst2=genes_X)
    X_rearranged = _lil_matrix(
        (X.shape[0], len(genes_target)),
    ).astype(X.dtype)
    valid_indices = idx_subgenes >= 0
    X_rearranged[:, valid_indices] = X[:, idx_subgenes[valid_indices]]
    return X_rearranged.tocsr()


def reinit_index(
    adata: _AnnData,
    colname_to_save_oldIndex: str = "old_index",
) -> None:
    """Save old index as a col of obs and re-index with integers (string type) (only apply
     for .obs).
    Inplace operation."""
    while colname_to_save_oldIndex in adata.obs_keys():
        print(
            f"Warning: {colname_to_save_oldIndex} already in obs! New name: {colname_to_save_oldIndex}_copy."
        )
        colname_to_save_oldIndex += "_copy"
    adata.obs[colname_to_save_oldIndex] = adata.obs.index.values
    adata.obs.index = _np.arange(adata.obs.shape[0]).astype(str)
    return


# <<< End of Reshaping operations


def radial_basis_function(
    location_vectors: _np.ndarray,
    centroid_vector: _np.ndarray | None = None,
    scale: float = 1.0,
) -> _NDArray[_np.float_]:
    """
    Computes the values of a multivariate Gaussian radial basis function (RBF) for a batch of vectors.

    Args:
        location_vectors (np.ndarray): An (N, D) array where each row is a D-dimensional input vector.
        centroid_vector (np.ndarray | None, optional): The D-dimensional center of the RBF. Defaults to the origin.
        scale (float, optional): The standard deviation (spread) of the RBF. Defaults to 1.0.

    Returns:
        np.ndarray: An (N,) array containing the RBF values for each input vector.
    """
    if centroid_vector is None:
        centroid_vector = _np.zeros(
            (1, location_vectors.shape[1])
        )  # Shape (1, D) for broadcasting
    scale_squared = scale**2
    dim = location_vectors.shape[1]

    coeff = 1 / ((2 * _np.pi * scale_squared) ** (dim / 2))
    dist_squared = _np.sum((location_vectors - centroid_vector) ** 2, axis=1)
    expo = -dist_squared / (2 * scale_squared)

    return coeff * _np.exp(expo)


def to_array(
    X: _np.ndarray | _csr_matrix | _dok_matrix | _matrix,
    squeeze: bool = False,
) -> _np.ndarray:
    """
    Converts various matrix types (NumPy array, SciPy sparse matrices, or NumPy matrix) into a NumPy array.

    Args:
        X (np.ndarray | csr_matrix | dok_matrix | np.matrix): Input matrix to be converted.
        squeeze (bool, optional): If True, the output array is flattened. Defaults to False.

    Returns:
        np.ndarray: The converted NumPy array.
    """
    if _issparse(X):
        X = X.toarray()
    elif isinstance(X, _matrix):
        X = _np.asarray(X)
    if squeeze:
        X = X.ravel().copy()
    return X


def truncate_top_n(
    arr: _np.ndarray,
    n_top: int,
    return_bools: bool = False,
) -> _np.ndarray:
    """
    Truncates the input array by setting all but the top `n_top` values to 0.

    Args:
        arr (np.ndarray): 1D input array.
        n_top (int): Number of top values to retain (sorted in descending order).
        return_bools (bool, optional): If True, returns a boolean array where True represents the top values.
                                        Defaults to False, which returns a float array.

    Returns:
        np.ndarray: A 1D array where only the top `n_top` values are set to 1.0 (or True if `return_bools=True`).
    """
    assert arr.ndim == 1, "Input array must be 1D"

    ilocs_truncated = _np.argsort(arr)[-n_top:][::-1]
    res = _np.zeros_like(arr, dtype=_np.bool_ if return_bools else arr.dtype)
    res[ilocs_truncated] = 1.0

    return res


def normalize_csr(mat_csr: _csr_matrix) -> _csr_matrix:
    """
    Normalize a sparse CSR matrix by dividing each row by its row sum.

    This function computes the inverse of the sum of each row in the input
    sparse matrix and multiplies the matrix by the corresponding diagonal matrix
    to normalize each row. The result is a sparse matrix where each row's sum is 1
    (if the original row sum was non-zero).

    Parameters:
    -----------
    mat_csr : _csr_matrix
        A sparse matrix in Compressed Sparse Row (CSR) format.

    Returns:
    --------
    _csr_matrix
        A normalized sparse matrix where each row is divided by its row sum.
        The input matrix remains in CSR format.

    Notes:
    ------
    - If any row has a sum of zero, it will be skipped, and the corresponding row
      will remain zero in the resulting matrix.
    - The function assumes that the input matrix is in CSR format and efficiently
      performs the normalization without converting the matrix to a dense format.
    """
    row_sums: _np.ndarray = to_array(mat_csr.sum(axis=1), squeeze=True)
    row_sums[row_sums == 0.0] = 1.0
    row_inv = 1.0 / row_sums
    diag_row_inv = _eye(
        m=mat_csr.shape[0],
        format="csr",
    )
    diag_row_inv.data = row_inv
    return diag_row_inv @ mat_csr  # Normalized


def prune_csr_per_row(
    csr_mat: _csr_matrix,
    prune_proportion: float = 0.5,
    tqdm_verbose: bool = True,
) -> _csr_matrix:
    """
    Deprecated.

    Prune the least `prune_proportion` terms per row.
    """
    new_data = []
    new_indices = []
    new_indptr = [0]

    if tqdm_verbose:
        itor_ = _tqdm(
            range(csr_mat.shape[0]),
            desc=f"Pruning off {prune_proportion:.1%} nodes",
            ncols=60,
        )
    else:
        itor_ = range(csr_mat.shape[0])
    for i in itor_:
        row_start = csr_mat.indptr[i]
        row_end = csr_mat.indptr[i + 1]
        row_data = csr_mat.data[row_start:row_end]
        row_indices = csr_mat.indices[row_start:row_end]

        if len(row_data) == 0:
            new_indptr.append(len(new_data))
            continue

        # Sort by value
        sorted_idx = _np.argsort(row_data)[::-1]
        keep_count = max(1, int(_np.ceil(len(row_data) * (1 - prune_proportion))))
        keep_idx = sorted_idx[:keep_count]

        # Keep only top proportion
        new_data.extend(row_data[keep_idx])
        new_indices.extend(row_indices[keep_idx])
        new_indptr.append(len(new_data))

    pruned_csr = _csr_matrix(
        (new_data, new_indices, new_indptr),
        shape=csr_mat.shape,
    )
    return pruned_csr


def prune_csr_per_row_cum_prob(
    csr_mat: _csr_matrix,
    cum_prob_keep: float = 0.5,
    tqdm_verbose: bool = True,
) -> _csr_matrix:
    """
    DEPRECATED.

    Keeps the top terms with cumulative probability of `cum_prob_keep` per row,
    and prune the rest. At least one term is retained.
    """
    if cum_prob_keep >= 1.0:
        return csr_mat.copy()
    new_data = []
    new_indices = []
    new_indptr = [0]

    if tqdm_verbose:
        itor_ = _tqdm(
            range(csr_mat.shape[0]),
            desc=f"Pruning off nodes",
            ncols=60,
        )
    else:
        itor_ = range(csr_mat.shape[0])
    for i in itor_:
        row_start = csr_mat.indptr[i]
        row_end = csr_mat.indptr[i + 1]
        row_data = csr_mat.data[row_start:row_end]
        row_indices = csr_mat.indices[row_start:row_end]

        if len(row_data) == 0:
            new_indptr.append(len(new_data))
            continue

        # Sort by value
        sorted_idx = _np.argsort(row_data)[::-1]
        cumprob = _np.cumsum(row_data[sorted_idx])
        keep_count = _np.argwhere(cumprob >= cum_prob_keep)[0][0] + 1
        keep_idx = sorted_idx[:keep_count]

        # Keep only top proportion
        new_data.extend(row_data[keep_idx])
        new_indices.extend(row_indices[keep_idx])
        new_indptr.append(len(new_data))

    pruned_csr = _csr_matrix(
        (new_data, new_indices, new_indptr),
        shape=csr_mat.shape,
    )
    pruned_csr.eliminate_zeros()
    return pruned_csr

def prune_csr_per_row_infl_point(
    csr_mat: _csr_matrix,
    min_points_to_keep: int = 1,
    tqdm_verbose: bool = True,
) -> _csr_matrix:
    """
    Keeps the top terms per row and prune the rest. The dividing point is the
    inflection point of the sorted similarity curve.
    """
    new_data = []
    new_indices = []
    new_indptr = [0]

    if tqdm_verbose:
        itor_ = _tqdm(
            range(csr_mat.shape[0]),
            desc=f"Pruning off nodes",
            ncols=60,
        )
    else:
        itor_ = range(csr_mat.shape[0])
    for i in itor_:
        row_start = csr_mat.indptr[i]
        row_end = csr_mat.indptr[i + 1]
        row_data = csr_mat.data[row_start:row_end]
        row_indices = csr_mat.indices[row_start:row_end]

        if len(row_data) == 0:
            new_indptr.append(len(new_data))
            continue
        
        # Sort by value
        sorted_idx = _np.argsort(row_data)[::-1]
        if len(row_data) <= min_points_to_keep:
            keep_idx = sorted_idx
        else:
            sorted_curve = row_data[sorted_idx]
            delta_curve = sorted_curve[1:] - sorted_curve[:-1]
            infl_point = _np.argmax(delta_curve)
            keep_idx = sorted_idx[:max(min_points_to_keep, infl_point+1)]

        # Keep only top proportion
        new_data.extend(row_data[keep_idx])
        new_indices.extend(row_indices[keep_idx])
        new_indptr.append(len(new_data))

    pruned_csr = _csr_matrix(
        (new_data, new_indices, new_indptr),
        shape=csr_mat.shape,
    )
    pruned_csr.eliminate_zeros()
    return pruned_csr


def rowwise_cosine_similarity(A: _np.ndarray, B: _np.ndarray) -> _NDArray[_np.float_]:
    """
    Compute row-wise cosine similarity between two matrices of shape (N, p).

    Each row in A and B represents a p-dimensional embedding of a sample.
    The function returns a 1D array of length N, where each entry is the
    cosine similarity between the corresponding rows of A and B.

    Parameters:
    -----------
    A : np.ndarray
        Array of shape (N, p), representing N sample embeddings.
    B : np.ndarray
        Array of shape (N, p), representing N sample embeddings.

    Returns:
    --------
    similarities : np.ndarray
        Array of shape (N,), containing the cosine similarity between
        each pair of corresponding rows in A and B.

    Notes:
    ------
    - If any row in A or B has zero norm, its similarity will be computed
      safely with a small epsilon to avoid division by zero.
    """
    assert A.shape == B.shape, "Input matrices must have the same shape"

    dot = _np.sum(A * B, axis=1)
    norm_A = _np.linalg.norm(A, axis=1)
    norm_B = _np.linalg.norm(B, axis=1)

    # Avoid zero division
    denom = norm_A * norm_B
    denom[denom == 0.0] = 1e-8

    return dot / denom
