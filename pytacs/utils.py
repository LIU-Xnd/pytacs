import numpy as _np
from scanpy import AnnData as _AnnData
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.sparse import dok_matrix as _dok_matrix
from scipy.sparse import issparse as _issparse
from numpy import matrix as _matrix
from numpy.typing import NDArray as _NDArray
from typing import Iterator as _Iterator


# Placeholder type
class _UndefinedType:
    def copy(self):
        return self

    def __repr__(self):
        return "_UNDEFINED"


# Alias
_Undefined = _UndefinedType

_UNDEFINED = _UndefinedType()


# >>> --- I/O operations ---
def read_list(filepath: str) -> list[str]:
    """Read list from txt, sep by \n. Strings are .rstrip()'d."""
    with open(filepath) as f:
        return [line.rstrip() for line in f]


def combine_to_str(*arg, sep: str = ",") -> str:
    """Combine arguments into a string.
    For example, `combine_to_str(1,2) -> '1,2'`."""
    args_to_str = [str(a) for a in arg]
    return sep.join(args_to_str)


def extract_ints_from_str(string_: str, sep: str = ",") -> list[int]:
    """Extract a list of integers from a string.
    For example, `extract_ints_from_str('1,2') -> [1,2]`."""
    return list(map(int, string_.split(sep)))


def print_newlines(*args, sep: str = "\n") -> None:
    """Print args one by one, seperated by newline."""
    print(*args, sep=sep)
    return None


# --- I/O operations --- <<<


# >>> --- Reshaping operations ---


def find_indices(lst1: _Iterator, lst2: _Iterator) -> _NDArray[_np.int_]:
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
    X: _np.ndarray, genes_X: _NDArray[_np.str_], genes_target: _NDArray[_np.str_]
) -> _np.ndarray:
    """Reshape X to match genes_target, setting absent gene counts to 0."""
    idx_subgenes = find_indices(lst1=genes_target, lst2=genes_X)
    X_rearranged = _np.zeros(
        shape=(X.shape[0], len(genes_target)),
        dtype=X.dtype,
    )
    valid_indices = idx_subgenes >= 0
    X_rearranged[:, valid_indices] = X[:, idx_subgenes[valid_indices]]
    return X_rearranged


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


# --- Reshaping operations --- <<<


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


# def deepcopy_dict(d: dict[int, list]) -> dict:
#     return {k: v.copy() for k, v in d.items()}


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
