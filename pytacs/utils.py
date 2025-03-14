import numpy as _np
from scanpy import AnnData as _AnnData
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.sparse import dok_matrix as _dok_matrix
from numpy import matrix as _matrix


# Placeholder type
class _UndefinedType:
    def copy(self):
        return self

    def __repr__(self):
        return "_UNDEFINED"


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
def _indices_of_ls1ElementInLst2(lst1: list, lst2: list) -> list[int]:
    """Returns a list whose elements are integers, where:
     the k-th integer are the index of lst1's k-th element in lst2,
     and if lst1's k-th element is not in lst2, the returned k-th element is -1.
    For example:
     input: lst1=[2,4,5,6,7], lst2=[5,4,7,3,2,1]
     output: [4,1,0,-1,2]"""
    return [lst2.index(ele) if ele in lst2 else -1 for ele in lst1]


def subCountMatrix_genes2InGenes1(
    X: _np.ndarray, genes1: list[str], genes2: list[str]
) -> _np.ndarray:
    """Select those genes in genes2 that appear in genes1 (order preserved as genes1).
     RNA count of those genes in genes1 but do not appear in genes2 is set to 0.
    X's columns correspond to genes2. They want reshaping into genes1.

    This function is often used to unify the covariates so as to fit in a pretrained local
     classifier.

    Args:
        X (ndarray): sample-by-gene count matrix.
        genes1 (list[str]): a standard gene list, often a snRNA-seq gene list.
        genes2 (list[str]): genes corresponding to X's columns. They want reshaping into genes1.

    Return:
        ndarray: a count matrix almost the same as X, but columns reshaped into genes1, and
         counts of those genes which are absent in genes2 set to 0.

    For example:
        input:
            X=[[...]] (n_samples * n_genes_sp, in this case, N * 4);
            genes1=['Malat1', 'Cgnl1', 'Golga4'] (local classifier-learnt "snRNA" gene list);
            genes2=['Cgnl1', 'Malat1', 'Otherfoo', 'Otherbar'] (spRNA-seq data, cols of X);
        output:
            [[..]] (n_samples * n_genes_sn, in this case, N * 3), whose columns are
             ['Malat1', 'Cgnl1', 'Golga4'], where X's ['Golga4'] column is set to 0 due
             to absence."""
    idx_subgenes = _indices_of_ls1ElementInLst2(list(genes1), list(genes2))
    # print(f"{len(idx_subgenes)=}")
    return _np.concatenate(
        [X, _np.zeros((X.shape[0], 1), dtype=int)],  # those not shown set to 0
        axis=1,
    )[:, _np.array(idx_subgenes)]


def save_and_tidy_index(
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
    adata.obs[colname_to_save_oldIndex] = adata.obs.index
    adata.obs.index = _np.arange(adata.obs.shape[0]).astype(str)
    return None


# --- Reshaping operations --- <<<


def radial_basis_function(
    location_vector: _np.ndarray,
    centroid_vector: _np.ndarray | None = None,
    scale: float = 1.0,
) -> float:
    if centroid_vector is None:
        centroid_vector = _np.zeros((location_vector.shape[0],))
    return (
        1 / _np.power(2 * _np.pi * _np.power(scale, 2), location_vector.shape[0] / 2)
    ) * _np.exp(
        -_np.power(_np.linalg.norm(location_vector - centroid_vector), 2)
        / (2 * _np.power(scale, 2))
    )


def to_array(
    X: _np.ndarray | _csr_matrix | _dok_matrix | _matrix,
    squeeze: bool = False,
) -> _np.ndarray:
    if isinstance(X, _csr_matrix) or isinstance(X, _dok_matrix):
        X = X.toarray()
    elif isinstance(X, _matrix):
        X = _np.array(X.tolist())
    assert isinstance(X, _np.ndarray)
    if squeeze:
        X = X.reshape(-1)
    return X


def deepcopy_dict(d: dict[int, list]) -> dict:
    return {k: v.copy() for k, v in d.items()}
