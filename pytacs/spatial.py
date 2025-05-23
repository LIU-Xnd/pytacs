"""A new version of spatial strategy based on Randon Walk, with fast computation,
low mem cost, and robust performance."""

from .types import (
    _AnnData,
    _NDArray,
    _csr_matrix,
    _coo_matrix,
    _lil_matrix,
    _Literal,
    _UndefinedType,
    _UNDEFINED,
    _1DArrayType,
    _NumberType,
    _Nx2ArrayType,
)
import scanpy as _sc
import numpy as _np
from scipy.spatial import cKDTree as _cKDTree  # to construct sparse distance matrix

from scipy.cluster.hierarchy import linkage as _linkage
from scipy.cluster.hierarchy import fcluster as _fcluster
from sklearn.cluster import MiniBatchKMeans as _MiniBatchKMeans

from .classifier import _LocalClassifier
from .utils import to_array as _to_array

from tqdm import tqdm as _tqdm
from dataclasses import dataclass as _dataclass
import pandas as _pd
from sklearn.decomposition import TruncatedSVD as _TruncatedSVD
from .utils import normalize_csr as _normalize_csr
from .utils import prune_csr_per_row_cum_prob as _prune_csr_per_row_cum_prob
from .utils import prune_csr_per_row_infl_point as _prune_csr_per_row_infl_point
from .utils import rowwise_cosine_similarity as _rowwise_cosine_similarity
from itertools import product as _product
from multiprocessing.pool import Pool as _Pool


def spatial_distances(
    sp_adata: _AnnData,
    max_spatial_distance: _NumberType,
    p_norm: _NumberType = 2,
    verbose: bool = True,
) -> None:
    """Computes spatial distances matrix in csr_matrix format. Saved in place."""
    if verbose:
        _tqdm.write('Loading spatial coordinates from .obsm["spatial"]..')
    ckdtree_spatial = _cKDTree(sp_adata.obsm["spatial"])
    if verbose:
        _tqdm.write("Building spatial distances, might take up large memory..")
    distances_propagation = _csr_matrix(
        ckdtree_spatial.sparse_distance_matrix(
            other=ckdtree_spatial,
            max_distance=max_spatial_distance,
            p=p_norm,
        )
    )
    distances_propagation.eliminate_zeros()
    sp_adata.obsp["spatial_distances"] = _csr_matrix(distances_propagation)
    sp_adata.uns["max_spatial_distance"] = max_spatial_distance
    if verbose:
        _tqdm.write(
            'Saved in .obsp["spatial_distances"]. Related param saved in .uns["max_spatial_distance"]'
        )
    return


@_dataclass
class AggregationResult:
    """Results of spot aggregation.

    Attrs:
        .dataframe (pd.DataFrame):
            ["cell_id"]: (int) centroid spot id
            ["cell_type"]: (str) cell-type name
            ["confidence"]: (float) probability of that class
            (optional ["cell_size"]: (int) spots in each cell)

        .expr_matrix (csr_matrix[float]): aggregated expression matrix of confident spots

        .weight_matrix (csr_matrix[float]): transition probability matrix of all spots
    """

    dataframe: _pd.DataFrame
    expr_matrix: _csr_matrix
    weight_matrix: _csr_matrix


def rw_aggregate(
    st_anndata: _AnnData,
    classifier: _LocalClassifier,
    max_iter: int = 20,
    steps_per_iter: int = 1,
    nbhd_radius: float = 1.5,
    max_propagation_radius: float = 6.0,
    normalize_: bool = True,
    log1p_: bool = True,
    mode_embedding: _Literal["raw", "pc"] = "pc",
    n_pcs: int = 30,
    mode_metric: _Literal["inv_dist", "cosine"] = "cosine",
    mode_aggregation: _Literal["weighted", "unweighted"] = "unweighted",
    mode_prune: _Literal['inflection_point', 'proportional'] = 'inflection_point',
    min_points_to_keep: int = 1,
    mode_walk: _Literal["rw"] = "rw",
    return_weight_matrix: bool = False,
    return_cell_sizes: bool = True,
    verbose: bool = True,
) -> AggregationResult:
    """
    Perform iterative random-walk-based spot aggregation and classification refinement
    for spatial transcriptomics data.

    This function aggregates local spot neighborhoods using a random walk or
    random walk with restart (RWR), then refines cell type predictions iteratively
    using a local classifier and aggregated gene expression until confident.

    Args:
        st_anndata (_AnnData):
            AnnData object containing spatial transcriptomics data.

        classifier (_LocalClassifier):
            A local cell-type classifier with `predict_proba` and `fit` methods, as well as
            `threshold_confidence` attribute for confidence determination.

        max_iter (int, optional):
            Number of refinement iterations to perform.

        steps_per_iter (int, optional):
            Number of random walk steps to perform in each iteration.

        nbhd_radius (float, optional):
            Radius for defining local neighborhood in spatial graph construction.

        max_propagation_radius (float, optional):
            Radius for maximum possible random walk distance in spatial graph propagation.

        normalize_ (bool, optional):
            Whether to perform normalization on raw count matrix before building spatial graph.
            Default is True.

        log1p_ (bool, optional):
            Whether to perform log1p on raw count matrix before building spatial graph. Default is True.

        mode_embedding (Literal['raw', 'pc'], optional):
            Embedding mode for similarity calculation.
            'raw' uses the original expression matrix; 'pc' uses PCA-reduced data. Default is 'pc'.

        n_pcs (int, optional):
            Number of principal components to retain when `mode_embedding='pc'`.

        mode_metric (Literal['inv_dist', 'cosine'], optional):
            Distance or similarity metric to define transition weights between spots.

        mode_aggregation (Literal['unweighted', 'weighted'], optional):
            Aggregation strategy to combine neighborhood gene expression.
            'unweighted' uses uniform averaging; 'weighted' uses transition probabilities.

        mode_walk (Literal['rw', 'rwr'], optional):
            Type of random walk to perform:
            'rw' for vanilla random walk,
            'rwr' for random walk with restart. Default is 'rw'.

        return_weight_matrix (bool, optional):
            If True, will return weight_matrix in AggregationResult. Note this process may increase
            computational time!

        return_cell_sizes (bool, optional):
            If True, will return cellsizes of confident cells in AggregationResult.dataframe. This process can
            help improve binning accuracy for downstream analysis.

    Returns:
        AggregationResult:
            A dataclass containing:
                - `dataframe`: DataFrame with predicted `cell_id`, `cell_type`, and confidence scores.
                - `expr_matrix`: CSR matrix of aggregated expression for confident spots.
                - `weight_matrix`: CSR matrix representing transition probabilities between all spots, if `return_weight_matrix`;
                otherwise an empty matrix of the same shape.
    """
    assert mode_embedding in ["raw", "pc"]
    assert mode_metric in ["inv_dist", "cosine"]
    assert mode_aggregation in ["weighted", "unweighted"]
    assert mode_prune in ['inflection_point', 'proportional']
    assert mode_walk in ["rw"]

    if normalize_:
        X_normalized = _normalize_csr(st_anndata.X.astype(float)).copy()
    else:
        X_normalized = st_anndata.X.astype(float).copy()
    if log1p_:
        X_normalized.data = _np.log1p(X_normalized.data)
    # Get SVD transformer
    if mode_embedding == "pc":
        n_pcs: int = min(n_pcs, st_anndata.shape[1])
        if n_pcs > 100:
            _tqdm.write(
                f"Warning: {n_pcs} pcs might be too large. Take care of your ram."
            )
        svd = _TruncatedSVD(
            n_components=n_pcs,
        )
        if verbose:
            _tqdm.write(f"Performing truncated PCA (n_pcs={n_pcs})..")
        svd.fit(
            X=X_normalized,
        )
        embed_loadings: _np.ndarray = svd.components_  # k x n_features
        del svd
    else:
        n_features = st_anndata.shape[1]
        if n_features > 100:
            _tqdm.write(
                f"Number of features {n_features} might be too large. Take care of your ram."
            )
        embed_loadings: _np.ndarray = _np.identity(
            n=n_features,
        )
    # Generate topology relation matrix
    # Get spatial neighborhood
    if verbose:
        _tqdm.write(f"Constructing spatial graph..")
    # Try to load from .obsp
    if verbose:
        _tqdm.write(f'Trying to load from cache .obsp["spatial_distances"]..')
    if "spatial_distances" in st_anndata.obsp:
        distances_propagation = _coo_matrix(st_anndata.obsp["spatial_distances"])
    else:
        if verbose:
            _tqdm.write(f"Cache not found. Computing..")
        spatial_distances(
            sp_adata=st_anndata,
            max_spatial_distance=max_propagation_radius,
            p_norm=2,
            verbose=verbose,
        )
        distances_propagation: _coo_matrix = _coo_matrix(
            st_anndata.obsp["spatial_distances"]
        )
    # Construct spatial dist matrix
    distances_propagation.eliminate_zeros()
    spatial_rows = distances_propagation.nonzero()[0]
    spatial_cols = distances_propagation.nonzero()[1]
    spatial_data = distances_propagation.data
    whr_within_nbhd = spatial_data <= nbhd_radius
    if verbose:
        _tqdm.write('Building neighborhoods..')
    distances_spatial = _coo_matrix(
        (
            spatial_data[whr_within_nbhd],
            (spatial_rows[whr_within_nbhd], spatial_cols[whr_within_nbhd]),
        ),
        shape=distances_propagation.shape,
    )
    del spatial_data, spatial_cols, spatial_rows
    # Boundaries for propagation
    if verbose:
        _tqdm.write('Building random-walk boundaries..')
    ilocs_propagation = _np.array(
        list(zip(distances_propagation.row, distances_propagation.col))
    )

    rows_nonzero, cols_nonzero = (
        ilocs_propagation[:, 0],
        ilocs_propagation[:, 1],
    )
    del ilocs_propagation
    rows_nonzero = _np.concatenate(
        [
            rows_nonzero,
            _np.arange(distances_propagation.shape[0]),
        ]  # including selves (diagonals)
    )
    cols_nonzero = _np.concatenate(
        [cols_nonzero, _np.arange(distances_propagation.shape[0])]  # including selves
    )
    del distances_propagation
    query_pool_propagation = set(
        zip(rows_nonzero, cols_nonzero)
    )  # all possible nonzero ilocs for propagation
    del rows_nonzero
    del cols_nonzero
    distances = _lil_matrix(
        (distances_spatial.shape[0], distances_spatial.shape[1]),
    )
    # Get defined embedding
    if verbose:
        _tqdm.write('Getting defined embeddings..')
    embeds: _np.ndarray = X_normalized @ embed_loadings.T
    del X_normalized
    del embed_loadings
    if verbose:
        _tqdm.write('Computing topology graph..')
    ilocs_nonzero = _np.array(list(zip(distances_spatial.row, distances_spatial.col)))
    if mode_metric == "inv_dist":
        distances[ilocs_nonzero[:, 0], ilocs_nonzero[:, 1]] = _np.linalg.norm(
            embeds[ilocs_nonzero[:, 0], :] - embeds[ilocs_nonzero[:, 1], :], axis=1
        )
        distances = distances.tocoo()

        # Compute inv_dist similarity using sparse operations: S_ij = 1 / (1 + d_ij)
        similarities = _coo_matrix(
            (1 / (1 + distances.data), (distances.row, distances.col)),
            shape=distances.shape,
        )
        del distances
        similarities = similarities.tolil()
    else:
        similarities = _lil_matrix(
            (distances_spatial.shape[0], distances_spatial.shape[1])
        )
        similarities[ilocs_nonzero[:, 0], ilocs_nonzero[:, 1]] = (
            _rowwise_cosine_similarity(
                embeds[ilocs_nonzero[:, 0], :],
                embeds[ilocs_nonzero[:, 1], :],
            )
        )
    similarities[
        _np.arange(similarities.shape[0]), _np.arange(similarities.shape[0])
    ] = 1.0
    similarities: _csr_matrix = similarities.tocsr()

    similarities = _normalize_csr(
        similarities
    )  # Normalize similarities row-wise, making it probability-like

    candidate_cellids = _np.arange(
        st_anndata.shape[0]
    )  # candidate (yet-undefined) cellids, gonna pop items with iterations
    cellids_confident = []  # cellids confident, gonna append items with iters
    celltypes_confident = (
        []
    )  # celltypes corrsponding to cellids_confident, gonna append items
    confidences_confident = (
        []
    )  # confidences corresponding to celltypes_confident, gonna append items
    cellsizes_confident = (
        []
    )  # cellsizes (in spots) corresponding to cellids_confident, gonna append items
    weight_matrix: dict = {
        "rows": [],
        "cols": [],
        "data": [],
    }  # final weight_matrix of all spots, gonna update

    # Random Walk
    if verbose:
        _tqdm.write('Random walking..')
    counter_celltypes_global = dict()  # counter of celltypes total
    for i_iter in range(max_iter):
        # Aggregate spots according to similarities
        if mode_aggregation == "unweighted":
            X_agg_candidate: _csr_matrix = similarities[candidate_cellids, :].astype(
                bool
            ).astype(float) @ st_anndata.X.astype(float)
        else:
            X_agg_candidate: _csr_matrix = similarities[
                candidate_cellids, :
            ] @ st_anndata.X.astype(float)
        # Classify
        if verbose:
            _tqdm.write("Classifying..")
        probs_candidate: _np.ndarray = classifier.predict_proba(
            X=X_agg_candidate,
            genes=st_anndata.var.index.values,
        )
        type_ids_candidate: _NDArray[_np.int_] = _np.argmax(probs_candidate, axis=1)
        confidences_candidate: _NDArray[_np.float_] = probs_candidate[
            _np.arange(probs_candidate.shape[0]), type_ids_candidate
        ]
        # Find those confident
        whr_confident_candidate: _NDArray[_np.bool_] = (
            confidences_candidate >= classifier._threshold_confidence
        )
        counter_celltypes = dict()  # counter of celltypes for this round
        ave_conf = 0.0
        if verbose:
            _itor = _tqdm(
                range(len(candidate_cellids)),
                desc=f"Gather iter {i_iter+1} results",
                ncols=60,
            )
        else:
            _itor = range(len(candidate_cellids))
        for i_candidate in _itor:
            cellid: int = candidate_cellids[i_candidate]
            is_conf: bool = whr_confident_candidate[i_candidate]
            conf = confidences_candidate[i_candidate]
            if is_conf:
                typeid: int = type_ids_candidate[i_candidate]
                typename: str = classifier._classes[typeid]
                cellids_confident.append(cellid)
                celltypes_confident.append(typename)
                confidences_confident.append(conf)
                if return_weight_matrix:
                    weight_matrix["rows"].append(cellid)
                    cols_nonzero = list(
                        similarities.getrow(cellid).nonzero()[1]
                    )  # FIXBUG: add list(..)
                    weight_matrix["cols"] += cols_nonzero
                    weight_matrix["data"] += _to_array(
                        similarities[cellid, cols_nonzero],
                        squeeze=True,
                    ).tolist()
                if return_cell_sizes:
                    cols_nonzero = list(similarities.getrow(cellid).nonzero()[1])
                    cellsizes_confident.append(len(cols_nonzero))
                counter_celltypes[typename] = counter_celltypes.get(typename, 0) + 1
                counter_celltypes_global[typename] = (
                    counter_celltypes_global.get(typename, 0) + 1
                )
            ave_conf += conf

        if verbose:
            _tqdm.write(f"Ave conf: {ave_conf/candidate_cellids.shape[0]:.2%}")
        candidate_cellids = candidate_cellids[~whr_confident_candidate]
        if verbose:
            _tqdm.write(f"{counter_celltypes=}")
            _tqdm.write(f"{counter_celltypes_global=}")
        if len(candidate_cellids) == 0:
            break
        # Random walk
        if verbose:
            _itor = _tqdm(
                range(steps_per_iter),
                desc="Random walk..",
                ncols=60,
            )
        else:
            _itor = range(steps_per_iter)
        for i_step in _itor:
            similarities: _csr_matrix = similarities @ similarities
            # Truncate propagation by max_propagation_radius for fast computation and stability.
            similarities: _coo_matrix = similarities.tocoo()
            # including diagonals

            mask = _np.zeros_like(similarities.data, dtype=bool)
            for i in range(len(similarities.data)):
                if (similarities.row[i], similarities.col[i]) in query_pool_propagation:
                    mask[i] = True

            # Filter the data to keep only selected values
            data_kept = similarities.data[mask]
            similarities: _coo_matrix = _coo_matrix(
                (data_kept, (similarities.row[mask], similarities.col[mask])),
                shape=similarities.shape,
            )
            # Convert back
            similarities: _csr_matrix = similarities.tocsr()
            # Re-normalize
            similarities: _csr_matrix = _normalize_csr(similarities)
        # prune
        if mode_prune == 'proportional':
            similarities = _prune_csr_per_row_cum_prob(
                csr_mat=similarities,
                cum_prob_keep=0.5,
                tqdm_verbose=verbose,
            )
        else:
            similarities = _prune_csr_per_row_infl_point(
                csr_mat=similarities,
                min_points_to_keep=min_points_to_keep,
                tqdm_verbose=verbose,
            )

    weight_matrix: _coo_matrix = _coo_matrix(
        (weight_matrix["data"], (weight_matrix["rows"], weight_matrix["cols"])),
        shape=(st_anndata.X.shape[0], st_anndata.X.shape[0]),
    )
    weight_matrix: _csr_matrix = weight_matrix.tocsr()
    # Construct Results
    result = AggregationResult(
        dataframe=_pd.DataFrame(
            {
                "cell_id": cellids_confident,
                "cell_type": celltypes_confident,
                "confidence": confidences_confident,
            },
        ),
        expr_matrix=similarities[cellids_confident, :] @ st_anndata.X,
        weight_matrix=weight_matrix,
    )
    if return_cell_sizes:
        result.dataframe["cell_size"] = cellsizes_confident
    return result


def extract_celltypes_full(
    aggregation_result: AggregationResult,
    name_undefined: str = "Undefined",
) -> _NDArray[_np.str_]:
    """
    Extract the cell-type labels for all spots from an aggregation result, including
    both confident and non-confident spots.

    This function retrieves the cell-type information for each spot (cell) from the
    aggregation result. The resulting cell-type
    labels will be sorted by the cell ID. Any missing spots will be assigned the label
    specified by `name_undefined`.

    The spot IDs are assumed to be continuous from 0 to n_samples-1. If there are missing
    spots in the data, they will be labeled with `name_undefined`.

    Parameters:
    -----------
    aggregation_result : AggregationResult
        The result of the aggregation process, which includes cell-type predictions
        for each spot, as well as their confidence levels.

    name_undefined : str, optional, default="Undefined"
        The label used for spots that are missing or undefined. If no cell-type is assigned
        to a spot, it will be labeled with this value.

    Returns:
    --------
    _NDArray[_np.str_]
        A 1D array of cell-type labels for each spot, where each label corresponds
        to a specific cell or region in the input dataset. The array will be sorted
        by cell ID. Undefined spots will be labeled
        with `name_undefined`.

    Notes:
    ------
    - This function includes both confident and non-confident spots.
    - The function assumes that the aggregation result has cell-type labels accessible.
    """

    celltypes_full = _np.zeros(
        shape=(aggregation_result.weight_matrix.shape[0],),
    ).astype(str)
    celltypes_full[:] = name_undefined
    cellids_conf = aggregation_result.dataframe["cell_id"].values.astype(int)
    celltypes_conf = aggregation_result.dataframe["cell_type"].values.astype(str)
    celltypes_full[cellids_conf] = celltypes_conf[:]

    return celltypes_full


@_dataclass
class SpatialTypeAnnCntMtx:
    """
    A data class representing a gene count matrix with spatial coordinates and annotated cell types.

    Attributes:
    -----------
    count_matrix : scipy.sparse.csr_matrix
        A sparse matrix of shape (n_samples, n_genes) where each entry represents
        the count of a specific gene in a specific sample (or spatial location).

    spatial_distances : csr_matrix
        A 2D sparse array of shape (n_samples, n_samples), indicating distances between
        each sample, with all distances above a threshold being set to zero.

    cell_types : numpy.ndarray
        A 1D array of length n_samples where each element is a string representing
        the cell type annotation for the corresponding sample or cell.

    Assertions:
    -----------
    - The number of rows in `count_matrix` must match the number of rows in `spatial_distances`
      (i.e., the number of spatial locations).
    - The number of rows in `count_matrix` must also match the number of entries in `cell_types`
      (i.e., the number of annotated cell types).

    Example:
    --------
    # Create a sparse count matrix, spatial coordinates, and cell types.
    count_matrix = csr_matrix([[5, 3], [4, 2], [6, 1]])
    spatial_coords = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]])
    cell_types = np.array(['TypeA', 'TypeB', 'TypeA'])

    # Initialize the SpatialTypeAnnCntMtx object.
    mtx = SpatialTypeAnnCntMtx(count_matrix, spatial_coords, cell_types)
    """

    count_matrix: _csr_matrix
    spatial_distances: _csr_matrix
    cell_types: _NDArray[_np.str_]

    def __post_init__(self):
        """Ensure the consistency of input data."""
        assert (
            self.count_matrix.shape[0] == self.spatial_distances.shape[0]
        ), "Number of rows in count_matrix must match the number of spatial coordinates."
        assert (
            self.count_matrix.shape[0] == self.cell_types.shape[0]
        ), "Number of rows in count_matrix must match the number of cell type annotations."
        if not isinstance(self.cell_types, _np.ndarray):
            self.cell_types = _np.array(self.cell_types).astype(str)
        assert isinstance(self.spatial_distances, _csr_matrix)
        return


# DONE TODO: Use spatial_distances cache
def celltype_refined_bin(
    ann_count_matrix: SpatialTypeAnnCntMtx,
    bin_radius: float = 3.0,  # bin-7
    name_undefined: str = "Undefined",
    fraction_subsampling: float = 1.0,
    verbose: bool = True,
) -> SpatialTypeAnnCntMtx:
    """
    Aggregate each spot in the spatial transcriptome with its
    neighboring spots of the same cell-type, based on a specified distance metric.

    This function bins each spatial sample (spot) by aggregating the gene count data
    of its neighboring spots that share the same cell-type annotation. The distance
    between spots is measured based on the specified distance norm (e.g., Euclidean or
    Manhattan). The result is a new `SpatialTypeAnnCntMtx` object with aggregated gene
    counts for each spot, taking into account only the neighboring spots with the same cell type.

    Parameters:
    -----------
    ann_count_matrix : SpatialTypeAnnCntMtx
        A `SpatialTypeAnnCntMtx` object containing the gene count matrix, spatial coordinates,
        and cell-type annotations. The function operates on this matrix to perform spatial binning
        and aggregation.

    bin_radius : float, optional (default=3.0)
        The radius within which neighboring spots are considered for aggregation.
        Spots that are within this radius of each other will be grouped together for aggregation.

    name_undefined : str, optional (default="Undefined")
        The name of the undefined cell-type. Any aggregated sample with undefined type will
        be removed from the final result.

    fraction_subsampling : float, optional (default=1.0)
        The fraction of samples to randomly subsample, 0.0 to 1.0, to reduce memory taken.

    Returns:
    --------
    SpatialTypeAnnCntMtx
        A new `SpatialTypeAnnCntMtx` object with the aggregated gene counts for each spot,
        where each spot's gene count has been updated by aggregating its own count and the
        counts of its neighboring spots that share the same cell type.

    Example:
    --------
    # Assuming ann_count_matrix is a valid SpatialTypeAnnCntMtx object
    aggregated_mtx = celltype_refined_bin(
        ann_count_matrix,
        bin_radius=5.0,
    )

    # The result is a new SpatialTypeAnnCntMtx object with aggregated counts.
    """
    assert 0.0 < fraction_subsampling <= 1.0
    n_samples_raw: int = ann_count_matrix.count_matrix.shape[0]
    bools_defined: _NDArray[_np.bool_] = ~(
        ann_count_matrix.cell_types == name_undefined
    )
    if fraction_subsampling < 1.0:
        n_subsample: int = int(round(fraction_subsampling * len(bools_defined)))
        n_subsample = max(1, n_subsample)
        n_subsample = min(len(bools_defined), n_subsample)
        if verbose:
            _tqdm.write(
                f"Subsampling {fraction_subsampling:.2%} i.e. {n_subsample} samples.."
            )
        ilocs_keep_from_defined: _NDArray[_np.int_] = _np.random.choice(
            a=_np.arange(int(_np.sum(bools_defined))),
            size=n_subsample,
            replace=False,
        )
        ilocs_keep: _NDArray[_np.int_] = _np.arange(len(bools_defined))[bools_defined][
            ilocs_keep_from_defined
        ]
        bools_defined[:] = False
        bools_defined[ilocs_keep] = True
    ilocs_defined: _NDArray[_np.int_] = _np.arange(n_samples_raw)[bools_defined]
    n_samples_def: int = len(ilocs_defined)
    celltype_pool: set[str] = set(
        _np.unique(ann_count_matrix.cell_types[bools_defined])
    )
    # Load distance matrix
    if verbose:
        _tqdm.write("Loading spatial distances..")
    # Get subsampled items
    dist_mat: _csr_matrix = ann_count_matrix.spatial_distances[bools_defined, :].copy()
    dist_mat.eliminate_zeros()
    whr_nonzero = dist_mat.data <= bin_radius
    dist_mat: _csr_matrix = _csr_matrix(
        _coo_matrix(
            (
                dist_mat.data[whr_nonzero],
                (
                    dist_mat.nonzero()[0][whr_nonzero],
                    dist_mat.nonzero()[1][whr_nonzero],
                ),
            ),
            shape=dist_mat.shape,
        )
    )
    dist_dict: dict[str, _NDArray[_np.int_]] = {
        "rows": dist_mat.row,
        "cols": dist_mat.col,
    }
    del dist_mat
    # Mask out those of different type
    ilocs_items_keep: list[int] = []
    itor_ = (
        _tqdm(
            celltype_pool,
            desc="Building CTRBin",
            ncols=60,
        )
        if verbose
        else celltype_pool
    )
    for ct in itor_:
        icols_this_ct: _NDArray[_np.int_] = _np.arange(n_samples_raw)[
            ann_count_matrix.cell_types == ct
        ]
        irows_this_ct: _NDArray[_np.int_] = _np.arange(n_samples_def)[
            ann_count_matrix.cell_types[bools_defined] == ct
        ]
        bools_items_keeprows: _NDArray[_np.bool_] = _np.isin(
            element=dist_dict["rows"],
            test_elements=irows_this_ct,
        )
        bools_subrows_keepcols: _NDArray[_np.bool_] = _np.isin(
            element=dist_dict["cols"][bools_items_keeprows],
            test_elements=icols_this_ct,
        )
        ilocs_items_keep_this_ct: _NDArray[_np.int_] = _np.arange(
            dist_dict["rows"].shape[0]
        )[bools_items_keeprows][bools_subrows_keepcols]
        ilocs_items_keep += ilocs_items_keep_this_ct.tolist()

    # Add diagonals
    rows = _np.append(
        arr=dist_dict["rows"],
        values=_np.arange(n_samples_def),
    )
    cols = _np.append(
        arr=dist_dict["cols"],
        values=_np.arange(n_samples_def),
    )
    ilocs_items_keep += list(
        range(dist_dict["rows"].shape[0], dist_dict["rows"].shape[0] + n_samples_def)
    )
    del dist_dict

    # Building final weight matrix
    rows = rows[ilocs_items_keep]
    cols = cols[ilocs_items_keep]
    data = _np.ones(
        shape=(len(rows),),
    )
    weight_matrix: _coo_matrix = _coo_matrix(
        (data, (rows, cols)),
        shape=(n_samples_def, n_samples_raw),
    ).astype(float)

    weight_matrix: _csr_matrix = weight_matrix.tocsr()

    # Get result
    return SpatialTypeAnnCntMtx(
        count_matrix=weight_matrix @ ann_count_matrix.count_matrix,
        spatial_distances=ann_count_matrix.spatial_distances[bools_defined, :][
            :, bools_defined
        ].copy(),
        cell_types=ann_count_matrix.cell_types[bools_defined],
    )


@_dataclass
class SpTypeSizeAnnCntMtx:
    """
    A data class representing a gene count matrix with spatial coordinates,
    annotated cell types, and estimated cell sizes.

    Attributes:
    -----------
    count_matrix : scipy.sparse.csr_matrix
        A sparse matrix of shape (n_samples, n_genes) where each entry represents
        the count of a specific gene in a specific sample (or spatial location).

    spatial_distances : csr_matrix
        A 2D sparse array of shape (n_samples, n_samples), indicating distances between
        each sample, with all distances above a threshold being set to zero.

    cell_types : numpy.ndarray
        A 1D array of length n_samples where each element is a string representing
        the cell type annotation for the corresponding sample or cell.

    cell_sizes: numpy.ndarray
        A 1D array of length n_samples where each element is an integer representing
        the cell size (in spots) annotation for the corresponding sample or cell.

    Assertions:
    -----------
    - The number of rows in `count_matrix` must match the number of rows in `spatial_coords`
      (i.e., the number of spatial locations).
    - The number of rows in `count_matrix` must also match the number of entries in `cell_types`
      (i.e., the number of annotated cell types), and `cell_sizes`.

    Example:
    --------
    # Create a sparse count matrix, spatial coordinates, and cell types.
    count_matrix = csr_matrix([[5, 3], [4, 2], [6, 1]])
    spatial_coords = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]])
    cell_types = np.array(['TypeA', 'TypeB', 'TypeA'])
    cell_sizes = np.array([5, 6, 3])

    # Initialize the SpatialTypeAnnCntMtx object.
    mtx = SpTypeSizeAnnCntMtx(count_matrix, spatial_coords, cell_types, cell_sizes)
    """

    count_matrix: _csr_matrix
    spatial_distances: _csr_matrix  # of _NumberType
    cell_types: _1DArrayType  # of str
    cell_sizes: _1DArrayType  # of int
    cell_mask: _1DArrayType | _UndefinedType = _UNDEFINED  # of int

    def __post_init__(self):
        """Ensure the consistency of input data."""
        assert (
            self.count_matrix.shape[0] == self.spatial_distances.shape[0]
        ), "Number of rows in count_matrix must match the number of spatial coordinates."
        assert (
            self.count_matrix.shape[0] == self.cell_types.shape[0]
        ), "Number of rows in count_matrix must match the number of cell type annotations."
        if not isinstance(self.cell_types, _np.ndarray):
            self.cell_types = _np.array(self.cell_types).astype(str)
        assert (
            self.count_matrix.shape[0] == self.cell_sizes.shape[0]
        ), "Number of rows in count_matrix must match the number of cell size annotations"
        if not isinstance(self.cell_sizes, _np.ndarray):
            self.cell_sizes = _np.array(self.cell_sizes).astype(self.cell_sizes.dtype)
        assert isinstance(self.spatial_distances, _csr_matrix)
        if not isinstance(self.cell_mask, _UndefinedType):
            assert self.cell_mask.shape[0] == self.count_matrix.shape[0]
        return


# TODO: Parallel by chunking spatial_distances (50 hrs -> 50/n hrs)
# DONE TODO: Use spatial_distances cache.
def ctrbin_cellseg(
    ann_count_matrix: SpTypeSizeAnnCntMtx,
    coeff_overlap_constraint: float = 1.0,
    verbose: bool = True,
) -> _1DArrayType:
    """
    Cell-Type-Refined Bin with cell-size estimated.

    Aggregate each spot in the spatial transcriptome with its
    neighboring spots of the same cell-type. The number of aggregated spots within a cell
    will be no less than corresponding entries in `ann_count_matrix.cell_sizes`

    This function bins each spatial sample (spot) by aggregating the gene count data
    of its neighboring spots that share the same cell-type annotation. The distance
    between spots is measured based on the specified distance norm (e.g., Euclidean or
    Manhattan). The result is a 1d-array of cell id masks, with -1 indicating not-assigned.

    Parameters:
    -----------
    ann_count_matrix : SpatialTypeAnnCntMtx
        A `SpatialTypeAnnCntMtx` object containing the gene count matrix, spatial coordinates,
        and cell-type annotations. The function operates on this matrix to perform spatial binning
        and aggregation.

    coeff_overlap_constraint : float, optional (default=1.0)
        A coefficient used to constrain cell overlap. In detail, an estimated range is used for
        removing cell centroids that are too close. This coefficient is used to multiply the
        range so it increases (>1, resulting in less cells) or decreases (<1, resulting in more
        cells) a little bit.

    Returns:
    --------
    The result is a 1d-array of cell id masks, with -1 indicating not-assigned.
    """
    n_samples_raw: int = ann_count_matrix.count_matrix.shape[0]
    # Estimate overlap ranges by coeff * 2 * sqrt(S/pi)
    ranges_overlap: _1DArrayType = (
        coeff_overlap_constraint * 2 * _np.sqrt(ann_count_matrix.cell_sizes / _np.pi)
    )
    # Calculate n_counts for each spot
    ns_counts: _1DArrayType = _to_array(
        ann_count_matrix.count_matrix.sum(axis=1), squeeze=True
    )
    ix_sorted_by_counts: _1DArrayType = _np.argsort(ns_counts)[::-1]
    if verbose:
        _tqdm.write(f"Loading spatial distances..")
    dist_matrix: dict = {
        "rows": [],
        "cols": [],
        "data": [],
    }
    ann_count_matrix.spatial_distances.eliminate_zeros()
    whr_nonzero = ann_count_matrix.spatial_distances.data <= _np.max(ranges_overlap)
    dist_matrix["data"] = ann_count_matrix.spatial_distances.data[whr_nonzero]
    dist_matrix["rows"] = ann_count_matrix.spatial_distances.nonzero()[0][whr_nonzero]
    dist_matrix["cols"] = ann_count_matrix.spatial_distances.nonzero()[1][whr_nonzero]
    dist_matrix: _csr_matrix = _csr_matrix(
        _coo_matrix(
            (dist_matrix["data"], (dist_matrix["rows"], dist_matrix["cols"])),
            shape=ann_count_matrix.spatial_distances.shape,
        )
    )
    cell_candidates_bool: _1DArrayType = _np.ones(
        shape=(n_samples_raw,),
        dtype=_np.bool_,
    )
    cell_masks: _1DArrayType = (
        _np.zeros(
            shape=(n_samples_raw,),
            dtype=_np.int_,
        )
        - 1
    )  # -1 indicates for unassigned
    if verbose:
        itor_ = _tqdm(
            range(n_samples_raw),
            desc="Finding cells",
            ncols=60,
        )
    else:
        itor_ = range(n_samples_raw)
    for _i in itor_:
        i_centroid: int = ix_sorted_by_counts[_i]
        # If removed, skip.
        if cell_candidates_bool[i_centroid] is False:
            continue
        ix_nbors: _1DArrayType = dist_matrix.getrow(i_centroid).nonzero()[1]
        # Filter out different types
        label_centroid: str = ann_count_matrix.cell_types[i_centroid]
        whr_sametype: _1DArrayType = (
            ann_count_matrix.cell_types[ix_nbors] == label_centroid
        )
        ix_nbors = ix_nbors[whr_sametype]
        if len(ix_nbors) == 0:
            cell_masks[i_centroid] = i_centroid
            continue
        dists_nbors: _1DArrayType = _to_array(
            dist_matrix[i_centroid, ix_nbors], squeeze=True
        )
        # Find neighbors of estimated size
        ix_aggregate = ix_nbors[
            _np.argsort(dists_nbors)[
                : max(1, ann_count_matrix.cell_sizes[i_centroid] - 1)
            ]
        ]
        cell_masks[ix_aggregate] = i_centroid
        cell_masks[i_centroid] = i_centroid
        # Remove too-close from centroid candidates
        cell_candidates_bool[ix_nbors[dists_nbors < ranges_overlap[i_centroid]]] = False
    return cell_masks


def ctrbin_cellseg_parallel(
    ann_count_matrix: SpTypeSizeAnnCntMtx,
    spatial_coordinates: _Nx2ArrayType,
    coeff_overlap_constraint: float = 1.0,
    n_workers: int = 40,
    verbose: bool = True,
) -> _1DArrayType:
    """
    Experimental. Use within `if __name__=='__main__':` statement!
    
    See `ctrbin_cellseg` for params.

    Needs to provide `spatial_coordinates` corresponding to samples in
    `ann_count_matrix` to produce chunks.

    Chunkwise parallel operation might bring a little inaccuracy to cell segmentation.
    """
    if verbose:
        _tqdm.write(f"Estimating chunk config..")
    xrange: tuple[_NumberType, _NumberType] = (
        spatial_coordinates[:, 0].min(),
        spatial_coordinates[:, 0].max(),
    )
    yrange: tuple[_NumberType, _NumberType] = (
        spatial_coordinates[:, 1].min(),
        spatial_coordinates[:, 1].max(),
    )
    assert xrange[1] - xrange[0] > 0
    assert yrange[1] - yrange[0] > 0
    yxratio: float = (yrange[1] - yrange[0]) / (xrange[1] - xrange[0])
    n_chunks_along_x: int = max(1, int(_np.sqrt(n_workers / yxratio)))
    n_chunks_along_y: int = max(1, int(n_chunks_along_x * yxratio))
    cut_points_x: _1DArrayType = _np.linspace(
        start=xrange[0],
        stop=xrange[1],
        num=n_chunks_along_x + 1,
    )
    cut_points_y: _1DArrayType = _np.linspace(
        start=yrange[0],
        stop=yrange[1],
        num=n_chunks_along_y + 1,
    )
    if verbose:
        _tqdm.write(
            f"Chunk size: {cut_points_x[1]-cut_points_x[0]:.1f} x {cut_points_y[1]-cut_points_y[0]:.1f} (The ones at the edges might be a little bit bigger)"
        )
        _tqdm.write(f"Total {n_chunks_along_x} x {n_chunks_along_y} chunks.")
    cut_points_x[0] = -_np.inf
    cut_points_x[-1] = _np.inf
    cut_points_y[0] = -_np.inf
    cut_points_y[-1] = _np.inf
    # Build fovs (field of view)
    if verbose:
        _tqdm.write("Building chunks..")
    fov_table: dict = {
        "upperleft": [],  # tuple
        "lowerright": [],  # tuple
        "indices": [],  # array
    }
    fov_table["upperleft"] = list(_product(cut_points_x[:-1], cut_points_y[:-1]))
    fov_table["lowerright"] = list(_product(cut_points_x[1:], cut_points_y[1:]))
    fov_table["indices"] = [
        _np.arange(spatial_coordinates.shape[0])[
            (spatial_coordinates[:, 0] >= fov_table["upperleft"][i_fov][0])
            * (spatial_coordinates[:, 0] < fov_table["lowerright"][i_fov][0])
            * (spatial_coordinates[:, 1] >= fov_table["upperleft"][i_fov][1])
            * (spatial_coordinates[:, 1] < fov_table["lowerright"][i_fov][1])
        ]
        for i_fov in range(len(fov_table["upperleft"]))
    ]
    assert len(fov_table["indices"]) == len(fov_table["lowerright"])
    if verbose:
        _tqdm.write("Chunks ready. Allocating jobs..")
    # Chunk items: (ann_count_matrix[chunked], coeff, verbose)
    chunks: list[tuple] = [
        (
            SpTypeSizeAnnCntMtx(
                count_matrix=ann_count_matrix.count_matrix[
                    fov_table["indices"][i_fov], :
                ].copy(),
                spatial_distances=ann_count_matrix.spatial_distances[
                    fov_table["indices"][i_fov], :
                ][:, fov_table["indices"][i_fov]].copy(),
                cell_types=ann_count_matrix.cell_types[fov_table["indices"][i_fov]],
                cell_sizes=ann_count_matrix.cell_sizes[fov_table["indices"][i_fov]],
            ),
            coeff_overlap_constraint,
            verbose,
        )
        for i_fov in range(len(fov_table["indices"]))
    ]
    if verbose:
        _tqdm.write("Running jobs..")
    with _Pool(len(fov_table["indices"])) as _p:
        results: list[_1DArrayType] = _p.starmap(func=ctrbin_cellseg, iterable=chunks)
    if verbose:
        _tqdm.write("Gathering and sorting results..")
    cellmasks_global: list[_NumberType] = []
    indices_global: list[_NumberType] = []
    for i_fov in range(len(results)):
        cellmasks_sub: _1DArrayType = results[i_fov]
        cellmasks_remap: _1DArrayType = cellmasks_sub.copy()
        original_indices: _1DArrayType = fov_table["indices"][i_fov]
        indices_global += list(original_indices)
        for ix_sub, ix_original in enumerate(original_indices):
            whr_thisix = cellmasks_sub == ix_sub
            if whr_thisix.sum() == 0:
                continue
            cellmasks_remap[whr_thisix] = ix_original
        cellmasks_global += list(cellmasks_remap)
    cellmasks_final: _1DArrayType = _np.array(
        sorted(list(zip(indices_global, cellmasks_global)))
    )[:, 1]
    if verbose:
        _tqdm.write("Done.")
    return cellmasks_final


# Utilities
def cluster_spatial_domain(
    coords: _NDArray[_np.float_],
    cell_types: _NDArray[_np.str_],
    radius_local: float = 10.0,
    n_clusters: int = 9,
    algorithm: _Literal["agglomerative", "kmeans"] = "agglomerative",
    on_grids: bool = True,
    return_grids_coords: bool = True,
    grid_size: float = 10.0,
) -> _NDArray[_np.int_] | tuple[_NDArray[_np.int_], _NDArray[_np.float_]]:
    """
    Cluster spatial spots into many domains based on
    cell-tpye proportion.

    Args:
        coords: n x 2 array, each row indicating spot location.
        cell_types: array of cell types of each spot.
        radius_local: radius of sliding window to compute cell-type proportion.
        n_clusters: number of clusters generated.
        on_grids: if True, clusters grids coordinates instead of spots coords (recommended for memory ease).
        grid_size: if `on_grid` is True, specifies the unit size of each grid.

    Return:
        tuple[_NDArray[_np.int_], _NDArray[_np.float_]]: (domain_ids, grids_coordinates), if `return_grids_coords`;
        otherwise _NDArray[_np.int_], an array of cluster indices in corresponding order.
    """
    # Validate params
    n_samples: int = coords.shape[0]
    assert n_samples == cell_types.shape[0]
    assert coords.shape[1] == 2
    assert len(coords.shape) == 2
    assert len(cell_types.shape) == 1
    assert algorithm in ["agglomerative", "kmeans"]
    if n_clusters > 10:
        _tqdm.write(f"Warning: {n_clusters=} could be large, might be a memory hog")

    # Create distance matrix
    # Build grids
    if on_grids:
        xmin = _np.min(coords[:, 0])
        xmax = _np.max(coords[:, 0])
        ymin = _np.min(coords[:, 1])
        ymax = _np.max(coords[:, 1])
        Xs = _np.arange(
            start=xmin,
            stop=xmax,
            step=grid_size,
        )
        Ys = _np.arange(
            start=ymin,
            stop=ymax,
            step=grid_size,
        )
        coords_grids = _np.array([[x, y] for x in Xs for y in Ys])
        del Xs
        del Ys
    else:
        coords_grids = coords
    ckdtree_grids = _cKDTree(coords_grids)

    n_grids: int = coords_grids.shape[0]
    ckdtree_spots = _cKDTree(coords)
    dist_matrix: _coo_matrix = ckdtree_grids.sparse_distance_matrix(
        other=ckdtree_spots,
        max_distance=radius_local,
        p=2,
        output_type="coo_matrix",
    )
    dist_matrix: _csr_matrix = dist_matrix.tocsr()

    # Create celltype-proportion observation matrix
    celltypes_unique: _NDArray[_np.str_] = _np.sort(
        _np.unique(cell_types)
    )  # alphabetically sort
    obs_matrix: _NDArray[_np.float_] = _np.zeros(
        shape=(n_grids, celltypes_unique.shape[0]),
        dtype=float,
    )
    for i_grid in _tqdm(
        range(n_grids),
        desc="Compute celltype proportions",
        ncols=60,
    ):
        dist_nbors = _to_array(dist_matrix[i_grid, :], squeeze=True)
        iloc_nbors = _np.where(dist_nbors > 0)[0]
        if len(iloc_nbors) == 0:
            continue
        ct_nbors = cell_types[iloc_nbors]
        for i_ct, ct in enumerate(celltypes_unique):
            obs_matrix[i_grid, i_ct] = (ct_nbors == ct).mean()

    if algorithm == "agglomerative":
        # Agglomerative cluster
        _tqdm.write("Agglomerative clustering..")
        Z = _linkage(
            obs_matrix,
            method="ward",
        )
        cluster_labels = _fcluster(
            Z,
            t=n_clusters,
            criterion="maxclust",
        )
        _tqdm.write("Done.")
    else:
        # Kmeans
        _tqdm.write("KMeans clustering..")
        kmeans = _MiniBatchKMeans(
            n_clusters=n_clusters,
            n_init=1,
            verbose=1,
        )
        cluster_labels = kmeans.fit_predict(
            X=obs_matrix,
        )
        _tqdm.write("Done.")

    if return_grids_coords:
        return (
            cluster_labels,
            coords_grids,
        )
    else:
        return cluster_labels


def aggregate_spots_to_cells(
    st_anndata: _AnnData,
    obs_name_cell_id: str = "cell_id_pytacs",
    obs_name_cell_type: str | None = "cell_type_pytacs",
    verbose: bool = True,
) -> _AnnData:
    """
    Aggregate spatial transcriptomics spots into single-cell resolution using a cell ID annotation.

    Parameters
    ----------
    st_anndata : AnnData
        The input AnnData object where each observation (row) corresponds to a spatial spot.
    obs_name_cell_id : str
        The name of the column in `st_anndata.obs` that contains cell ID annotations.
        Spots with the same cell ID will be aggregated together.

    Returns
    -------
    AnnData
        A new AnnData object where each observation corresponds to a single cell, obtained by
        aggregating the gene expression and spatial coordinates (if available) of all
        spots belonging to the same cell ID.
    """
    cell_id_pool: _1DArrayType = _np.unique(st_anndata.obs[obs_name_cell_id].values)
    whr_def: _1DArrayType = cell_id_pool != -1
    cell_id_pool = cell_id_pool[whr_def]
    del whr_def
    if verbose:
        itor_ = _tqdm(
            range(len(cell_id_pool)),
            desc="Aggregating spots",
            ncols=60,
        )
    else:
        itor_ = range(len(cell_id_pool))
    X_sc: _lil_matrix = _lil_matrix((len(cell_id_pool), st_anndata.X.shape[1])).astype(
        int
    )
    sp_coords: _Nx2ArrayType = _np.empty(
        shape=(X_sc.shape[0], 2),
        dtype=float,
    )
    # df_obs: _pd.DataFrame = st_anndata.obs.loc[cell_id_pool.astype(str), :].copy()  # buggy
    celltype_obs: list[str] = []
    for i_cellid in itor_:
        cellid: int = cell_id_pool[i_cellid]
        whr_thiscell: _1DArrayType = st_anndata.obs[obs_name_cell_id].values == cellid
        X_sc[i_cellid, :] = st_anndata.X[whr_thiscell, :].sum(axis=0)
        if "spatial" in st_anndata.obsm:
            sp_coords[i_cellid, :] = st_anndata.obsm["spatial"][whr_thiscell, :].mean(
                axis=0
            )
        if obs_name_cell_type is not None:
            celltype_obs.append(st_anndata.obs.loc[whr_thiscell, obs_name_cell_type].values[0])
    res = _AnnData(
        X=X_sc.tocsr(),
        # obs=df_obs,  # buggy
        var=st_anndata.var.copy(),
        obsm={"spatial": sp_coords} if "spatial" in st_anndata.obsm else None,
    )
    res.obs[obs_name_cell_id] = cell_id_pool
    if obs_name_cell_type is not None:
        res.obs[obs_name_cell_type] = celltype_obs
    return res


def aggregate_spots_to_cells_parallel(
    st_anndata: _AnnData,
    obs_name_cell_id: str = "cell_id_pytacs",
    obs_name_cell_type: str | None = 'cell_type_pytacs',
    n_workers: int = 10,
    verbose: bool = True,
) -> _AnnData:
    """
    Aggregate spatial transcriptomics spots into single-cell resolution using a cell ID annotation.

    Parameters
    ----------
    st_anndata : AnnData
        The input AnnData object where each observation (row) corresponds to a spatial spot.
    obs_name_cell_id : str
        The name of the column in `st_anndata.obs` that contains cell ID annotations.
        Spots with the same cell ID will be aggregated together.

    Returns
    -------
    AnnData
        A new AnnData object where each observation corresponds to a single cell, obtained by
        aggregating the gene expression and spatial coordinates (if available) of all
        spots belonging to the same cell ID.
    """
    ix_argsort: _1DArrayType = _np.argsort(st_anndata.obs[obs_name_cell_id].values)
    cellix_sort: _1DArrayType = st_anndata.obs[obs_name_cell_id].values[ix_argsort]
    change_points_sort: _1DArrayType = _np.where(cellix_sort[:-1] != cellix_sort[1:])[0]
    n_changepoints_per_chunk: int = len(change_points_sort) // n_workers
    chunks: list[_AnnData] = []
    if verbose:
        _tqdm.write(f"Allocating {n_workers} jobs..")
    for i_job in range(n_workers):
        if i_job == 0:
            chunks.append(
                (
                    st_anndata[
                        ix_argsort[
                            : change_points_sort[(i_job + 1) * n_changepoints_per_chunk]+1
                        ],
                        :,
                    ].copy(),
                    obs_name_cell_id,
                    obs_name_cell_type,
                    verbose,
                )
            )
            continue
        if i_job == n_workers - 1:
            chunks.append(
                (
                    st_anndata[
                        ix_argsort[
                            change_points_sort[i_job * n_changepoints_per_chunk]+1:
                        ],
                        :,
                    ].copy(),
                    obs_name_cell_id,
                    obs_name_cell_type,
                    verbose,
                )
            )
            continue
        chunks.append(
            (
                st_anndata[
                    ix_argsort[
                        change_points_sort[
                            i_job * n_changepoints_per_chunk
                        ]+1 : change_points_sort[(i_job + 1) * n_changepoints_per_chunk]+1
                    ],
                    :,
                ].copy(),
                obs_name_cell_id,
                obs_name_cell_type,
                verbose,
            )
        )
    with _Pool(n_workers) as p_:
        results: list[_AnnData] = p_.starmap(
            func=aggregate_spots_to_cells, iterable=chunks
        )
    if verbose:
        _tqdm.write(f"Gathering results..")
    res = _sc.concat(adatas=results, axis="obs", join="inner")
    if verbose:
        _tqdm.write("Done.")
    return res
