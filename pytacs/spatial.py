"""A new version of spatial strategy based on Randon Walk, with fast computation,
low mem cost, and robust performance."""

from scanpy import AnnData as _AnnData
import numpy as _np
from numpy.typing import NDArray as _NDArray
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.sparse import coo_matrix as _coo_matrix
from scipy.sparse import lil_matrix as _lil_matrix
from scipy.spatial import cKDTree as _cKDTree  # to construct sparse distance matrix

from scipy.cluster.hierarchy import linkage as _linkage
from scipy.cluster.hierarchy import fcluster as _fcluster

from .classifier import _LocalClassifier
from .utils import to_array as _to_array

from tqdm import tqdm as _tqdm
from dataclasses import dataclass as _dataclass
import pandas as _pd
from typing import Literal as _Literal
from sklearn.decomposition import TruncatedSVD as _TruncatedSVD
from .utils import normalize_csr as _normalize_csr
from .utils import trim_csr_per_row as _trim_csr_per_row
from .utils import rowwise_cosine_similarity as _rowwise_cosine_similarity


@_dataclass
class AggregationResult:
    """Results of spot aggregation.

    Attrs:
        .dataframe (pd.DataFrame):
            ["cell_id"]: (int) centroid spot id
            ["cell_type"]: (str) cell-type name
            ["confidence"]: (float) probability of that class

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
    trim_proportion: float = 0.5,
    mode_walk: _Literal["rw"] = "rw",
    return_weight_matrix: bool = False,
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

        trim_proportion (float, optional):
            Proportion of nodes to trim off after each iter of random walk, in order to prevent accumulation of
            erroneous spots.

        mode_walk (Literal['rw', 'rwr'], optional):
            Type of random walk to perform:
            'rw' for vanilla random walk,
            'rwr' for random walk with restart. Default is 'rw'.

        return_weight_matrix (bool, optional):
            If True, will return weight_matrix in AggregationResult. Note this process may drastically increase
            computational time!

    Returns:
        AggregationResult:
            A dataclass containing:
                - `dataframe`: DataFrame with predicted `cell_id`, `cell_type`, and confidence scores.
                - `expr_matrix`: CSR matrix of aggregated expression for confident spots.
                - `weight_matrix`: CSR matrix representing transition probabilities between all spots, if `return_weight_matrix`;
                otherwise an empty matrix.
    """
    assert mode_embedding in ["raw", "pc"]
    assert mode_metric in ["inv_dist", "cosine"]
    assert mode_aggregation in ["weighted", "unweighted"]
    assert trim_proportion >= 0.0 and trim_proportion < 1.0
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
    ckdtree_spatial = _cKDTree(st_anndata.obsm["spatial"])
    distances_spatial = ckdtree_spatial.sparse_distance_matrix(
        other=ckdtree_spatial,
        max_distance=nbhd_radius,
        output_type="coo_matrix",
    )
    distances_propagation = ckdtree_spatial.sparse_distance_matrix(
        other=ckdtree_spatial,
        max_distance=max_propagation_radius,
        output_type="coo_matrix",
    )
    # Boundaries for propagation
    ilocs_propagation = _np.array(
        list(zip(distances_propagation.row, distances_propagation.col))
    )

    rows_nonzero, cols_nonzero = (
        ilocs_propagation[:, 0],
        ilocs_propagation[:, 1],
    )
    del ilocs_propagation
    rows_nonzero = _np.concatenate(
        [rows_nonzero, _np.arange(distances_propagation.shape[0])]
    )
    cols_nonzero = _np.concatenate(
        [cols_nonzero, _np.arange(distances_propagation.shape[0])]
    )
    del distances_propagation
    query_pool_propagation = set(zip(rows_nonzero, cols_nonzero))
    del rows_nonzero
    del cols_nonzero
    distances = _lil_matrix(
        (distances_spatial.shape[0], distances_spatial.shape[1]),
    )
    # Get defined embedding
    embeds: _np.ndarray = X_normalized @ embed_loadings.T
    del X_normalized
    del embed_loadings
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
    # Normalize similarities row-wise
    similarities = _normalize_csr(similarities)

    # candidate cellids, gonna shrink with iterations
    candidate_cellids = _np.arange(st_anndata.shape[0])
    # cellids confident, gonna bloat with iters
    cellids_confident = []
    # celltypes corrsponding to cellids_confident, gonna bloat
    celltypes_confident = []
    # confidences corresponding to celltypes_confident, gonna bloat
    confidences_confident = []
    # final weight_matrix of all spots, gonna update
    # weight_matrix = _lil_matrix((similarities.shape[0], similarities.shape[1]))
    # Modified: using efficient construction with coo
    weight_matrix: dict = {
        "rows": [],
        "cols": [],
        "data": [],
    }
    # Random Walk
    counter_celltypes_global = dict()
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
        counter_celltypes = dict()
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
                    # weight_matrix[cellid, :] = similarities[cellid, :]
                    # Modified version with coo
                    weight_matrix["rows"].append(cellid)
                    cols_nonzero = similarities.getrow(cellid).nonzero()[1]
                    weight_matrix["cols"] += cols_nonzero
                    weight_matrix["data"] += _to_array(
                        similarities[cellid, cols_nonzero],
                        squeeze=True,
                    ).tolist()
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
        # Trim
        if trim_proportion == 0.0:
            continue
        similarities = _trim_csr_per_row(
            csr_mat=similarities,
            trim_proportion=trim_proportion,
            tqdm_verbose=verbose,
        )

    weight_matrix: _coo_matrix = _coo_matrix(
        (weight_matrix["data"], (weight_matrix["rows"], weight_matrix["cols"])),
        shape=(st_anndata.X.shape[0], st_anndata.X.shape[0]),
    )
    weight_matrix: _csr_matrix = weight_matrix.tocsr()
    # Construct Results
    return AggregationResult(
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

    spatial_coords : numpy.ndarray
        A 2D array of shape (n_samples, 2), where each row contains the spatial
        coordinates (e.g., x, y) corresponding to the sample or cell.

    cell_types : numpy.ndarray
        A 1D array of length n_samples where each element is a string representing
        the cell type annotation for the corresponding sample or cell.

    Assertions:
    -----------
    - The number of rows in `count_matrix` must match the number of rows in `spatial_coords`
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
    spatial_coords: _np.ndarray
    cell_types: _NDArray[_np.str_]

    def __post_init__(self):
        """Ensure the consistency of input data."""
        assert (
            self.count_matrix.shape[0] == self.spatial_coords.shape[0]
        ), "Number of rows in count_matrix must match the number of spatial coordinates."
        assert (
            self.count_matrix.shape[0] == self.cell_types.shape[0]
        ), "Number of rows in count_matrix must match the number of cell type annotations."
        if not isinstance(self.cell_types, _np.ndarray):
            self.cell_types = _np.array(self.cell_types).astype(str)
        assert isinstance(self.spatial_coords, _np.ndarray)
        return


def celltype_refined_bin(
    ann_count_matrix: SpatialTypeAnnCntMtx,
    bin_radius: float = 3.0,  # bin-7
    bin_norm_p: int = 2,  # 2 for Euclidean and 1 for Manhattan
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

    bin_norm_p : int, optional (default=2)
        The p-norm to use for distance computation between spots.
        - `p=2` corresponds to the **Euclidean distance**.
        - `p=1` corresponds to the **Manhattan distance**.

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
        bin_norm_p=2
    )

    # The result is a new SpatialTypeAnnCntMtx object with aggregated counts.
    """
    assert 0.0 < fraction_subsampling <= 1.0
    n_samples_raw: int = ann_count_matrix.count_matrix.shape[0]
    bools_defined: _NDArray[_np.bool_] = ~(
        ann_count_matrix.cell_types == name_undefined
    )
    if fraction_subsampling > 0.0:
        n_subsample: int = int(round(fraction_subsampling * len(bools_defined)))
        n_subsample = max(1, n_subsample)
        n_subsample = min(len(bools_defined), n_subsample)
        if verbose:
            _tqdm.write(
                f"Subsampling {fraction_subsampling:%} i.e. {n_subsample} samples.."
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
    # Build distance matrix
    if verbose:
        _tqdm.write("Building spatial distance matrix..")
    ckdtree_defined = _cKDTree(
        data=ann_count_matrix.spatial_coords[bools_defined, :],
    )
    ckdtree_all = _cKDTree(
        data=ann_count_matrix.spatial_coords,
    )
    dist_mat: _coo_matrix = ckdtree_defined.sparse_distance_matrix(
        other=ckdtree_defined,
        max_distance=bin_radius,
        p=bin_norm_p,
        output_type="coo_matrix",
    )
    del ckdtree_all
    del ckdtree_defined
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
        spatial_coords=ann_count_matrix.spatial_coords[bools_defined, :],
        cell_types=ann_count_matrix.cell_types[bools_defined],
    )


# Utilities
def cluster_spatial_domain(
    coords: _NDArray[_np.float_],
    cell_types: _NDArray[_np.str_],
    radius_local: float = 10.0,
    n_clusters: int = 9,
) -> _NDArray[_np.int_]:
    """
    Cluster spatial spots into many domains based on
    cell-tpye proportion.

    Args:
        coords: n x 2 array, each row indicating spot location.
        cell_types: array of cell types of each spot.
        radius_local: radius of sliding window to compute cell-type proportion.
        n_clusters: number of clusters generated.

    Return:
        array of cluster indices in corresponding order.
    """
    # Validate params
    n_samples: int = coords.shape[0]
    assert n_samples == cell_types.shape[0]
    assert coords.shape[1] == 2
    assert len(coords.shape) == 2
    assert len(cell_types.shape) == 1

    # Create distance matrix
    ckdtree = _cKDTree(coords)
    dist_matrix: _coo_matrix = ckdtree.sparse_distance_matrix(
        other=ckdtree,
        max_distance=radius_local,
        p=2,
        output_type="coo_matrix",
    )

    # Create celltype-proportion observation matrix
    celltypes_unique: _NDArray[_np.str_] = _np.sort(
        _np.unique(cell_types)
    )  # alphabetically sort
    obs_matrix: _NDArray[_np.float_] = _np.zeros(
        shape=(n_samples, cell_types.shape[0]),
        dtype=float,
    )
    for i_sample in _tqdm(
        range(n_samples),
        desc="Compute celltype proportions",
        ncols=60,
    ):
        dist_nbors = _to_array(dist_matrix[i_sample, :], squeeze=True)
        dist_nbors[i_sample] = 1.0
        iloc_nbors = _np.where(dist_nbors > 0)[0]
        ct_nbors = cell_types[iloc_nbors]
        for i_ct, ct in enumerate(celltypes_unique):
            obs_matrix[i_sample, i_ct] = (ct_nbors == ct).mean()

    # Agglomerative cluster
    _tqdm.write("Agglomerative clustering ...")
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
    return cluster_labels
