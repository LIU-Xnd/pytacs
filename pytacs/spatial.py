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
    max_iter: int = 3,
    steps_per_iter: int = 1,
    nbhd_radius: float = 1.5,
    max_propagation_radius: float = 4.5,
    mode_embedding: _Literal["raw", "pc"] = "pc",
    n_pcs: int = 30,
    mode_metric: _Literal["inv_dist"] = "inv_dist",
    mode_aggregation: _Literal["weighted"] = "weighted",
    mode_walk: _Literal["rw"] = "rw",
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
            Number of refinement iterations to perform. Default is 3.

        steps_per_iter (int, optional):
            Number of random walk steps to perform in each iteration. Default is 1.

        nbhd_radius (float, optional):
            Radius for defining local neighborhood in spatial graph construction. Default is 1.5.

        max_propagation_radius (float, optional):
            Radius for maximum possible random walk distance in spatial graph propagation. Default is 4.5.

        mode_embedding (Literal['raw', 'pc'], optional):
            Embedding mode for similarity calculation.
            'raw' uses the original expression matrix; 'pc' uses PCA-reduced data. Default is 'pc'.

        n_pcs (int, optional):
            Number of principal components to retain when `mode_embedding='pc'`. Default is 100.

        mode_metric (Literal['inv_dist'], optional):
            Distance or similarity metric to define transition weights between spots. Default is 'inv_dist'.

        mode_aggregation (Literal['unweighted', 'weighted'], optional):
            Aggregation strategy to combine neighborhood gene expression.
            'unweighted' uses uniform averaging; 'weighted' uses transition probabilities. Default is 'weighted'.

        mode_walk (Literal['rw', 'rwr'], optional):
            Type of random walk to perform:
            'rw' for vanilla random walk,
            'rwr' for random walk with restart. Default is 'rw'.

    Returns:
        AggregationResult:
            A dataclass containing:
                - `dataframe`: DataFrame with predicted `cell_id`, `cell_type`, and confidence scores.
                - `expr_matrix`: CSR matrix of aggregated expression for confident spots.
                - `weight_matrix`: CSR matrix representing transition probabilities between all spots.
    """
    assert mode_embedding in ["raw", "pc"]
    assert mode_metric in ["inv_dist"]
    assert mode_aggregation in ["weighted"]
    assert mode_walk in ["rw"]

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
        _tqdm.write(f"Performing truncated PCA (n_pcs={n_pcs})..")
        svd.fit(
            X=_normalize_csr(st_anndata.X.astype(float)),
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
    embeds: _np.ndarray = _normalize_csr(st_anndata.X.astype(float)) @ embed_loadings.T
    ilocs_nonzero = _np.array(list(zip(distances_spatial.row, distances_spatial.col)))
    distances[ilocs_nonzero[:, 0], ilocs_nonzero[:, 1]] = _np.linalg.norm(
        embeds[ilocs_nonzero[:, 0], :] - embeds[ilocs_nonzero[:, 1], :], axis=1
    )
    distances = distances.tocoo()

    # Compute inv_dist similarity using sparse operations: S_ij = 1 / (1 + d_ij)
    similarities = _coo_matrix(
        (1 / (1 + distances.data), (distances.row, distances.col)),
        shape=distances.shape,
    )
    similarities = similarities.tolil()
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
    weight_matrix = _lil_matrix((similarities.shape[0], similarities.shape[1]))
    # Random Walk
    counter_conf_global = dict()
    for i_iter in range(max_iter):
        # Aggregate spots according to similarities
        X_agg_candidate: _csr_matrix = similarities[
            candidate_cellids, :
        ] @ st_anndata.X.astype(float)
        # Classify
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
        counter_conf = dict()
        ave_conf = 0.0
        for i_candidate in _tqdm(
            range(len(candidate_cellids)),
            desc=f"Gather iter {i_iter+1} results",
            ncols=60,
        ):
            cellid: int = candidate_cellids[i_candidate]
            is_conf: bool = whr_confident_candidate[i_candidate]
            conf = confidences_candidate[i_candidate]
            if is_conf:
                typeid: int = type_ids_candidate[i_candidate]
                typename: str = classifier._classes[typeid]
                cellids_confident.append(cellid)
                celltypes_confident.append(typename)
                confidences_confident.append(conf)
                weight_matrix[cellid, :] = similarities[cellid, :]
                counter_conf[typename] = counter_conf.get(typename, 0) + 1
                counter_conf_global[typename] = counter_conf_global.get(typename, 0) + 1
            ave_conf += conf

        _tqdm.write(f"Ave conf: {ave_conf/candidate_cellids.shape[0]:.2%}")
        candidate_cellids = candidate_cellids[~whr_confident_candidate]
        _tqdm.write(f"{counter_conf=}")
        _tqdm.write(f"{counter_conf_global=}")
        if len(candidate_cellids) == 0:
            break
        # Random walk
        for i_step in _tqdm(
            range(steps_per_iter),
            desc="Random walk..",
            ncols=60,
        ):
            similarities: _csr_matrix = similarities @ similarities
            # Truncate propagation by max_propagation_radius for fast computation and stability.
            similarities: _coo_matrix = similarities.tocoo()
            # including diagonals

            _tqdm.write("Trimming..")
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

    celltypes_full = _np.empty(
        shape=(aggregation_result.weight_matrix.shape[0],),
        dtype=str,
    )
    celltypes_full[:] = name_undefined
    cellids_conf = aggregation_result.dataframe["cell_id"].values
    celltypes_conf = aggregation_result.dataframe["cell_type"].values
    celltypes_full[cellids_conf] = celltypes_conf[:]

    return celltypes_full


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
