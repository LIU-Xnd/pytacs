from .types import (
    _AnnData,
    _Axes,
    _NDArray,
    _csr_matrix,
    _dok_matrix,
    _lil_matrix,
    _Literal,
    _Iterable,
    _UNDEFINED,
    _UndefinedType,
)
from collections import defaultdict as _defaultdict

import scanpy as _sc
import pandas as _pd
import numpy as _np


from scipy.spatial import cKDTree as _cKDTree

from scipy.cluster.hierarchy import linkage as _linkage
from scipy.cluster.hierarchy import fcluster as _fcluster
from scipy.sparse import hstack as _hstack


from tqdm import tqdm as _tqdm
from .utils import reinit_index as _reinit_index
from .utils import to_array as _to_array
from .utils import truncate_top_n as _truncate_top_n

from .classifier import SVM as _SVM

from collections import Counter as _Counter


class AnnDataPreparer:
    """
    Initializes an object for processing snRNA-seq and spatial transcriptomic data
    for PyTACS. This method ensures that the input AnnData objects meet required
    conditions and applies transformations as needed.

    Upon initialization:
    - Copies of `sn_adata` and `sp_adata` are stored in `self.sn_adata` and `self.sp_adata`.
    - `sn_adata` is filtered to retain only genes that overlap with `sp_adata`, as these
      are used to train local classifiers.
    - If `sn_colname_celltype` differs from `"cell_type"`, it is renamed accordingly.
    - If `sp_obsmname_xycoords` differs from `"spatial"`, it is renamed accordingly.
    - If the number of overlapping genes is below `overlapped_genes_warning`, a warning
      message is printed.

    Args:
        sn_adata (_AnnData, optional): Single-nucleus or single-cell RNA-seq AnnData object
            (Scanpy object). Used to train local classifiers. Defaults to None.
        sp_adata (_AnnData, optional): Spatial transcriptomic AnnData object (Scanpy object).
            Defaults to None.
        sn_colname_celltype (str, optional): Column name in `sn_adata.obs` that contains cell-type
            annotations. If not `"cell_type"`, it will be renamed. Defaults to `"cell_type"`.
        sp_obsmname_xycoords (str, optional): Key in `sp_adata.obsm` that stores spatial coordinates.
            If not `"spatial"`, it will be renamed. Defaults to `"spatial"`.
        overlapped_genes_warning (int, optional): Threshold below which a warning is printed
            if the number of overlapping genes between `sn_adata` and `sp_adata` is too low.
            Defaults to 10.

    Raises:
        AssertionError: If both `sn_adata` and `sp_adata` are None.
        AssertionError: If provided `sn_adata` or `sp_adata` is not a valid `_AnnData` object.
        AssertionError: If the structure of `sn_adata` or `sp_adata` is invalid (e.g., mismatch
            between `X.shape` and index lengths).
        AssertionError: If `sn_colname_celltype` is not found in `sn_adata.obs`.
        AssertionError: If `sp_obsmname_xycoords` is not found in `sp_adata.obsm_keys()`.

    Attributes:
        sn_adata (_AnnData): Processed `sn_adata` with only overlapping genes retained.
        sp_adata (_AnnData): Copied and preprocessed `sp_adata`.
        downsampled_adata (_AnnData | _UndefinedType): Placeholder for downsampled data.
    """

    def __repr__(self) -> str:
        return f"""--- AnnDataPreparer (pytacs) ---
- sn_adata: {self.sn_adata}
- sp_adata: {self.sp_adata}
- downsampled_adata: {self.downsampled_adata}
--- --- --- --- ---
"""

    def __init__(
        self,
        sn_adata: _AnnData | None = None,
        sp_adata: _AnnData | None = None,
        sn_colname_celltype: str = "cell_type",
        sp_obsmname_xycoords: str = "spatial",
        overlapped_genes_warning: int = 10,
    ):
        """
        Initializes an object for processing snRNA-seq and spatial transcriptomic data
        for PyTACS. This method ensures that the input AnnData objects meet required
        conditions and applies transformations as needed.

        Upon initialization:
        - Copies of `sn_adata` and `sp_adata` are stored in `self.sn_adata` and `self.sp_adata`.
        - `sn_adata` is filtered to retain only genes that overlap with `sp_adata`, as these
        are used to train local classifiers.
        - If `sn_colname_celltype` differs from `"cell_type"`, it is renamed accordingly.
        - If `sp_obsmname_xycoords` differs from `"spatial"`, it is renamed accordingly.
        - If the number of overlapping genes is below `overlapped_genes_warning`, a warning
        message is printed.

        Args:
            sn_adata (_AnnData, optional): Single-nucleus or single-cell RNA-seq AnnData object
                (Scanpy object). Used to train local classifiers. Defaults to None.
            sp_adata (_AnnData, optional): Spatial transcriptomic AnnData object (Scanpy object).
                Defaults to None.
            sn_colname_celltype (str, optional): Column name in `sn_adata.obs` that contains cell-type
                annotations. If not `"cell_type"`, it will be renamed. Defaults to `"cell_type"`.
            sp_obsmname_xycoords (str, optional): Key in `sp_adata.obsm` that stores spatial coordinates.
                If not `"spatial"`, it will be renamed. Defaults to `"spatial"`.
            overlapped_genes_warning (int, optional): Threshold below which a warning is printed
                if the number of overlapping genes between `sn_adata` and `sp_adata` is too low.
                Defaults to 10.

        Raises:
            AssertionError: If both `sn_adata` and `sp_adata` are None.
            AssertionError: If provided `sn_adata` or `sp_adata` is not a valid `_AnnData` object.
            AssertionError: If the structure of `sn_adata` or `sp_adata` is invalid (e.g., mismatch
                between `X.shape` and index lengths).
            AssertionError: If `sn_colname_celltype` is not found in `sn_adata.obs`.
            AssertionError: If `sp_obsmname_xycoords` is not found in `sp_adata.obsm_keys()`.

        Attributes:
            sn_adata (_AnnData): Processed `sn_adata` with only overlapping genes retained.
            sp_adata (_AnnData): Copied and preprocessed `sp_adata`.
            downsampled_adata (_AnnData | _UndefinedType): Placeholder for downsampled data.
        """
        # Checklist
        assert not (sn_adata is None and sp_adata is None)
        if sn_adata is not None:
            assert isinstance(sn_adata, _AnnData)
        if sp_adata is not None:
            assert isinstance(sp_adata, _AnnData)
        sn_adata_copy: _AnnData | _UndefinedType = _UNDEFINED
        sp_adata_copy: _AnnData | _UndefinedType = _UNDEFINED
        if sn_adata is not None:
            sn_adata_copy = sn_adata.copy()
        if sp_adata is not None:
            sp_adata_copy = sp_adata.copy()

        for adata_i in (sn_adata_copy, sp_adata_copy):
            if not isinstance(adata_i, _AnnData):
                continue
            assert adata_i.X.shape[0] == len(adata_i.obs.index)
            assert adata_i.X.shape[1] == len(adata_i.var.index)
            _reinit_index(adata_i)
            if not isinstance(adata_i.X, _csr_matrix):
                adata_i.X = _csr_matrix(adata_i.X)
        if isinstance(sn_adata_copy, _AnnData):
            assert sn_colname_celltype in sn_adata.obs.columns
            if sn_colname_celltype != "cell_type":
                sn_adata_copy.obs["cell_type"] = sn_adata_copy.obs[
                    sn_colname_celltype
                ].copy()
                del sn_adata_copy.obs[sn_colname_celltype]
                if (sn_colname_celltype + "_colors") in sn_adata_copy.uns.keys():
                    sn_adata_copy.uns["cell_type_colors"] = sn_adata_copy.uns[
                        sn_colname_celltype + "_colors"
                    ].copy()

        if isinstance(sp_adata_copy, _AnnData):
            assert sp_obsmname_xycoords in sp_adata_copy.obsm_keys()

            if sp_obsmname_xycoords != "spatial":
                sp_adata_copy.obsm["spatial"] = sp_adata_copy.obsm[sp_obsmname_xycoords]
                del sp_adata_copy.obsm[sp_obsmname_xycoords]

        if isinstance(sn_adata, _AnnData) and isinstance(sp_adata, _AnnData):
            overlapped_genes: list[str] = list(
                set(sp_adata_copy.var.index) & set(sn_adata_copy.var.index)
            )
            if len(overlapped_genes) <= overlapped_genes_warning:
                print(f"Warning: Overlapped genes of two datasets too few!")

        if isinstance(sn_adata, _AnnData) and isinstance(sp_adata, _AnnData):
            self.sn_adata: _AnnData = sn_adata_copy[:, overlapped_genes].copy()
            # only keeps overlapped genes for sn_adata
            # sp_adata remains untouched
        else:
            self.sn_adata: _AnnData = sn_adata_copy
        self.sp_adata: _AnnData = sp_adata_copy
        self.downsampled_adata: _AnnData | _UndefinedType = _UNDEFINED
        return

    def filter_genes_highly_variable(
        self,
        min_counts: int = 3,
        n_top_genes: int = 3000,
    ) -> None:
        """
        Filters `sn_adata` to retain only highly variable genes.

        This method:
        1. Stores raw counts in `layers["counts"]`.
        2. Filters genes based on `min_counts`.
        3. Normalizes total counts to 10,000 per cell.
        4. Applies log1p transformation and stores in `layers["log1p"]`.
        5. Identifies and retains only the top `n_top_genes` most variable genes.
        6. Restores raw counts back to `.X`.

        Args:
            min_counts (int, optional): Minimum counts required to retain a gene. Defaults to 3.
            n_top_genes (int, optional): Number of highly variable genes to keep. Defaults to 3000.

        Raises:
            AssertionError: If `sn_adata` is not an instance of `_AnnData`.

        Notes:
            - If you need to apply this to a modified version of `sn_adata`
            (e.g., `downsampled_adata`), you should first back up the object and
            assign the modified `sn_adata` before calling this method.
            - For advanced preprocessing, use `scanpy` directly.
        """
        assert isinstance(
            self.sn_adata, _AnnData
        ), "sn_adata must be an AnnData object."

        print("Backing up raw counts...")
        self.sn_adata.layers["counts"] = self.sn_adata.X.copy()

        print(f"Filtering genes with min_counts >= {min_counts}...")
        _sc.pp.filter_genes(self.sn_adata, min_counts=min_counts, inplace=True)

        print("Normalizing total counts to 10,000 per cell...")
        _sc.pp.normalize_total(self.sn_adata, target_sum=1e4, inplace=True)

        print("Applying log1p transformation...")
        _sc.pp.log1p(self.sn_adata)
        self.sn_adata.layers["log1p"] = self.sn_adata.X.copy()

        print(f"Selecting top {n_top_genes} highly variable genes...")
        _sc.pp.highly_variable_genes(
            self.sn_adata, n_top_genes=n_top_genes, subset=True
        )

        print("Restoring raw counts...")
        self.sn_adata.X = self.sn_adata.layers["counts"].copy()

        print("Highly variable gene filtering completed.")
        return

    def sample_signatures_logl1_local_maxima(
        self,
        radius_downsampling: float = 1.5,  # 8 neighbors
        threshold_adjacent: float = 1.2,  # 4 neighbors
        n_samples: int = 2000,
        n_clusters: int = 9,
        colname_cluster: str = "cluster",
    ) -> None:
        """
        Samples from spatial anndata to generate a reference sn-RNA data.

        First, it performs downsampling:
        A log-L1-based filter strategy is adopted to only select those locally highly
        expressed spots as an approximation of single cells.

        `logl1 = sum(log1p(counts_of_genes))`.

        Second, it performs binning to further alleviate sparsity issue:
        Each time, a spot is taken as the centroid, and it aggregates
        surrounding spots to form a corpus to approximate single-cell level,
        just like the traditional way. Set `threshold_adjacent` to 0. to skip performing binning.

        Finally, it performs clustering to get several
        clusters as reference signatures to train downstream local classifiers.

        Update self.downsampled_adata
            .obs:
                ['cluster']: 'cluster 1', 'cluster 2', ...
                ['logl1']: 3.68, 4.45, ...  # original logl1 of target, not binned
                ['id_target']: '1324', '2433', ...
                ...
        Args:
            radius_downsampling (float): Radius of a window for downsampling. The spot with
            the highest logl1 within the window is selected.

            threshold_adjacent (float): Used in binning phase. Spots within this range
            are considered neighbors. Set to 0 to skip binning post-process.

            n_samples (int): Number of samples to generate.

            n_clusters (int): Number of clusters to generate. Set it a little larger than expected
            for novel cell type exploration.
        """
        assert isinstance(self.sp_adata, _AnnData)
        # Downsampling
        # Compute logl1 scores for each spot
        if isinstance(self.sp_adata.X, _np.ndarray):
            print(f"Warning: sp_adata.X is dense type. Converting to csr_matrix type.")
            self.sp_adata.X = _csr_matrix(self.sp_adata.X)
        assert isinstance(self.sp_adata.X, _csr_matrix)
        self.sp_adata.layers["log1p"] = self.sp_adata.X.copy()
        self.sp_adata.layers["log1p"].data = _np.log1p(self.sp_adata.X.data)
        logl1 = _np.array(self.sp_adata.layers["log1p"].sum(axis=1).tolist()).reshape(
            -1
        )
        self.sp_adata.obs["logl1"] = logl1

        # Create spatial distance matrix
        ckdtree_points = _cKDTree(self.sp_adata.obs[["x", "y"]].values)
        dist_matrix: _dok_matrix = ckdtree_points.sparse_distance_matrix(
            other=ckdtree_points,
            max_distance=max(radius_downsampling, threshold_adjacent) + 1e-8,
            p=2,
            output_type="dok_matrix",
        )
        # Prepare output anndata
        out_matrix = _dok_matrix(
            (min(self.sp_adata.shape[0], n_samples), self.sp_adata.shape[1]), int
        )
        out_ids = list()
        out_ids_target = list()
        out_logl1 = list()
        # temporarily use dok_matrix for fast re-assignment
        # Make sure indices are integerized
        indices_pool: _NDArray[_np.int_] = self.sp_adata.obs.index.values.astype(int)
        assert _np.all(indices_pool == _np.arange(self.sp_adata.shape[0]))

        for i_sampling in _tqdm(
            range(min(self.sp_adata.shape[0], n_samples)),
            desc="Sampling",
            ncols=60,
        ):
            assert len(indices_pool) > 0  # DEBUG: This should never raise
            iloc_sampled: int = _np.random.choice(range(len(indices_pool)))
            id_sampled: int = indices_pool[iloc_sampled]
            indices_pool = _np.delete(indices_pool, iloc_sampled)
            out_ids.append(id_sampled)
            dist_array = _to_array(dist_matrix[id_sampled, :], squeeze=True)
            where_local: _NDArray[_np.bool_] = (dist_array > 0.0) * (
                dist_array <= radius_downsampling
            )
            where_local[id_sampled] = True
            local_logl1: _pd.Series = self.sp_adata.obs.loc[where_local, "logl1"]
            iloc_target: int = _np.argmax(local_logl1)
            id_target: int = local_logl1.index[iloc_target]
            out_ids_target.append(id_target)
            logl1_target: float = local_logl1.loc[str(id_target)]
            out_logl1.append(logl1_target)

            # Post-process: binning (threshold_adjacent=0 to skip this step)
            # Find binning neighbors
            where_binning: _NDArray[_np.bool_] = (dist_array > 0.0) * (
                dist_array <= threshold_adjacent
            )
            where_binning[id_sampled] = True
            expr_vector: _NDArray = _np.array(
                self.sp_adata.X[where_binning, :].sum(axis=0).tolist()
            ).reshape(-1)
            out_matrix[i_sampling, :] = expr_vector

        self.downsampled_adata = _AnnData(
            X=_csr_matrix(out_matrix),
            obs=_pd.DataFrame(
                {
                    "logl1": out_logl1,
                    "id_target": out_ids_target,
                },
                index=_np.arange(out_matrix.shape[0]).astype(str),
            ),
            var=_pd.DataFrame(
                index=self.sp_adata.var.index,
            ),
            uns={
                "method": "logl1_local_maxima",
                "threshold_adjacent": threshold_adjacent,
                "radius_downsampling": radius_downsampling,
            },
        )

        # Cluster
        _tqdm.write("Clustering ...")
        self.downsampled_adata.layers["counts"] = self.downsampled_adata.X.copy()
        _sc.pp.normalize_total(self.downsampled_adata, target_sum=1e4)
        _sc.pp.log1p(self.downsampled_adata)
        _sc.pp.pca(self.downsampled_adata)
        X_pca = self.downsampled_adata.obsm["X_pca"]
        Z = _linkage(X_pca, method="ward")
        clusters_labels = _fcluster(Z, t=n_clusters, criterion="maxclust")
        self.downsampled_adata.obs[colname_cluster] = [
            f"Cluster {i}" for i in clusters_labels
        ]
        self.downsampled_adata.layers["log1p"] = self.downsampled_adata.X.copy()
        self.downsampled_adata.X = self.downsampled_adata.layers["counts"]
        _tqdm.write("Done.")
        return

    def sample_signatures_simple_bin(
        self,
        threshold_adjacent: float = 1.5,  # 8 neighbors
        n_samples: int | None = None,
        n_clusters: int = 9,
        colname_cluster: str = "cluster",
    ) -> None:
        """
        Samples from spatial anndata to generate a reference sn-RNA data.

        It performs binning on each spot to alleviate sparsity issue:
        Each time, a spot is taken as the centroid, and it aggregates
        surrounding spots to form a corpus to approximate single-cell level,
        just like the traditional way. Set `threshold_adjacent` to 0. to avoid binning.

        Finally, it performs clustering to get several
        clusters as reference signatures to train downstream local classifiers.

        Update self.downsampled_adata.
            .obs:
                ['cluster']: 'cluster 1', 'cluster 2', ...
                ...
        Args:
            threshold_adjacent (float): Used in binning phase. Spots within this range
            are considered neighbors. Set to 0 to skip binning post-process.

            n_samples (int): Number of samples to generate.

            n_clusters (int): Number of clusters to generate. Set it a little larger than expected
            for novel cell type exploration.
        """
        assert isinstance(self.sp_adata, _AnnData)
        # Downsampling
        # Compute logl1 scores for each spot
        if isinstance(self.sp_adata.X, _np.ndarray):
            print(f"Warning: sp_adata.X is dense type. Converting to csr_matrix type.")
            self.sp_adata.X = _csr_matrix(self.sp_adata.X)
        assert isinstance(self.sp_adata.X, _csr_matrix)

        if n_samples is None:
            n_samples: int = self.sp_adata.shape[0]

        # Create spatial distance matrix
        ckdtree_points = _cKDTree(self.sp_adata.obs[["x", "y"]].values)
        dist_matrix: _dok_matrix = ckdtree_points.sparse_distance_matrix(
            other=ckdtree_points,
            max_distance=threshold_adjacent + 1e-8,
            p=2,
            output_type="dok_matrix",
        )
        # Prepare output anndata
        out_matrix = _dok_matrix(
            (min(self.sp_adata.shape[0], n_samples), self.sp_adata.shape[1]), int
        )
        out_ids = list()
        out_ids_target = list()

        # temporarily use dok_matrix for fast re-assignment
        # Make sure indices are integerized
        indices_pool: _NDArray[_np.int_] = self.sp_adata.obs.index.values.astype(int)
        assert _np.all(indices_pool == _np.arange(self.sp_adata.shape[0]))

        for i_sampling in _tqdm(
            range(min(self.sp_adata.shape[0], n_samples)),
            desc="Sampling",
            ncols=60,
        ):
            assert len(indices_pool) > 0  # DEBUG: This should never raise
            iloc_sampled: int = _np.random.choice(range(len(indices_pool)))
            id_sampled: int = indices_pool[iloc_sampled]
            indices_pool = _np.delete(indices_pool, iloc_sampled)
            out_ids.append(id_sampled)
            dist_array = _to_array(dist_matrix[id_sampled, :], squeeze=True)
            where_local: _NDArray[_np.bool_] = (dist_array > 0.0) * (
                dist_array <= threshold_adjacent
            )
            where_local[id_sampled] = True
            id_target: int = id_sampled
            out_ids_target.append(id_target)

            # Post-process: binning (threshold_adjacent=0 to skip this step)
            # Find binning neighbors
            where_binning: _NDArray[_np.bool_] = where_local
            expr_vector: _NDArray = _np.array(
                self.sp_adata.X[where_binning, :].sum(axis=0).tolist()
            ).reshape(-1)
            out_matrix[i_sampling, :] = expr_vector

        self.downsampled_adata = _AnnData(
            X=_csr_matrix(out_matrix),
            obs=_pd.DataFrame(
                {
                    "id_target": out_ids_target,
                },
                index=_np.arange(out_matrix.shape[0]).astype(str),
            ),
            var=_pd.DataFrame(
                index=self.sp_adata.var.index,
            ),
            uns={
                "method": "simple_bin",
                "threshold_adjacent": threshold_adjacent,
            },
        )

        # Cluster
        _tqdm.write("Clustering ...")
        self.downsampled_adata.layers["counts"] = self.downsampled_adata.X.copy()
        _sc.pp.normalize_total(self.downsampled_adata, target_sum=1e4)
        _sc.pp.log1p(self.downsampled_adata)
        _sc.pp.pca(self.downsampled_adata)
        X_pca = self.downsampled_adata.obsm["X_pca"]
        Z = _linkage(X_pca, method="ward")
        clusters_labels = _fcluster(Z, t=n_clusters, criterion="maxclust")
        self.downsampled_adata.obs[colname_cluster] = [
            f"Cluster {i}" for i in clusters_labels
        ]
        self.downsampled_adata.layers["log1p"] = self.downsampled_adata.X.copy()
        self.downsampled_adata.X = self.downsampled_adata.layers["counts"]
        _tqdm.write("Done.")
        return

    def match_signatures_to_types(
        self,
        colname_cluster_downsampled: str = "cluster",
        colname_type_sn_adata: str = "cell_type",
        new_colname_match: str = "cell_type",
        sep_for_multiple_types: str = "+",
        new_name_novel: str = "novel",
        method: _Literal["SVM", "cosine", "jaccard"] = "cosine",
        n_top_genes_truncate_jaccard: int | None = None,
        pretrained_svm_clf: _SVM | None = None,
    ) -> tuple[_pd.DataFrame, _pd.DataFrame]:
        """updates .sn_adata_downsampledFromSpAdata.

        Returns a tuple of two dataframes:
            DataFrame of Type-by-Cluster similarities, and
            DataFrame of Type-by-Cluster matchedness.

        When clusters are too few, jaccard might fail."""
        assert method in ["SVM", "cosine", "jaccard"]
        assert isinstance(self.downsampled_adata, _AnnData)
        assert isinstance(self.sn_adata, _AnnData)
        assert colname_type_sn_adata in self.sn_adata.obs.columns
        assert colname_cluster_downsampled in self.downsampled_adata.obs.columns
        if new_colname_match in self.downsampled_adata.obs.columns:
            _tqdm.write(
                f"Warning: {new_colname_match} already in sn_adata_downsampledFromSpAdata.obs!"
            )
        overlapped_genes = _np.array(
            list(
                set(self.sn_adata.var.index.values)
                & set(self.downsampled_adata.var.index.values)
            )
        )
        if len(overlapped_genes) < 100:
            _tqdm.write(f"Warning: overlapped genes < 100 might be too few!")

        sn_adata = self.sn_adata[:, overlapped_genes].copy()
        sn_adata_downsampled = self.downsampled_adata[:, overlapped_genes].copy()
        # Compute type-wise mean
        celltypes = _np.unique(sn_adata.obs[colname_type_sn_adata])
        cellclusters = _np.unique(sn_adata_downsampled.obs[colname_cluster_downsampled])
        celltype_signatures = dict()
        cellclusters_signatures = dict()

        for ct in _tqdm(celltypes, desc="Compute type signatures", ncols=60):
            where_thistype = (sn_adata.obs[colname_type_sn_adata] == ct).values
            expr_vector = _np.array(
                sn_adata.X[where_thistype, :].mean(axis=0).tolist()
            ).reshape(-1)
            expr_vector /= max(_np.sum(expr_vector), 1e-8)
            # expr_vector = _np.log1p(expr_vector)
            celltype_signatures[ct] = expr_vector

        for clt in _tqdm(cellclusters, desc="Compute cluster signatures", ncols=60):
            where_thistype = (
                sn_adata_downsampled.obs[colname_cluster_downsampled] == clt
            ).values
            expr_vector = _np.array(
                sn_adata_downsampled.X[where_thistype, :].mean(axis=0).tolist()
            ).reshape(-1)
            expr_vector /= max(_np.sum(expr_vector), 1e-8)
            # expr_vector = _np.log1p(expr_vector)
            cellclusters_signatures[clt] = expr_vector

        df_match = _pd.DataFrame(
            index=celltypes,
            columns=cellclusters,
            dtype=float,
        )

        if method == "SVM":
            if pretrained_svm_clf is None:
                _tqdm.write("Pretrained SVM absent. Train from start.")
                svc = _SVM(
                    threshold_confidence=0.0,
                    log1p=True,
                    normalize=True,
                    on_PCs=False,
                )

                _tqdm.write("Training SVM ...")
                svc.fit(sn_adata=sn_adata, colname_classes=colname_type_sn_adata)
            else:
                svc = pretrained_svm_clf

        for clt in _tqdm(cellclusters, desc="Compute mutual similarity", ncols=60):
            if method == "SVM":
                probas_ct = svc.predict_proba(
                    X=_np.array([cellclusters_signatures[clt]]),
                    genes=sn_adata_downsampled.var.index.values,
                )[0]
                df_match.loc[:, clt] = probas_ct
                continue
            for ct in celltypes:
                if method == "jaccard":
                    ctsig = celltype_signatures[ct]
                    ccsig = cellclusters_signatures[clt]
                    if isinstance(n_top_genes_truncate_jaccard, int):
                        ctsig = _truncate_top_n(
                            ctsig, n_top=n_top_genes_truncate_jaccard
                        )
                        ccsig = _truncate_top_n(
                            ccsig, n_top=n_top_genes_truncate_jaccard
                        )
                    df_match.loc[ct, clt] = (
                        _np.bool_(ctsig) == _np.bool_(ccsig)
                    ).mean()
                elif method == "cosine":
                    ctsig = _np.log1p(celltype_signatures[ct])
                    ccsig = _np.log1p(cellclusters_signatures[clt])
                    df_match.loc[ct, clt] = (ctsig @ ccsig) / max(
                        1e-8,
                        _np.linalg.norm(ctsig) * _np.linalg.norm(ccsig),
                    )
        iloc_matched_clusters = _np.argmax(df_match.values, axis=1)
        df_match_bool = df_match.copy().astype(bool)
        df_match_bool.loc[:, :] = False
        for iloc_type, iloc_clt in enumerate(iloc_matched_clusters):
            df_match_bool.iloc[iloc_type, iloc_clt] = True

        self.downsampled_adata.obs[new_colname_match] = ""
        new_annotations_mapping = dict()
        for clt in cellclusters:
            ct_matched: list[str] = list(
                df_match_bool.index.values[df_match_bool.loc[:, clt].values].astype(str)
            )
            if len(ct_matched) == 0:
                new_annotations_mapping[clt] = new_name_novel
            else:
                new_annotations_mapping[clt] = sep_for_multiple_types.join(ct_matched)
        new_annotations = self.downsampled_adata.obs[
            colname_cluster_downsampled
        ].values.copy()
        for clt in _tqdm(cellclusters, desc="Record in .obs", ncols=60):
            new_annotations[
                self.downsampled_adata.obs[colname_cluster_downsampled].values == clt
            ] = new_annotations_mapping[clt]
        self.downsampled_adata.obs[new_colname_match] = new_annotations
        _tqdm.write(f'Annotations written in .obs["{new_colname_match}"].')
        return (df_match, df_match_bool)


# Some utililties


def downsample_cells(
    adata: _AnnData,
    n_samples: int = 10000,
    colname_celltype: str = "cell_type",
    reindex: bool = True,
) -> tuple[_AnnData, _Counter]:
    """Downsample a fraction of cells from adata.

    Return Tuple of (result_adata, Counter of downsampled cell-types)"""
    ids_sampled = _np.random.choice(
        _np.arange(adata.shape[0]), size=n_samples, replace=False
    )
    result = adata[ids_sampled, :].copy()
    if reindex:
        result.obs.index = _np.arange(result.shape[0]).astype(str)
    counter = _Counter(result.obs[colname_celltype].values)
    return (result, counter)


def compare_umap(
    adata1: _AnnData,
    adata2: _AnnData | None = None,
    col_color: str | None = None,
) -> list[_Axes]:
    """
    Compare the umaps of two AnnData with a
    common observed column, e.g., "cell_type" or "cluster".
    in one unified UMAP embedding space.

    Return:
        axes[0]: joined data with batch colored;
        Below are for `col_color is not None`:
        axes[1]: joined data with col_color (e.g. cluster) colored;
        axes[2]: adata1 with col_color colored;
        axes[3]: adata2 with col_color colored;
    """
    if col_color is not None:
        assert col_color in adata1.obs.columns
        assert col_color in adata2.obs.columns
    if adata2 is None:
        adata_joined: _AnnData = adata1.copy()
        adata_joined.obs["batch"] = "1"
    else:
        adata_joined: _AnnData = _sc.anndata.concat(
            adatas=[adata1, adata2],
            axis="obs",
            join="inner",
            label="batch",
            keys=["1", "2"],
            index_unique="_",
        )

    _sc.pp.normalize_total(adata_joined)
    _sc.pp.log1p(adata_joined)
    _sc.pp.pca(adata_joined)
    _sc.pp.neighbors(adata_joined)
    _sc.tl.umap(adata_joined)
    axes: list[_Axes] = []
    axes.append(
        _sc.pl.umap(
            adata_joined,
            color="batch",
            show=False,
        )
    )
    xlim_universal = axes[0].get_xlim()
    ylim_universal = axes[0].get_ylim()
    if col_color is None:
        return axes
    axes.append(
        _sc.pl.umap(
            adata_joined,
            color=col_color,
            show=False,
        )
    )
    axes[1].set_title("Joined Data")
    axes.append(
        _sc.pl.umap(
            adata_joined[(adata_joined.obs["batch"] == "1").values, :],
            color=col_color,
            show=False,
        )
    )
    axes[2].set_title("Anndata1")
    axes.append(
        _sc.pl.umap(
            adata_joined[(adata_joined.obs["batch"] == "2").values, :],
            color=col_color,
            show=False,
        )
    )
    axes[3].set_title("Anndata2")
    for ax in axes[1:]:
        ax.set_xlim(xlim_universal)
        ax.set_ylim(ylim_universal)
    return axes


def binX(
    adata: _sc.AnnData,
    obsm_name_spatial_coords: str = 'spatial',
    key_added: str | None = None,
    binsize: int = 48,
) -> _sc.AnnData:
    """
    Bin a spatial trx anndata.

    Args:
        adata (AnnData): spatial trx.

        obsm_name_spatial_coords (str): name of key in obsm of spatial coordinates.

        key_added (str | None): an auxiliary obs column will be added to adata, with name
        defaulting to 'spatial_binX' (when this parameter set to None) where X is the binsize.

        binsize (int): size of bin.
    
    Return:
        AnnData: binned anndata with spatial coordinates saved in obsm with the same name as adata.
    """
    assert isinstance(binsize, int)
    if key_added is None:
        key_added = f'spatial_bin{binsize}'
    adata.obsm[key_added] = adata.obsm[obsm_name_spatial_coords] // binsize
    # Build index
    index_loc2id = dict()
    for i in _tqdm(range(adata.shape[0]),desc='Building index'):
        loc = tuple(adata.obsm[key_added][i,:])
        if loc not in index_loc2id:
            index_loc2id[loc] = [i]
        else:
            index_loc2id[loc].append(i)
    n_new = len(index_loc2id)
    cols_ = []
    rows_ = []
    for i_new, loc in enumerate(_tqdm(index_loc2id, desc='Binning')):
        cols_.extend(index_loc2id[loc])
        rows_.extend([i_new for _ in range(len(index_loc2id[loc]))])
        
    W_leftmul = _csr_matrix(
        (
            _np.ones(adata.shape[0]),
            (
                rows_,
                cols_,
            ),
        ),
        shape=(n_new, adata.shape[0]),
    )
    X_new = (W_leftmul @ adata.X).astype(adata.X.dtype)
    coords_new = _np.array(tuple(index_loc2id.keys()))
    return _sc.AnnData(
        X=X_new,
        var=_pd.DataFrame(index=adata.var.index),
        obsm={obsm_name_spatial_coords: coords_new},
    )

def annotate_mt(
    adata: _AnnData,
    startswith: str | _Iterable[str] = "MT-",
    key_added: str = 'mt',
) -> None:
    """
    Annotate mitochondrial genes in an AnnData object.

    Args:
        adata (_AnnData): The AnnData object to process.
        startswith (str): The prefix of mitochondrial genes to remove. Defaults to "MT-".
        key_added (str): The key under which to store the annotation in `adata.var`.
            Defaults to 'mt'.
    """
    assert isinstance(adata, _AnnData)
    if isinstance(startswith, str):
        startswith = [startswith]
    startswith = tuple(startswith)
    mt_genes = adata.var.index.str.startswith(startswith)
    adata.var[key_added] = mt_genes
    if mt_genes.sum() == 0:
        _tqdm.write(f"Warning: No mitochondrial genes found starting with '{startswith}'!")
    else:
        _tqdm.write(f"Annotated {mt_genes.sum()} mitochondrial genes starting with '{startswith}' in adata.var['{key_added}'].")
    return

def annotate_ribosomal(
    adata: _AnnData,
    startswith: str | _Iterable[str] = ("RPL", "RPS"),
    key_added: str = 'ribosomal',
) -> None:
    """
    Annotate ribosomal genes in an AnnData object.

    Args:
        adata (_AnnData): The AnnData object to process.
        startswith (str | Iterable[str]): The prefix of ribosomal genes to remove.
            Defaults to ("RPL", "RPS").
        key_added (str): The key under which to store the annotation in `adata.var`.
            Defaults to 'ribosomal'.
    """
    assert isinstance(adata, _AnnData)
    if isinstance(startswith, str):
        startswith = [startswith]
    ribosomal_genes = adata.var.index.str.startswith(tuple(startswith))
    adata.var[key_added] = ribosomal_genes
    if ribosomal_genes.sum() == 0:
        _tqdm.write(f"Warning: No ribosomal genes found starting with {startswith}!")
    else:
        _tqdm.write(f"Annotated {ribosomal_genes.sum()} ribosomal genes starting with {startswith} in adata.var['{key_added}'].")
    return

def merge_gene_version(
    adata: _AnnData,
    version_sep: str = ".",
) -> _AnnData:
    """
    Merge gene versions in an AnnData object. Keep the maximum counts among versions of gene.
    Note that .layers and .var will be empty.

    Args:
        adata (_AnnData): The AnnData object to process.
        version_sep (str): The separator used to split gene names and versions. Defaults to ".".

    Returns:
        _AnnData: A new AnnData object with merged gene versions.
    """
    
    var_names = adata.var_names
    base_names = var_names.str.split(version_sep).str[0]

    # Build indices of the same base gene
    gene_to_indices = _defaultdict(list)
    for idx, name in enumerate(base_names):
        gene_to_indices[name].append(idx)
    
    # Prepare for building new matrix
    merged_columns = []
    new_var_names = []

    _tqdm.write('Converting to column-efficient mode..')
    X_old = adata.X.copy().tocsc()

    for base_name, idxs in _tqdm(gene_to_indices.items()):
        if len(idxs) == 1:
            col = X_old[:, idxs[0]]
        else:
            cols = X_old[:, idxs].tocsc()
            col = cols.max(axis=1)
        merged_columns.append(col)
        new_var_names.append(base_name)
    
    # Stack
    X_new = _csr_matrix(_hstack(merged_columns, format='csr'))
    
    adata_merged = _AnnData(
        X=X_new,
        obs=adata.obs.copy(),
        var=_pd.DataFrame(index=new_var_names),
        obsm=adata.obsm.copy(),
        obsp=adata.obsp.copy(),
        uns=adata.uns.copy(),
    )

    return adata_merged


def scale_genes(
    adata: _AnnData,
) -> None:
    """Scale each gene in adata.X so that the maximum value per gene is 1.

    Modifies adata.X in place.

    Args:
        adata (_AnnData): AnnData object with expression matrix in adata.X
    """
    # First check if there's zero gene
    X = adata.X.copy().tocsc()

    max_per_gene = X.max(axis=0).toarray()
    whr_zero = (max_per_gene==0)
    if whr_zero.sum() > 0:
        _tqdm.write('Warning: exist genes with 0 counts!')
        max_per_gene[whr_zero] = 1
    X = _csr_matrix(
        X.multiply(
            1.0 / max_per_gene
        )
    )
    adata.X = X
    return


def sort_by_coords(
    anndata: _AnnData,
    major_axis: _Literal['x', 'y', 0, 1] = 'x',
) -> _AnnData:
    """
    major_axis: if 'x' or 0, sort by x first; if 'y' or 1, sort by y first.

    NOTE:
        This function DOES NOT re-init obs indices after operation.
    """
    assert 'spatial' in anndata.obsm_keys()
    assert major_axis in ['x', 'y', 0, 1]
    keys = [
        anndata.obsm['spatial'][:,1],
        anndata.obsm['spatial'][:,0],
    ]
    if major_axis in ['y', 1]:
        keys = [keys[1], keys[0]]
    
    ix_sort = _np.lexsort(keys=keys)
    return anndata[ix_sort].copy()
