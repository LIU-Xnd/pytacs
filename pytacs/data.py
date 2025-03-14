from scanpy import AnnData as _AnnData

import scanpy as _sc
import pandas as _pd
import numpy as _np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.sparse import dok_matrix as _dok_matrix
from scipy.spatial import cKDTree as _cKDTree

from scipy.cluster.hierarchy import linkage as _linkage
from scipy.cluster.hierarchy import fcluster as _fcluster
from typing import Literal

from tqdm import tqdm
from .utils import save_and_tidy_index as _save_and_tidy_index
from .utils import _UNDEFINED, _UndefinedType
from .utils import to_array as _to_array


class AnnDataPreparer:
    """Prepare snRNA-seq and spatial transcriptomic data for pytacs.
    Will check data requirements when initialized.
    Will save properly transformed AnnData copies in self.sn_adata and
     self.sp_adata.
    The sn_adata is kept with only overlapped genes with sp_adata, which is
     used for training local classifiers.
    The sn_adata with simulated negative control samples is saved in
     self.sn_adata_withNegativeControl.

    Args:
        sn_adata (_AnnData): snRNA-seq (or scRNA-seq) AnnData (scanpy object)
        sp_adata (_AnnData): spatial transcriptomic AnnData (scanpy object)
    """

    def __repr__(self) -> str:
        return f"""--- AnnDataPreparer (pytacs) ---
- sn_adata: {self.sn_adata}
- sp_adata: {self.sp_adata}
- sn_adata_withNegativeControl: {self.sn_adata_withNegativeControl}
- sn_adata_downsampledFromSpaAdata: {self.sn_adata_downsampledFromSpAdata}
--- --- --- --- ---
"""

    def __init__(
        self,
        sn_adata: _AnnData | None = None,
        sp_adata: _AnnData | None = None,
        sn_colname_celltype: str = "cell_type",
        sp_colnames_x_and_y: tuple[str, str] = ("x", "y"),
        overlapped_genes_warning: int = 10,
    ):
        """Prepare snRNA-seq and spatial transcriptomic data for pytacs.
        Will check data requirements when initialized.
        Will save properly transformed AnnData copies in self.sn_adata and self.sp_adata.
        The sn_adata is kept with only overlapped genes with sp_adata, which is used
         for training local classifiers (but we recommend using
         self.sn_adata_withNegativeControl for training).
        The sn_adata with simulated negative control samples is saved in self.sn_adata_withNegativeControl.

        Args:
            sn_adata (_AnnData): snRNA-seq (or scRNA-seq) AnnData (scanpy object)
            sp_adata (_AnnData): spatial transcriptomic AnnData (scanpy object)
        """
        # Checklist
        assert isinstance(sn_adata, _AnnData) or isinstance(sp_adata, _AnnData)
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
            _save_and_tidy_index(adata_i)
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
            assert len(sp_colnames_x_and_y) == 2
            sp_names_standard = ("x", "y")
            for i_name, sp_name in enumerate(sp_colnames_x_and_y):
                assert sp_name in sp_adata.obs.columns
                if sp_name != sp_names_standard[i_name]:
                    sp_adata_copy.obs[sp_names_standard[i_name]] = sp_adata_copy.obs[
                        sp_name
                    ].copy()
                    del sp_adata_copy.obs[sp_name]

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
        self.sn_adata_withNegativeControl: _AnnData | _UndefinedType = _UNDEFINED
        self.sn_adata_downsampledFromSpAdata: _AnnData | _UndefinedType = _UNDEFINED
        return

    def simulate_negative_control(
        self,
        ratio_samplingFrom: float = 0.1,
        ratio_mask: float = 0.3,
        negative_samples_proportion: float = 0.5,  # (0 to 1, exclusive)
        update_self: bool = True,
    ) -> _AnnData:
        """Simulate negative control samples.

        Args:
            ratio_samplingFrom (float, optional): sampling from the least (this ratio * n_samples)
             expressed samples for each gene (0 to 1, supposedly exclusive). Defaults to 0.1.
            ratio_mask (float, optional): this much ratio of sampled expression values
             are randomly set to 0.
             Supposedly 0 to 1, exclusive. Defaults to 0.3.
            negative_samples_proportion (float, optional): to generate this much proportion of negative
             control samples (0 to 1, exclusive). Defaults to 0.5.
            update_self (bool, optional): whether to update self.sn_adata_withNegativeControl with this result.
             Defaults to True.

        Returns:
            _AnnData: AnnData with the old and newly generated (negative control) samples, whose
             .obs['cell_type'] are '__NegativeControl'.
        """
        # negative_samples_proportion = n_new / (n_old + n_new) = 1 - n_old / (n_old + n_new)
        #  -> n_new = n_old * ((1/(1-negative_samples_proportion)) - 1) =: n_old * folds_newToOld
        assert 0 < negative_samples_proportion < 1
        folds_newToOld = (1 / (1 - negative_samples_proportion)) - 1
        n_old = self.sn_adata.X.shape[0]
        n_new = int(_np.ceil(n_old * folds_newToOld))
        n_samplingFrom = int(_np.ceil(ratio_samplingFrom * n_old))
        X_old: _np.ndarray = _to_array(self.sn_adata.X)
        X_extra: _np.ndarray = _np.zeros(shape=(n_new, X_old.shape[1]), dtype=int)
        for i_gene in range(X_old.shape[1]):
            # Select the least n_samplingFrom expressed samples
            lowly_expressed = _np.sort(X_old[:, i_gene])[:n_samplingFrom].max()
            X_extra[:, i_gene] = (
                _np.random.random(size=(X_extra.shape[0],)) * lowly_expressed
            )

        icols_tozero = _np.random.randint(
            0,
            X_extra.shape[1],
            size=int(ratio_mask * X_extra.shape[0] * X_extra.shape[1]),
        )
        irows_tozero = _np.random.randint(
            0, X_extra.shape[0], size=icols_tozero.shape[0]
        )

        X_extra[irows_tozero, icols_tozero] = 0

        sn_adata_withNegativeControl = _AnnData(
            X=_csr_matrix(_np.concatenate([X_old, X_extra], axis=0)),
            obs=_pd.DataFrame(
                data=_pd.concat(
                    [
                        self.sn_adata.obs["cell_type"],
                        _pd.Series(
                            ["__NegativeControl" for _ in range(X_extra.shape[0])]
                        ),
                    ]
                ).values,
                index=_pd.Series(
                    _np.arange(X_old.shape[0] + X_extra.shape[0]), dtype=str
                ),
                columns=["cell_type"],
            ),
            var=self.sn_adata.var.copy(),
        )
        if update_self:
            self.sn_adata_withNegativeControl = sn_adata_withNegativeControl
        return sn_adata_withNegativeControl

    def filter_genes_highly_variable(
        self,
        min_counts: int = 3,
        n_top_genes: int = 3000,
    ) -> None:
        """Filter on genes of sn_adata, keeping only highly variable genes.

        If you want to apply it on modified sn_adata, e.g., sn_adata_downsampledFromSpAdata
        or sn_adata_withNegativeControl, you'd better backup current object and re-assign
        the corresponding modified sn_adata to this object's .sn_adata as a workaround.

        First back up raw counts into layers['counts'],
        then filter genes, normalize, logarithmize, and find highly variable genes.
        Finally saves this expression matrix into layers['log1p'] and activate back raw counts,
        and only keeps highly variable genes.

        For more preprocessing please use scanpy directly."""
        assert isinstance(self.sn_adata, _AnnData)
        self.sn_adata.layers["counts"] = self.sn_adata.X.copy()
        _sc.pp.filter_genes(
            data=self.sn_adata,
            min_counts=min_counts,
            inplace=True,
        )
        _sc.pp.normalize_total(
            adata=self.sn_adata,
            target_sum=1e4,
            inplace=True,
        )
        _sc.pp.log1p(self.sn_adata)
        _sc.pp.highly_variable_genes(
            adata=self.sn_adata,
            n_top_genes=n_top_genes,
            subset=True,
        )
        self.sn_adata.layers["log1p"] = self.sn_adata.X.copy()
        self.sn_adata.X = self.sn_adata.layers["counts"].copy()
        return

    def downsample_signatures(
        self,
        radius_downsampling: float = 1.5,  # 8 neighbors
        threshold_adjacent: float = 1.2,  # 4 neighbors
        n_samples: int = 2000,
        n_clusters: int = 9,
        colname_cluster: str = "cluster",
    ) -> None:
        """
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

        Update self.sn_adata_downsampledFromSpAdata.
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
        indices_pool: NDArray[_np.int_] = self.sp_adata.obs.index.values.astype(int)
        assert _np.all(indices_pool == _np.arange(self.sp_adata.shape[0]))

        for i_sampling in tqdm(
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
            where_local: NDArray[_np.bool_] = (dist_array > 0.0) * (
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
            where_binning: NDArray[_np.bool_] = (dist_array > 0.0) * (
                dist_array <= threshold_adjacent
            )
            where_binning[id_sampled] = True
            expr_vector: NDArray = _np.array(
                self.sp_adata.X[where_binning, :].sum(axis=0).tolist()
            ).reshape(-1)
            out_matrix[i_sampling, :] = expr_vector

        self.sn_adata_downsampledFromSpAdata = _AnnData(
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
                "threshold_adjacent": threshold_adjacent,
                "radius_downsampling": radius_downsampling,
            },
        )

        # Cluster
        tqdm.write("Clustering ...")
        self.sn_adata_downsampledFromSpAdata.layers["counts"] = (
            self.sn_adata_downsampledFromSpAdata.X.copy()
        )
        _sc.pp.normalize_total(self.sn_adata_downsampledFromSpAdata, target_sum=1e4)
        _sc.pp.log1p(self.sn_adata_downsampledFromSpAdata)
        _sc.pp.pca(self.sn_adata_downsampledFromSpAdata)
        X_pca = self.sn_adata_downsampledFromSpAdata.obsm["X_pca"]
        Z = _linkage(X_pca, method="ward")
        clusters_labels = _fcluster(Z, t=n_clusters, criterion="maxclust")
        self.sn_adata_downsampledFromSpAdata.obs[colname_cluster] = [
            f"Cluster {i}" for i in clusters_labels
        ]
        self.sn_adata_downsampledFromSpAdata.layers["log1p"] = (
            self.sn_adata_downsampledFromSpAdata.X.copy()
        )
        self.sn_adata_downsampledFromSpAdata.X = (
            self.sn_adata_downsampledFromSpAdata.layers["counts"]
        )
        tqdm.write("Done.")
        return

    def match_signatures_to_types(
        self,
        colname_cluster_downsampled: str = "cluster",
        colname_type_sn_adata: str = "cell_type",
        new_colname_match: str = "cell_type",
        sep_for_multiple_types: str = "+",
        new_name_novel: str = "novel",
        method: Literal["cosine", "jaccard"] = "cosine",
    ) -> _pd.DataFrame:
        """updates .sn_adata_downsampledFromSpAdata.

        Returns a tuple of two dataframes:
            DataFrame of Type-by-Cluster similarities, and
            DataFrame of Type-by-Cluster matchedness.

        When clusters are too few, jaccard might fail."""
        assert method in ["cosine", "jaccard"]
        assert isinstance(self.sn_adata_downsampledFromSpAdata, _AnnData)
        assert isinstance(self.sn_adata, _AnnData)
        assert colname_type_sn_adata in self.sn_adata.obs.columns
        assert (
            colname_cluster_downsampled
            in self.sn_adata_downsampledFromSpAdata.obs.columns
        )
        if new_colname_match in self.sn_adata_downsampledFromSpAdata.obs.columns:
            tqdm.write(
                f"Warning: {new_colname_match} already in sn_adata_downsampledFromSpAdata.obs!"
            )
        overlapped_genes = _np.array(
            list(
                set(self.sn_adata.var.index.values)
                & set(self.sn_adata_downsampledFromSpAdata.var.index.values)
            )
        )
        if len(overlapped_genes) < 100:
            tqdm.write(f"Warning: overlapped genes < 100 might be too few!")

        sn_adata = self.sn_adata[:, overlapped_genes].copy()
        sn_adata_downsampled = self.sn_adata_downsampledFromSpAdata[
            :, overlapped_genes
        ].copy()
        # Compute type-wise mean
        celltypes = _np.unique(sn_adata.obs[colname_type_sn_adata])
        cellclusters = _np.unique(sn_adata_downsampled.obs[colname_cluster_downsampled])
        celltype_signatures = dict()
        cellclusters_signatures = dict()

        for ct in tqdm(celltypes, desc="Compute type signatures", ncols=60):
            where_thistype = (sn_adata.obs[colname_type_sn_adata] == ct).values
            expr_vector = _np.array(
                sn_adata.X[where_thistype, :].mean(axis=0).tolist()
            ).reshape(-1)
            expr_vector /= max(_np.sum(expr_vector), 1e-8)
            expr_vector = _np.log1p(expr_vector)
            celltype_signatures[ct] = expr_vector

        for clt in tqdm(cellclusters, desc="Compute cluster signatures", ncols=60):
            where_thistype = (
                sn_adata_downsampled.obs[colname_cluster_downsampled] == clt
            ).values
            expr_vector = _np.array(
                sn_adata_downsampled.X[where_thistype, :].mean(axis=0).tolist()
            ).reshape(-1)
            expr_vector /= max(_np.sum(expr_vector), 1e-8)
            expr_vector = _np.log1p(expr_vector)
            cellclusters_signatures[clt] = expr_vector

        df_match = _pd.DataFrame(
            index=celltypes,
            columns=cellclusters,
            dtype=float,
        )

        for ct in tqdm(celltypes, desc="Compute mutual similarity", ncols=60):
            for clt in cellclusters:
                if method == "jaccard":
                    df_match.loc[ct, clt] = (
                        _np.bool_(celltype_signatures[ct])
                        == _np.bool_(cellclusters_signatures[clt])
                    ).mean()
                else:  # 'cosine'
                    df_match.loc[ct, clt] = (
                        celltype_signatures[ct] @ cellclusters_signatures[clt]
                    ) / max(
                        1e-8,
                        _np.linalg.norm(celltype_signatures[ct])
                        * _np.linalg.norm(cellclusters_signatures[clt]),
                    )
        iloc_matched_clusters = _np.argmax(df_match.values, axis=1)
        df_match_bool = df_match.copy().astype(bool)
        df_match_bool.loc[:, :] = False
        for iloc_type, iloc_clt in enumerate(iloc_matched_clusters):
            df_match_bool.iloc[iloc_type, iloc_clt] = True

        self.sn_adata_downsampledFromSpAdata.obs[new_colname_match] = ""
        new_annotations_mapping = dict()
        for clt in cellclusters:
            ct_matched: list[str] = list(
                df_match_bool.index.values[df_match_bool.loc[:, clt].values].astype(str)
            )
            if len(ct_matched) == 0:
                new_annotations_mapping[clt] = new_name_novel
            else:
                new_annotations_mapping[clt] = sep_for_multiple_types.join(ct_matched)
        new_annotations = self.sn_adata_downsampledFromSpAdata.obs[
            colname_cluster_downsampled
        ].values.copy()
        for clt in tqdm(cellclusters, desc="Record in .obs", ncols=60):
            new_annotations[
                self.sn_adata_downsampledFromSpAdata.obs[
                    colname_cluster_downsampled
                ].values
                == clt
            ] = new_annotations_mapping[clt]
        self.sn_adata_downsampledFromSpAdata.obs[new_colname_match] = new_annotations
        tqdm.write(f'Annotations written in .obs["{new_colname_match}"].')
        return (df_match, df_match_bool)
