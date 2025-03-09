from scanpy import AnnData as _AnnData

import scanpy as _sc
import pandas as _pd
import numpy as _np
from scipy.sparse import csr_matrix as _csr_matrix
from .utils import save_and_tidy_index as _save_and_tidy_index
from .utils import _UNDEFINED, _UndefinedType


class AnnDataPreparer:
    """Prepare snRNA-seq and spatial transcriptomic data for pytacs.
    Will check data requirements when initialized.
    Will save properly transformed AnnData copies in self.sn_adata and
     self.sp_adata.
    The sn_adata is kept with only overlapped genes with sp_adata, which is
     used for training local classifiers (but we recommend using
     self.sn_adata_withNegativeControl for training).
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
- sn_adata_withNegativeControl: {self.__sn_adata_withNegativeControl}
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
        self.__sn_adata_withNegativeControl: _AnnData | _UndefinedType = _UNDEFINED
        return

    @property
    def sn_adata_withNegativeControl(self) -> _AnnData:
        return self.__sn_adata_withNegativeControl

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
        X_old: _np.ndarray = self.sn_adata.X.toarray()
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
            self.__sn_adata_withNegativeControl = sn_adata_withNegativeControl
        return sn_adata_withNegativeControl

    def filter_genes_highly_variable(
        self,
        min_counts: int = 3,
        n_top_genes: int = 3000,
    ) -> None:
        """Filter on genes of sn_adata, keeping only highly variable genes.
        First back up raw counts into layers['counts'],
        then filter genes, normalize, logarithmize, and find highly variable genes.
        Finally saves this expression matrix into layers['log1p'] and activate back raw counts,
        and only keeps highly variable genes.
        For more preprocessing please use scanpy directly."""
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
