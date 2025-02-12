from scanpy import AnnData as _AnnData
import numpy as _np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.spatial import distance_matrix as _distance_matrix
from .classifier import _LocalClassifier
from .utils import radial_basis_function as _rbf
from .utils import to_array as _to_array
from .utils import _UNDEFINED, _Undefined
from tqdm import tqdm


class _SpatialHandlerBase:
    """A spatial handler to produce filtrations, integrate spots into single cells,
    and estimate their confidences.

    Args:
        adata_spatial (AnnData): subcellular spatial transcriptomic
        AnnData, like Stereo-seq, with .obs[['x', 'y']] indicating
        the locations of spots.

        local_classifier (LocalClassifier): a trained local classifier
        on reference scRNA data.

        threshold_adjacent (float): spots within this distance are
        considered adjacent. for integer-indexed spots, 1.2 for
        4-neighbor adjacency and 1.5 for 8-neighbor adjacency.

        max_spots_per_cell (int): max number of spots of a single
        cell.

        scale_rbf (float): next spot to add is selected from adjacent
        spots, with a radial basis function probability, whose scale
        factor is this parameter.

        allow_cell_overlap (bool): allows cells to share some of the
        spots. The cell-type of a spot is defined as the type it was
        last assigned to."""

    def __init__(
        self,
        adata_spatial: _AnnData,
        local_classifier: _LocalClassifier,
        threshold_adjacent: float = 1.2,
        max_spots_per_cell: int = 81,
        scale_rbf: float = 1.0,
        allow_cell_overlap: bool = True,
    ):
        # Make sure the indices are integer-ized.
        assert _np.all(
            adata_spatial.obs.index.astype(_np.int_)
            == _np.arange(adata_spatial.shape[0])
        ), "Spatial AnnData needs tidying using AnnDataPreparer!"
        self.adata_spatial = adata_spatial
        self.threshold_adjacent = threshold_adjacent
        self.local_classifier = local_classifier

        self.max_spots_per_cell = max_spots_per_cell
        self.scale_rbf = scale_rbf
        self.allow_cell_overlap: bool = allow_cell_overlap

        self._filtrations: dict[int, list[int]] = dict()
        # dict {idx_centroid: [idx_centroid, idx_spot_level1, idx_spot_level2, ...]}
        # If allow_cell_overlap, then same spots might appear multiple times in
        # different filtrations.
        # The dict is an ordered dict, so the idx_centroid maintains its order of
        # being contructed, making it possible to find the last idx_centroid that
        # contains a certain spot.

        self._mask_newIds: NDArray[_np.int_] = _np.full(
            (self.adata_spatial.X.shape[0],), fill_value=-1, dtype=int
        )
        # mask on each old sample. -1 for not assigned; otherwise the new id.
        # If allow_cell_overlap, then theoretically each spot might has multiple
        # new ids. For this we choose the last assigned one. It's just a mask. What
        # is important is whether a spot's id is -1 or not -1; usually we do not care
        # what specific value an id of spot is.
        # If you want the full new ids, refer to self._filtrations.

        self._classes_new: dict[int, int] = dict()
        # new id -> new class

        self._confidences_new: dict[int, float] = dict()
        # new id -> confidence

        self.cache_distance_matrix: NDArray[_np.float_] | _Undefined = _UNDEFINED
        self.cache_aggregated_counts: dict[int, NDArray[_np.int_]] = dict()
        self.cache_singleCellAnnData: _AnnData | _Undefined = _UNDEFINED
        return

    @property
    def threshold_confidence(self) -> float:
        return self.local_classifier.threshold_confidence

    @property
    def has_negative_control(self) -> bool:
        return self.local_classifier.has_negative_control

    @property
    def filtrations(self) -> dict:
        """Return a copy of filtrations."""
        copy_ = dict()
        for k, v in self._filtrations.items():
            copy_[k] = v.copy()
        return copy_

    @property
    def mask_newIds(self) -> NDArray[_np.int_]:
        """Return a copy of mask of new ids.
        (Last come first, if allow_cell_overlap.)"""
        return self._mask_newIds.copy()

    @property
    def masked_spotIds(self) -> NDArray[_np.int_]:
        """Already positively masked spot ids."""
        return _np.where(self.mask_newIds > -1)[0]

    @property
    def unmasked_spotIds(self) -> NDArray[_np.int_]:
        """Unassigned spot ids (those that are -1)."""
        return _np.where(self.mask_newIds == -1)[0]

    @property
    def sampleIds_new(self) -> NDArray[_np.int_]:
        """Return an array of curretly existing new sample indices, EXCLUDING -1."""
        return _np.sort(_np.array(list(set(list(_np.unique(self.mask_newIds))) - {-1})))

    @property
    def classes_new(self) -> dict:
        """Return a copy of cell classes."""
        return self._classes_new.copy()

    @property
    def confidences_new(self) -> dict:
        """Return a copy of confidences of cells."""
        return self._confidences_new.copy()

    def __repr__(self) -> str:
        return f"""--- Spatial Handler (pytacs) ---
- adata_spatial: {self.adata_spatial}
- threshold_adjacent: {self.threshold_adjacent}
- local_classifier: {self.local_classifier}
    + threshold_confidence: {self.threshold_confidence}
    + has_negative_control: {self.has_negative_control}
- max_spots_per_cell: {self.max_spots_per_cell}
- scale_rbf: {self.scale_rbf}
- allow_cell_overlap: {self.allow_cell_overlap}
- filtrations: {len(self.filtrations)} fitted
- single-cell segmentation:
    + new samples: {len(self.sampleIds_new)}
    + AnnData: {self.cache_singleCellAnnData}
--- --- --- --- --- ---
"""

    def clear_cache(self) -> None:
        self.cache_distance_matrix = _UNDEFINED
        self.cache_singleCellAnnData = _UNDEFINED
        self.cache_aggregated_counts = dict()
        return

    def _compute_distance_matrix(self):
        points = self.adata_spatial.obs[["x", "y"]].values
        self.cache_distance_matrix = _distance_matrix(points, points)
        # TODO: here we can define a sparse version of
        # _distance_matrix for saving memory.

        return

    def _find_adjacentOfOneSpot_spotIds(self, idx_this_spot: int) -> NDArray[_np.int_]:
        """Find all adjacent spots, including self."""
        if self.cache_distance_matrix is _UNDEFINED:
            self._compute_distance_matrix()
        assert type(self.cache_distance_matrix) is _np.ndarray
        distances = self.cache_distance_matrix[idx_this_spot, :]
        idxs_adjacent = _np.where(distances <= self.threshold_adjacent)[0]
        return idxs_adjacent

    def _find_adjacentOfManySpots_spotIds(
        self, filtration: list[int]
    ) -> NDArray[_np.int_]:
        """Find all adjacent spots of a list of indices of spots (called filtration),
        EXCLUDING selves, and:
        If allow_cell_overlap, INCLUDING already positively masked spots;
        otherwise, EXCLUDING already positively masked spots.
        """
        assert len(filtration) > 0
        adj_spots = []
        for spot in filtration:
            adj_spots += list(self._find_adjacentOfOneSpot_spotIds(spot))
        return _np.array(
            list(
                (set(adj_spots) - set(filtration))
                - (
                    set(list(self.masked_spotIds))
                    if not self.allow_cell_overlap
                    else set()
                )
            )
        )

    def _buildFiltration_addOneSpot(
        self,
        idx_centroid: int,
        verbose: bool = True,
    ) -> int:
        """Randomly (with rbf probs) adds one spot for the cell
         centered at idx_centroid.

        Filtration list includes idx_centroid itself.

        Update the self.filtrations, self.cache_aggregated_counts,
         and returns the added spot index (-1 for not added)."""
        loc_centroid = self.adata_spatial.obs[["x", "y"]].values[idx_centroid, :]
        self._filtrations[idx_centroid] = self._filtrations.get(
            idx_centroid, [idx_centroid]
        )
        if len(self._filtrations[idx_centroid]) >= self.max_spots_per_cell:
            if verbose:
                print(
                    f"Warning: reaches max_spots_per_cell {
                        self.max_spots_per_cell}"
                )
            return -1
        # Find adjacent candidate spots
        idxs_adjacent = self._find_adjacentOfManySpots_spotIds(
            self._filtrations[idx_centroid]
        )
        if len(idxs_adjacent) == 0:
            if verbose:
                print(
                    f"Warning: no adjacent spots found! Check threshold_adjacent {
                        self.threshold_adjacent}"
                )
            return -1
        # Calculate the probs
        probs_: list[float] = []
        for idx in idxs_adjacent:
            loc_adj = self.adata_spatial.obs[["x", "y"]].values[idx, :]
            probs_.append(_rbf(loc_adj, loc_centroid, scale=self.scale_rbf))
        probs: NDArray[_np.float_] = _np.array(probs_)
        probs /= _np.sum(probs_)
        # Select one randomly
        idx_selected = _np.random.choice(idxs_adjacent, p=probs)
        # Update the filtration
        self._filtrations[idx_centroid].append(idx_selected)
        # Update the aggregated counts cache
        self.cache_aggregated_counts[idx_centroid] = self.cache_aggregated_counts.get(
            idx_centroid, _to_array(self.adata_spatial.X[[idx_centroid], :])[0, :]
        )
        self.cache_aggregated_counts[idx_centroid] += _to_array(
            self.adata_spatial.X[[idx_selected], :]
        )[0, :]

        return idx_selected

    def _aggregate_spots_given_filtration(
        self,
        idx_centroid: int,
    ) -> NDArray:
        """Returns a 1d-array of counts of genes.
        Load from cache.
        """
        self.cache_aggregated_counts[idx_centroid] = self.cache_aggregated_counts.get(
            idx_centroid, _to_array(self.adata_spatial.X[[idx_centroid], :])[0, :]
        )
        return self.cache_aggregated_counts[idx_centroid]

    def _compute_confidence_of_filtration(
        self,
        idx_centroid: int,
    ) -> NDArray[_np.float_]:
        """Calculate confidences of filtrations[idx_centroid] to each class,
        EXCLUDING the nagetive control, if exists."""
        self._filtrations[idx_centroid] = self._filtrations.get(
            idx_centroid, [idx_centroid]
        )
        probas = self.local_classifier.predict_proba(
            X=_np.array(
                [list(self._aggregate_spots_given_filtration(idx_centroid))]
            ),  # TODO: Here could be parallelized.
            genes=self.adata_spatial.var.index,
        )[0, :]
        # Remove the NegativeControl proba
        if self.has_negative_control:
            probas = probas[:-1]
        return probas

    def _buildFiltration_addSpotsUntilConfident(
        self,
        idx_centroid: int,
        n_spots_add_per_step: int = 1,
        verbose: bool = True,
    ) -> tuple[float, int, int]:
        """Find many spots centered at idx_centroid that are confidently in a class.

        Update the self.filtrations, update the self.mask_newIds, self.confidences_new,
         self.classes_new,
         and returns the (confidence, class_id, new_sampleId).
         If reaches max_spots_per_cell and still not confident, returns the
         (confidence, -1, and idx_centroid)."""
        label: int = -1
        confidence: float = 0.0
        for _ in range(self.max_spots_per_cell):
            probas = self._compute_confidence_of_filtration(idx_centroid)
            label = int(_np.argmax(probas))
            confidence = probas[label]
            if confidence >= self.threshold_confidence:
                self._mask_newIds[_np.array(self.filtrations[idx_centroid])] = (
                    idx_centroid
                )
                self._classes_new[idx_centroid] = label
                self._confidences_new[idx_centroid] = confidence
                break
            # Add n cells per step.
            for i_add in range(n_spots_add_per_step):
                idx_added = self._buildFiltration_addOneSpot(
                    idx_centroid, verbose=verbose
                )
                if idx_added == -1:
                    break
            if idx_added == -1:
                label = -1
                break
        else:
            label = -1
        if label == -1:
            # Clear filtrations that are not confident
            del self._filtrations[idx_centroid]
        # Clear aggregated counts cache once their confidences are determined,
        # whether positive or not.
        try:
            del self.cache_aggregated_counts[idx_centroid]
        except KeyError as kerr:
            pass  # Key not existing is allowed.
        return (confidence, label, idx_centroid)

    # Need to be careful with input idx_centroid - you don't want to
    # input idx_centroid that is already positively masked.

    def run_segmentation(
        self,
        n_spots_add_per_step: int = 1,
        coverage_to_stop: float = 0.8,
        max_iter: int = 200,
        verbose: bool = True,
        warnings: bool = False,
        print_summary: bool = True,
    ):
        """Segments the spots into single cells. Spots to query are selected randomly and sequentially.
        Updates self.sampleIds_new, self.confidences_new, self.classes_new."""
        confident_count = 0
        class_count: dict[int, int] = dict()
        for i_iter in range(max_iter):
            if verbose and i_iter % 5 == 0:
                print(f"Iteration {i_iter+1}:")
            available_spots = self.unmasked_spotIds
            if len(available_spots) == 0:
                print("All spots queried. Done.")
                break
            ix_centroid = _np.random.choice(available_spots)
            if verbose and i_iter % 5 == 0:
                print(f"Querying spot {ix_centroid} ...")
            conf, label, _ = self._buildFiltration_addSpotsUntilConfident(
                idx_centroid=ix_centroid,
                n_spots_add_per_step=n_spots_add_per_step,
                verbose=warnings,
            )
            if conf >= self.threshold_confidence:
                confident_count += 1
                class_count[label] = class_count.get(label, 0) + 1
            if verbose and i_iter % 5 == 0:
                print(
                    f"Spot {ix_centroid} | confidence: {conf:.3e} | confident total: {
                        confident_count} | class: {label}"
                )
                print(f"Classes total: {class_count}")
            coverage = (self.mask_newIds > -1).sum() / len(self.mask_newIds)
            if verbose and i_iter % 5 == 0:
                print(f"Coverage: {coverage*100:.2f}%")
            if coverage >= coverage_to_stop:
                break
        else:
            if warnings:
                print(f"Reaches max_iter {max_iter}!")
        if verbose:
            print("Done.")
        if print_summary:
            print(
                f"""--- Summary ---
Queried <={max_iter} spots (with replacement), of which {confident_count} made up confident single cells.
Classes total (this round): {class_count}
Coverage: {coverage*100:.2f}%
--- --- --- --- ---
"""
            )
        return

    def run_getSingleCellAnnData(
        self,
        cache: bool = True,
        force: bool = False,
    ) -> _AnnData:
        """Get segmented single-cell level spatial transcriptomic AnnData.
        Note: cache shares the same id with what this method returns."""
        if (not force) and (not (self.cache_singleCellAnnData is _UNDEFINED)):
            return self.cache_singleCellAnnData
        sc_X = []
        raw_X = self.adata_spatial.X.toarray()
        for ix_new in self.sampleIds_new:
            sc_X.append(list(raw_X[_np.array(self.filtrations[ix_new]), :].sum(axis=0)))
        sc_adata = _AnnData(
            X=_csr_matrix(sc_X),
            obs=self.adata_spatial.obs.copy().iloc[self.sampleIds_new],
            var=self.adata_spatial.var.copy(),
        )
        sc_adata.obs["confidence"] = 0.0
        for ix_new in self.sampleIds_new:
            sc_adata.obs.loc[str(ix_new), "confidence"] = self.confidences_new[ix_new]
        if "cell_type" in sc_adata.obs.columns:
            sc_adata.obs["cell_type_old"] = sc_adata.obs["cell_type"].copy()
        sc_adata.obs["cell_type"] = list(self.classes_new.values())
        # Save cache
        if cache:
            self.cache_singleCellAnnData = sc_adata
        return sc_adata

    def get_spatial_classes(self) -> NDArray[_np.int_]:
        """Get an array of integers, corresponding to class ids of each old sample (spot)."""
        res = _np.zeros(shape=(self.adata_spatial.shape[0],), dtype=int)
        for i_sample in range(len(res)):
            new_id = self.mask_newIds[i_sample]
            if new_id == -1:
                res[i_sample] = -1
                continue
            new_class = self.classes_new[new_id]
            res[i_sample] = new_class
        return res

    def run_plotClasses(self):
        import seaborn as sns

        spatial_classes = self.get_spatial_classes().astype(str)
        hue_order = _np.sort(_np.unique(spatial_classes))
        if "-1" == hue_order[0]:
            hue_order[:-1] = hue_order[1:]
            hue_order[-1] = "-1"
        return sns.scatterplot(
            x=self.adata_spatial.obs["x"].values,
            y=self.adata_spatial.obs["y"].values,
            hue=spatial_classes,
            hue_order=hue_order,
        )

    def run_plotNewIds(self):
        new_ids = self.mask_newIds
        import seaborn as sns

        return sns.scatterplot(
            x=self.adata_spatial.obs["x"].values,
            y=self.adata_spatial.obs["y"].values,
            hue=new_ids.astype(_np.str_),
        )


class SpatialHandler(_SpatialHandlerBase):
    """A spatial handler to produce filtrations
    self-guidedly (autopilot), integrate spots into single cells,
    and estimate their confidences.

    Args:
        adata_spatial (AnnData): subcellular spatial transcriptomic
        AnnData, like Stereo-seq, with .obs[['x', 'y']] indicating
        the locations of spots.

        local_classifier (LocalClassifier): a trained local classifier
        on reference scRNA data.

        threshold_adjacent (float): spots within this distance are
        considered adjacent. for integer-indexed spots, 1.2 for
        4-neighbor adjacency and 1.5 for 8-neighbor adjacency.

        max_spots_per_cell (int): max number of spots of a single
        cell.

        scale_rbf (float): next spot to add is selected from adjacent
        spots, with a coefficient of radial basis function probability,
        whose scale factor is this parameter.

        threshold_delta_n_features (int): after reaching threshold_confidence,
        the process of building a filtration (adding spots to it) would not
        stop until the stepwise change in n_features is less than
        `threshold_delta_n_features`.

        allow_cell_overlap (bool): allows cells to share some of the
        spots. The cell-type of a spot is defined as the type it was
        last assigned to.
    """

    def __init__(
        self,
        adata_spatial: _AnnData,
        local_classifier: _LocalClassifier,
        threshold_adjacent: float = 1.2,
        max_spots_per_cell: int = 81,
        scale_rbf: float = 1.0,
        threshold_delta_n_features: int = 10,
        allow_cell_overlap: bool = True,
    ) -> None:
        super().__init__(
            adata_spatial=adata_spatial,
            local_classifier=local_classifier,
            threshold_adjacent=threshold_adjacent,
            max_spots_per_cell=max_spots_per_cell,
            scale_rbf=scale_rbf,
            allow_cell_overlap=allow_cell_overlap,
        )
        self.threshold_delta_n_features: int = threshold_delta_n_features
        self.cache_n_features: dict[int, list[int]] = dict()
        self._premapped: bool = False
        return

    # Overwrite
    def __repr__(self) -> str:
        return f"""--- Spatial Handler Autopilot (pytacs) ---
- adata_spatial: {self.adata_spatial}
- threshold_adjacent: {self.threshold_adjacent}
- local_classifier: {self.local_classifier}
    + threshold_confidence: {self.threshold_confidence}
    + has_negative_control: {self.has_negative_control}
- max_spots_per_cell: {self.max_spots_per_cell}
- scale_rbf: {self.scale_rbf}
- allow_cell_overlap: {self.allow_cell_overlap}
- pre-mapped: {self._premapped}
- filtrations: {len(self.filtrations)} fitted
- single-cell segmentation:
    + new samples: {len(self.sampleIds_new)}
    + AnnData: {self.cache_singleCellAnnData}
--- --- --- --- --- ---
"""

    # Overwrite
    def clear_cache(self):
        self.cache_n_features = dict()
        super().clear_cache()
        return

    def _firstRound_preMapping(self) -> None:
        """Updates .obsm['confidence_premapping1'].

        After the first round mapping, each spot has a confidence.
        But some (or many) spots have rather low confidences due
        to sparsity. They would almost always be left the last ones
        to be added to filtrations preferably by the filtration
        builder when adding next-spots to it, potentially causing
        bias. This could be addressed by performing a second-round
        pre-mapping, which takes context into consideration."""

        # >>> Temporarily change the confidence threshold
        # We do not want spots to be predicted as class -1
        threshold_confidence_old: float = self.threshold_confidence
        self.local_classifier.set_threshold_confidence(0.0)
        confidence_premapping: NDArray[_np.float_] = (
            self.local_classifier.predict_proba(
                X=_to_array(self.adata_spatial.X), genes=self.adata_spatial.var.index
            )
        )
        if self.has_negative_control:
            confidence_premapping = confidence_premapping[:, :-1]
            # only preserves real classes (ids).
        self.adata_spatial.obsm["confidence_premapping1"] = confidence_premapping

        # <<< Reset the confidence threshold
        self.local_classifier.set_threshold_confidence(value=threshold_confidence_old)
        return

    def _secondRound_preMapping(self) -> None:
        """Updates self.adata_spatial.obs['cell_type_premapping2']
        and .obsm['confidence_premapping2'].

        After the first round mapping, each spot has a confidence.
        But some (or many) spots have rather low confidences due
        to sparsity. They would almost always be left the last ones
        to be added to filtrations preferably by the filtration
        builder when adding next-spots to it, potentially causing
        bias. This could be addressed by performing a second-round
        pre-mapping.

        The second-round pre-mapping takes into account context
        information."""

        self.adata_spatial.obsm["confidence_premapping2"] = self.adata_spatial.obsm[
            "confidence_premapping1"
        ].copy()

        for i_spot in tqdm(range(self.adata_spatial.shape[0])):
            # Get adjacent neighbors
            ixs_adj: NDArray[_np.int_] = super()._find_adjacentOfOneSpot_spotIds(i_spot)
            # Exlucding self
            ixs_adj = _np.array(list(set(ixs_adj) - {i_spot}))
            if len(ixs_adj) == 0:  # if no neighbors, skip
                continue
            # Extract confidences
            confidences_adj: NDArray[_np.float_] = self.adata_spatial.obsm[
                "confidence_premapping1"
            ][ixs_adj, :]
            # If confident enough, skip
            if (
                _np.max(self.adata_spatial.obsm["confidence_premapping1"][i_spot, :])
                >= self.threshold_confidence
            ):
                continue
            # Probs within context
            self.adata_spatial.obsm["confidence_premapping2"][i_spot, :] *= _np.mean(
                confidences_adj, axis=0
            )
            # Normalize to sum of 1
            self.adata_spatial.obsm["confidence_premapping2"][i_spot, :] /= _np.sum(
                self.adata_spatial.obsm["confidence_premapping2"][i_spot, :]
            )

        # Annotate the second-round cell-type
        self.adata_spatial.obs["cell_type_premapping2"] = _np.argmax(
            self.adata_spatial.obsm["confidence_premapping2"],
            axis=1,
        )
        return

    def _buildFiltration_addOneSpot(self, idx_centroid, verbose=True):
        """Adds one spot for the cell centered at idx_centroid.

        Filtration list includes idx_centroid itself.

        Update the self.filtrations, self.cache_aggregated_counts,
         self.cache_n_features,
         and returns the added spot index (-1 for not added)."""
        loc_centroid: _np.ndarray = self.adata_spatial.obs[["x", "y"]].values[
            idx_centroid, :
        ]
        self._filtrations[idx_centroid] = self._filtrations.get(
            idx_centroid, [idx_centroid]
        )
        self.cache_aggregated_counts[idx_centroid] = self.cache_aggregated_counts.get(
            idx_centroid, _to_array(self.adata_spatial.X[[idx_centroid], :])[0, :]
        )
        self.cache_n_features[idx_centroid] = self.cache_n_features.get(
            idx_centroid, [int((self.cache_aggregated_counts[idx_centroid] > 0).sum())]
        )

        # Stop if max_spots_per_cell reached
        if len(self._filtrations[idx_centroid]) >= self.max_spots_per_cell:
            if verbose:
                print(
                    f"Warning: reaches max_spots_per_cell {
                        self.max_spots_per_cell}"
                )
            return -1
        # Find adjacent candidate spots
        idxs_adjacent = self._find_adjacentOfManySpots_spotIds(
            self._filtrations[idx_centroid]
        )
        if len(idxs_adjacent) == 0:
            if verbose:
                print(
                    f"Warning: no adjacent spots found! Check threshold_adjacent {
                        self.threshold_adjacent}"
                )
            return -1
        # Calculate the probs
        probs_: list[float] = []
        for idx in idxs_adjacent:
            loc_adj = self.adata_spatial.obs[["x", "y"]].values[idx, :]
            probs_.append(
                _rbf(
                    loc_adj,
                    loc_centroid,
                    scale=self.scale_rbf,
                )
                * self.adata_spatial.obsm["confidence_premapping2"][
                    idx,
                    self.adata_spatial.obs.loc[
                        str(idx_centroid), "cell_type_premapping2"
                    ],
                ]
            )
        probs: NDArray[_np.float_] = _np.array(probs_)
        probs /= _np.sum(probs_)
        # Select one randomly
        idx_selected = _np.random.choice(idxs_adjacent, p=probs)
        # Update the filtration
        self._filtrations[idx_centroid].append(idx_selected)
        # Update the aggregated counts cache, n_features cache
        self.cache_aggregated_counts[idx_centroid] += _to_array(
            self.adata_spatial.X[[idx_selected], :]
        )[0, :]
        self.cache_n_features[idx_centroid].append(
            int((self.cache_aggregated_counts[idx_centroid] > 0).sum())
        )
        return idx_selected

    def _buildFiltration_addSpotsUntilConfident(
        self,
        idx_centroid: int,
        n_spots_add_per_step: int = 1,
        verbose: bool = True,
    ) -> tuple[float, int, int]:
        """Find many spots centered at idx_centroid that are confidently in a class.

        Update the self.filtrations, update the self.mask_newIds, self.confidences_new,
         self.classes_new,
         and returns the (confidence, class_id, new_sampleId).
        Keeps building until self.threshold_delta_n_features met or max_spots_per_cell
         met, whether confident or not.
        If reaches max_spots_per_cell and still not confident, returns the
         (confidence, -1, and idx_centroid)."""
        label: int = -1  # cell type assigned, -1 for not confident
        confidence: float = 0.0
        n_features_old: int = 0
        self.cache_n_features[idx_centroid] = self.cache_n_features.get(
            idx_centroid,
            [int((self.cache_aggregated_counts[idx_centroid] > 0).sum())],
        )
        for _ in range(self.max_spots_per_cell):
            probas = self._compute_confidence_of_filtration(idx_centroid)
            label = int(_np.argmax(probas))
            n_features: int = self.cache_n_features[idx_centroid][-1]
            # Dynamically changes premapped cell-type
            self.adata_spatial.obs.loc[str(idx_centroid), "cell_type_premapping2"] = (
                label
            )
            confidence = probas[label]
            if confidence >= self.threshold_confidence:
                self._mask_newIds[_np.array(self.filtrations[idx_centroid])] = (
                    idx_centroid
                )
                self._classes_new[idx_centroid] = label
                self._confidences_new[idx_centroid] = confidence
                if n_features - n_features_old <= self.threshold_delta_n_features:
                    break
            n_features_old = n_features
            # Add n spots per step.
            for i_add in range(n_spots_add_per_step):
                idx_added = self._buildFiltration_addOneSpot(
                    idx_centroid, verbose=verbose
                )
                if idx_added == -1:
                    # Final calculation
                    probas = self._compute_confidence_of_filtration(idx_centroid)
                    label = int(_np.argmax(probas))
                    confidence = probas[label]
                    if confidence >= self.threshold_confidence:
                        self._mask_newIds[_np.array(self.filtrations[idx_centroid])] = (
                            idx_centroid
                        )
                        self._classes_new[idx_centroid] = label
                        self._confidences_new[idx_centroid] = confidence
                    break
            if idx_added == -1:
                if confidence < self.threshold_confidence:
                    label = -1
                break
        else:
            if confidence < self.threshold_confidence:
                label = -1
        if label == -1:
            # Clear filtrations that are not confident
            del self._filtrations[idx_centroid]
            del self.cache_n_features[idx_centroid]
        # Clear aggregated counts cache once their confidences are determined,
        # whether positive or not.
        del self.cache_aggregated_counts[idx_centroid]

        return (confidence, label, idx_centroid)

    def run_preMapping(self, verbose: bool = True) -> None:
        if verbose:
            print("Running first round premapping ...")
        self._firstRound_preMapping()
        if verbose:
            print("Running second round premapping ...")
        self._secondRound_preMapping()
        self._premapped = True
        if verbose:
            print("Done.")
        return

    # Overwrite
    def run_segmentation(
        self,
        n_spots_add_per_step: int = 1,
        coverage_to_stop: float = 0.8,
        max_iter: int = 200,
        verbose: bool = True,
        warnings: bool = False,
        print_summary: bool = True,
    ):
        assert self._premapped, "Must .run_preMapping() first!"
        return super().run_segmentation(
            n_spots_add_per_step,
            coverage_to_stop,
            max_iter,
            verbose,
            warnings,
            print_summary,
        )


class SpatialHandlerParallel(SpatialHandler):
    """A spatial handler to produce filtrations
    self-guidedly, integrate spots into single cells,
    and estimate their confidences, in a parallel manner.

    Parallelism makes computation faster, but requires more memory.

    Note that this module always allows cells to share some of the
    spots. The cell-type of a spot is defined as the type it was
    last assigned to.

    Args:
        adata_spatial (AnnData): subcellular spatial transcriptomic
        AnnData, like Stereo-seq, with .obs[['x', 'y']] indicating
        the locations of spots.

        local_classifier (LocalClassifier): a trained local classifier
        on reference scRNA data.

        threshold_adjacent (float): spots within this distance are
        considered adjacent. for integer-indexed spots, 1.2 for
        4-neighbor adjacency and 1.5 for 8-neighbor adjacency.

        max_spots_per_cell (int): max number of spots of a single
        cell.

        scale_rbf (float): next spot to add is selected from adjacent
        spots, with a coefficient of radial basis function probability,
        whose scale factor is this parameter.

        threshold_delta_n_features (int): after reaching threshold_confidence,
        the process of building a filtration (adding spots to it) would not
        stop until the stepwise change in n_features is less than
        `threshold_delta_n_features`.

        n_parallel (int): build `n_parallel` filtrations parallelly through a
        vector-broadcasting mechanism.
    """

    def __init__(
        self,
        adata_spatial: _AnnData,
        local_classifier: _LocalClassifier,
        threshold_adjacent: float = 1.2,
        max_spots_per_cell: int = 81,
        scale_rbf: float = 1.0,
        threshold_delta_n_features: int = 10,
        n_parallel: int = 50,
    ):
        super().__init__(
            adata_spatial=adata_spatial,
            local_classifier=local_classifier,
            threshold_adjacent=threshold_adjacent,
            max_spots_per_cell=max_spots_per_cell,
            scale_rbf=scale_rbf,
            threshold_delta_n_features=threshold_delta_n_features,
            allow_cell_overlap=True,
        )
        self.n_parallel: int = n_parallel
        return

    # Overwrite
    def __repr__(self) -> str:
        return f"""--- Spatial Handler Autopilot Parallel (pytacs) ---
- adata_spatial: {self.adata_spatial}
- threshold_adjacent: {self.threshold_adjacent}
- local_classifier: {self.local_classifier}
    + threshold_confidence: {self.threshold_confidence}
    + has_negative_control: {self.has_negative_control}
- max_spots_per_cell: {self.max_spots_per_cell}
- scale_rbf: {self.scale_rbf}
- pre-mapped: {self._premapped}
- n_parallel: {self.n_parallel}
- filtrations: {len(self.filtrations)} fitted
- single-cell segmentation:
    + new samples: {len(self.sampleIds_new)}
    + AnnData: {self.cache_singleCellAnnData}
--- --- --- --- --- ---
"""

    # Overwrite
    def _aggregate_spots_given_filtration(
        self,
        idx_centroids: NDArray[_np.int_],
    ) -> NDArray[_np.int_ | _np.float_]:
        """Returns a 2d-array of counts of genes (idx-by-gene). Load from cache."""
        dtype = self.adata_spatial.X.dtype
        result: _np.ndarray = _np.zeros(
            shape=(len(idx_centroids), self.adata_spatial.shape[1]),
            dtype=dtype,
        )
        for i_idx, idx in enumerate(idx_centroids):
            self.cache_aggregated_counts[idx] = self.cache_aggregated_counts.get(
                idx,
                _to_array(self.adata_spatial.X[[idx], :])[0, :],
            )
            result[i_idx, :] = self.cache_aggregated_counts[idx]
        return result

    # Overwrite
    def _compute_confidence_of_filtration(
        self,
        idx_centroids: NDArray[_np.int_],
    ) -> NDArray[_np.float_]:
        """Calculate confidences of filtrations to each class,
        EXCLUDING the negative control, if exists.
        Return an idx-by-class 2d-array."""
        # Collect filtrations
        for idx in idx_centroids:
            self._filtrations[idx] = self._filtrations.get(idx, [idx])
        probas: NDArray[_np.float_] = self.local_classifier.predict_proba(
            X=self._aggregate_spots_given_filtration(idx_centroids),
            genes=self.adata_spatial.var.index,
        )
        # Only keeps positive classes
        if self.has_negative_control:
            probas = probas[:-1]
        return probas

    def _buildFiltration_addSpotsUntilConfident(
        self,
        idx_centroids: NDArray[_np.int_],
        n_spots_add_per_step: int = 1,
        verbose: bool = True,
    ) -> tuple[NDArray[_np.float_], NDArray[_np.int_], NDArray[_np.int_]]:
        """Find many spots centered at idx_centroids that are confidently in a class.

        Update the self.filtrations, update the self.mask_newIds, self.confidences_new,
         self.classes_new,
         and returns the (confidences, class_ids, new_sampleIds).
        Keeps building until self.threshold_delta_n_features met or max_spots_per_cell
         met, whether confident or not.
        If reaches max_spots_per_cell and still not confident, that class_id is set to -1.
        """
        labels: NDArray[_np.int_] = _np.full(
            shape=len(idx_centroids),
            fill_value=-1,
            dtype=int,
        )  # cell types assigned, -1 for not confident
        confidences: NDArray[_np.float_] = _np.full(
            shape=len(idx_centroids),
            fill_value=0.0,
            dtype=float,
        )
        nums_features_old: NDArray[_np.int_] = _np.zeros(
            shape=len(idx_centroids),
            dtype=int,
        )
        for idx in idx_centroids:
            self.cache_n_features[idx] = self.cache_n_features.get(
                idx,
                [int((self.cache_aggregated_counts[idx] > 0).sum())],
            )
        for i_step_add_spot in range(self.max_spots_per_cell):
            probas = self._compute_confidence_of_filtration(idx_centroids)
            labels = _np.argmax(probas, axis=1)
            nums_features: NDArray[_np.int_] = _np.array(
                [self.cache_n_features[idx][-1] for idx in idx_centroids]
            )
            # Dynamically changes premapped cell-type
            self.adata_spatial.obs.loc[
                _np.array(idx_centroids).astype(str), "cell_type_premapping2"
            ] = labels
            confidences = probas[_np.arange(len(idx_centroids)), labels]
            for i_idx, confidence in enumerate(confidences):
                idx: int = idx_centroids[i_idx]
                if confidence >= self.threshold_confidence:
                    self._mask_newIds[_np.array(self.filtrations[idx])] = idx
                    self._classes_new[idx] = labels[i_idx]
                    self._confidences_new[idx] = confidence
                if (
                    nums_features[i_idx] - nums_features_old[i_idx]
                    > self.threshold_delta_n_features
                ) or (confidence < self.threshold_confidence):
                    # Add n spots
                    for i_add in range(n_spots_add_per_step):
                        idx_added = self._buildFiltration_addOneSpot(
                            idx,
                            verbose,
                        )
                        if idx_added == -1:  # exhausted
                            break
                    nums_features_old[i_idx] = nums_features[i_idx]
                    continue
            # Stop criteria
            if _np.all(confidences > self.threshold_confidence) and _np.all(
                nums_features - nums_features_old <= self.threshold_delta_n_features
            ):
                break

        else:  # reaches max_spots
            labels[confidences < self.threshold_confidence] = -1
        # Clear unconfident caches
        for i_idx, label in enumerate(labels):
            if label == -1:
                idx = idx_centroids[i_idx]
                del self._filtrations[idx]
                del self.cache_n_features[idx]
            # Clear aggregated counts cache once their confidences are determined,
            # whether positive or not.
            del self.cache_aggregated_counts[idx]

        return (
            confidences,
            labels,
            idx_centroids,
        )

    # Overwrite
    def run_segmentation(
        self,
        n_spots_add_per_step: int = 1,
        coverage_to_stop: float = 0.8,
        max_iter: int = 200,
        verbose: bool = True,
        warnings: bool = False,
        print_summary: bool = True,
    ):
        """Segments the spots into single cells. Spots to query are selected sequentially,
        `self.n_parallel` in a batch.
        Updates self.sampleIds_new, self.confidences_new, self.classes_new."""
        assert self._premapped, "Must .run_preMapping() first!"
        confident_count = 0
        class_count: dict[int, int] = dict()
        queried_spotIds = set()
        for i_iter in range(max_iter):
            if verbose and i_iter % 5 == 0:
                print(f"Iter {i_iter+1}:")
            available_spots: list[int] = list(
                set(self.unmasked_spotIds) - queried_spotIds
            )
            if len(available_spots) == 0:
                print("All spots queried.")
                break
            idx_centroids: NDArray[_np.int_] = _np.random.choice(
                a=available_spots,
                size=min(self.n_parallel, len(available_spots)),
                replace=False,
            )
            queried_spotIds |= set(idx_centroids)
            if verbose and i_iter % 5 == 0:
                print(f"Querying {len(idx_centroids)} spots ...")
            confs, labels, _ = self._buildFiltration_addSpotsUntilConfident(
                idx_centroids=idx_centroids,
                n_spots_add_per_step=n_spots_add_per_step,
                verbose=warnings,
            )
            confident_count += (confs >= self.threshold_confidence).sum()
            for label in labels:
                class_count[label] = class_count.get(label, 0) + 1
            if verbose and i_iter % 5 == 0:
                print(
                    f"Cell {idx_centroids[0]}, ... | Confidence: {confs[0]:.3e}, ... | Confident total: {confident_count} | class: {labels[0]}, ..."
                )
                print(f"Classes total: {class_count}")
            coverage = (self.mask_newIds > -1).sum() / len(self.mask_newIds)
            if verbose and i_iter % 5 == 0:
                print(f"Coverage: {coverage*100:.2f}%")
            if coverage == coverage_to_stop:
                break
        else:
            if warnings:
                print(f"Reaches max_iter {max_iter}!")
        if verbose:
            print("Done.")
        if print_summary:
            print(
                f"""--- Summary ---
Queried {len(queried_spotIds)} spots, of which {confident_count} made up confident single cells.
Classes total: {class_count}
Coverage: {coverage*100:.2f}%
--- --- --- --- ---
"""
            )
        return
