"""Deprecated."""

from scanpy import AnnData as _AnnData
import numpy as _np
from numpy.typing import NDArray as _NDArray
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.sparse import dok_matrix as _dok_matrix  # for cache_aggregated_counts
from scipy.spatial import cKDTree as _cKDTree  # to construct sparse distance matrix

from scipy.cluster.hierarchy import linkage as _linkage
from scipy.cluster.hierarchy import fcluster as _fcluster

from .classifier import _LocalClassifier
from .utils import radial_basis_function as _rbf
from .utils import to_array as _to_array
from .utils import _UNDEFINED, _UndefinedType

# from multiprocess.pool import Pool as _Pool
from tqdm import tqdm as _tqdm

# from .utils import deepcopy_dict as _deepcopy_dict


class SpatialHandler:
    """A spatial handler to produce filtrations, integrate spots into single cells,
    and estimate their confidences.
    Note that this module always allows cells to share some of the spots.

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
        spots, relying partly on a radial basis function probability, whose scale
        factor is this parameter.

        max_distance (float): distance greater than this value is considered infinite
        for ease of RAM.
    """

    def __init__(
        self,
        adata_spatial: _AnnData,
        local_classifier: _LocalClassifier,
        threshold_adjacent: float = 1.2,
        max_spots_per_cell: int = 60,
        scale_rbf: float = 20.0,
        max_distance: float = 10.0,
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
        self.max_distance = max_distance
        self._premapped: bool = False

        self._filtrations: dict[int, list[int]] = dict()
        # dict {idx_centroid: [idx_centroid, idx_spot_level1, idx_spot_level2, ...]}
        # Same spots might appear multiple times in
        # different filtrations.

        self._mask_newIds: _NDArray[_np.int_] = _np.full(
            (self.adata_spatial.X.shape[0],), fill_value=-1, dtype=int
        )
        # mask on each old sample. -1 for not assigned; otherwise the last assigned
        # new id.
        # Theoretically each spot might has multiple
        # new ids. For this mask we choose the last assigned one. It's just a mask.
        # What is important is whether a spot's id is -1 or not -1;
        # usually we do not care
        # what specific value an id of spot is.
        # But beware that if a spot is a centroid of an aggregated corpus, then
        # the original id of the spot itself is usually more representative
        # than that in this mask.
        # If you want the full new ids, refer to self._filtrations.

        self._classes_new: dict[int, int] = dict()
        # new id -> new class

        self._confidences_new: dict[int, float] = dict()
        # new id -> confidence

        self.cache_distance_matrix: _dok_matrix | _UndefinedType = _UNDEFINED
        self.cache_aggregated_counts: _dok_matrix = _dok_matrix(
            self.adata_spatial.shape, dtype=int
        )
        self.cache_singleCellAnnData: _AnnData | _UndefinedType = _UNDEFINED
        return

    @property
    def threshold_confidence(self) -> float:
        return self.local_classifier.threshold_confidence

    @property
    def filtrations(self) -> dict:
        """Return a copy of filtrations."""
        copy_ = dict()
        for k, v in self._filtrations.items():
            copy_[k] = v.copy()
        return copy_

    @property
    def mask_newIds(self) -> _NDArray[_np.int_]:
        """Return a copy of mask of new ids.
        Note! A last-come-first strategy
        is used, in which case, each spot
        might be assigned to more than one cell, meaning that
        each cell might have more than one new sample id.
        In this case, !! DO NOT !! use this mask for sample_id finding
        because some new ids might be overhidden by other cells, but
        just use this property as an indicator of whether a spot
        is being assigned to a cell.
        If you want to find the UNIQUE new id of each cell, use
        .filtrations.keys().
        """
        return self._mask_newIds.copy()

    @property
    def masked_spotIds(self) -> _NDArray[_np.int_]:
        """Already positively masked spot ids."""
        return _np.where(self.mask_newIds > -1)[0]

    @property
    def unmasked_spotIds(self) -> _NDArray[_np.int_]:
        """Unassigned spot ids (those that are -1)."""
        return _np.where(self.mask_newIds == -1)[0]

    @property
    def sampleIds_new(self) -> _NDArray[_np.int_]:
        """Return an array of currently existing new sample indices, EXCLUDING -1."""
        return _np.sort(list(self._filtrations.keys()))

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
- max_spots_per_cell: {self.max_spots_per_cell}
- scale_rbf: {self.scale_rbf}
- pre-mapped: {self._premapped}
- filtrations: {len(self.filtrations)} fitted
- single-cell segmentation:
    + new samples: {len(self.sampleIds_new)}
    + AnnData: {self.cache_singleCellAnnData}
--- --- --- --- --- ---
"""

    def clear_cache(self) -> None:
        self.cache_distance_matrix = _UNDEFINED
        self.cache_singleCellAnnData = _UNDEFINED
        self.cache_aggregated_counts = _dok_matrix(self.adata_spatial.shape, dtype=int)
        return

    def _compute_distance_matrix(self):
        points = self.adata_spatial.obs[["x", "y"]].values
        ckdtree_points = _cKDTree(points)
        self.cache_distance_matrix = ckdtree_points.sparse_distance_matrix(
            other=ckdtree_points,
            max_distance=self.max_distance,
            p=2,
            output_type="dok_matrix",
        )
        return

    def _find_adjacentOfOneSpot_spotIds(self, idx_this_spot: int) -> _NDArray[_np.int_]:
        """Find all adjacent spots, including self."""
        if isinstance(self.cache_distance_matrix, _UndefinedType):
            self._compute_distance_matrix()
        assert isinstance(self.cache_distance_matrix, _dok_matrix)
        distances = _to_array(
            self.cache_distance_matrix[[idx_this_spot], :], squeeze=True
        )
        idxs_adjacent = _np.where(
            (0 < distances) * (distances <= self.threshold_adjacent)
        )[0]
        idxs_adjacent = _np.array(
            [idx_this_spot] + list(idxs_adjacent)
        )  # including self
        return idxs_adjacent

    def _find_adjacentOfManySpots_spotIds(
        self, filtration: list[int]
    ) -> _NDArray[_np.int_]:
        """Find all adjacent spots of a list of indices of spots (called filtration),
        EXCLUDING selves, and
        INCLUDING already positively masked spots.
        """
        assert len(filtration) > 0
        adj_spots = []
        for spot in filtration:
            adj_spots += list(self._find_adjacentOfOneSpot_spotIds(spot))
        return _np.array(list(set(adj_spots) - set(filtration)))

    def _firstRound_preMapping(self, n_parallel: int = 1000) -> None:
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
        confidence_premapping: _NDArray[_np.float_] = _np.zeros(
            shape=(self.adata_spatial.shape[0], len(self.local_classifier.classes)),
        )
        for i_batch in _tqdm(
            range(confidence_premapping.shape[0] // n_parallel + 1),
            desc="1st premapping",
            ncols=60,
        ):
            i_samples = _np.arange(
                i_batch * n_parallel,
                min((i_batch + 1) * n_parallel, self.adata_spatial.shape[0]),
            )
            if len(i_samples) == 0:
                continue
            confidence_premapping[i_samples, :] = self.local_classifier.predict_proba(
                X=_to_array(self.adata_spatial.X[i_samples, :]),
                genes=self.adata_spatial.var.index,
            )

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

        for i_spot in _tqdm(
            range(self.adata_spatial.shape[0]),
            desc="2nd premapping",
            ncols=60,
        ):
            # Get adjacent neighbors
            ixs_adj: _NDArray[_np.int_] = self._find_adjacentOfOneSpot_spotIds(i_spot)
            # Excluding self
            ixs_adj = _np.array(list(set(ixs_adj) - {i_spot}))
            if len(ixs_adj) == 0:  # if no neighbors, skip
                continue
            # Extract confidences
            confidences_adj: _NDArray[_np.float_] = self.adata_spatial.obsm[
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

    def _aggregate_spots_given_filtration(
        self,
        idx_centroid: int,
    ) -> _NDArray:
        """Returns a 1d-array of counts of genes.
        Load from cache.
        """
        return _to_array(self.cache_aggregated_counts[[idx_centroid], :], squeeze=True)

    def _compute_confidence_of_filtration(
        self,
        idx_centroids: _NDArray[_np.int_],
    ) -> _NDArray[_np.float_]:
        """Calculate confidences of filtrations to each class
        Return an idx-by-class 2d-array."""
        # Collect filtrations
        for idx in idx_centroids:
            if idx not in self._filtrations:
                self._filtrations[idx] = [idx]
                self.cache_aggregated_counts[idx, :] = self.adata_spatial.X[
                    idx, :
                ].copy()
        probas: _NDArray[_np.float_] = self.local_classifier.predict_proba(
            X=_to_array(self.cache_aggregated_counts[idx_centroids, :]),
            genes=self.adata_spatial.var.index,
        )
        return probas

    def _buildFiltration_addOneSpot(self, idx_centroid, verbose=True) -> int:
        """Adds one spot for the cell centered at idx_centroid.

        Filtration list includes idx_centroid itself.

        Update the self.filtrations, self.cache_aggregated_counts,
         and returns the added spot index (-1 for not added)."""
        loc_centroid: _np.ndarray = self.adata_spatial.obs[["x", "y"]].values[
            idx_centroid, :
        ]
        if idx_centroid not in self._filtrations:
            self._filtrations[idx_centroid] = [idx_centroid]
            self.cache_aggregated_counts[idx_centroid, :] = self.adata_spatial.X[
                idx_centroid, :
            ].copy()

        # Stop if max_spots_per_cell reached
        if len(self._filtrations[idx_centroid]) >= self.max_spots_per_cell:
            if verbose:
                _tqdm.write(
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
                _tqdm.write(
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
        probs: _NDArray[_np.float_] = _np.array(probs_)
        probs /= _np.sum(probs_)
        # Select one randomly
        idx_selected = _np.random.choice(idxs_adjacent, p=probs)
        # Update the filtration
        self._filtrations[idx_centroid].append(idx_selected)
        # Update the aggregated counts cache
        self.cache_aggregated_counts[idx_centroid, :] += self.adata_spatial.X[
            idx_selected, :
        ]

        return idx_selected

    def _buildFiltration_addSpotsUntilConfident(
        self,
        idx_centroids: _NDArray[_np.int_],
        n_spots_add_per_step: int = 1,
        verbose: bool = True,
    ) -> tuple[_NDArray[_np.float_], _NDArray[_np.int_], _NDArray[_np.int_]]:
        """Find many spots centered at idx_centroids that are confidently in a class.

        Update the self.filtrations, update the self.mask_newIds, self.confidences_new,
         self.classes_new,
         and returns the (confidences, class_ids, new_sampleIds).
        Keeps building until max_spots_per_cell met.
        If reaches max_spots_per_cell and still not confident, that class_id is set to -1.
        """
        labels: _NDArray[_np.int_] = _np.full(
            shape=len(idx_centroids),
            fill_value=-1,
            dtype=int,
        )  # cell types assigned, -1 for not confident
        confidences: _NDArray[_np.float_] = _np.full(
            shape=len(idx_centroids),
            fill_value=0.0,
            dtype=float,
        )

        where_running = _np.arange(len(labels))  # Running terms
        for i_step_add_spot in _tqdm(
            range(self.max_spots_per_cell // n_spots_add_per_step + 1),
            desc="Building a batch of cells",
            ncols=60,
        ):
            probas = self._compute_confidence_of_filtration(
                idx_centroids[where_running]
            )
            labels[where_running] = _np.argmax(probas, axis=1)

            # Dynamically changes premapped cell-type
            self.adata_spatial.obs.loc[
                idx_centroids[where_running].astype(str), "cell_type_premapping2"
            ] = labels[where_running]
            confidences[where_running] = probas[
                _np.arange(len(where_running)), labels[where_running]
            ]
            where_to_drop = []
            for i_idx, confidence in enumerate(confidences[where_running]):
                idx: int = idx_centroids[where_running][i_idx]
                if confidence >= self.threshold_confidence:
                    self._mask_newIds[_np.array(self.filtrations[idx])] = idx
                    self._classes_new[idx] = labels[where_running[i_idx]]
                    self._confidences_new[idx] = confidence
                    # Mark confident and drop
                    where_to_drop.append(i_idx)
                    continue
                # Add n spots
                for i_add in range(n_spots_add_per_step):
                    idx_added = self._buildFiltration_addOneSpot(
                        idx,
                        verbose,
                    )
                    if idx_added == -1:  # exhausted
                        # Final test and mark unconfident and drop
                        proba_ = self._compute_confidence_of_filtration(
                            _np.array([idx])
                        )[0, :]
                        label_ = _np.argmax(proba_)
                        conf_ = proba_[label_]
                        confidences[where_running[i_idx]] = conf_
                        if conf_ >= self.threshold_confidence:
                            labels[where_running[i_idx]] = label_
                            self._mask_newIds[_np.array(self.filtrations[idx])] = idx
                            self._classes_new[idx] = label_
                            self._confidences_new[idx] = conf_
                        else:
                            labels[where_running[i_idx]] = -1
                        # Drop this
                        where_to_drop.append(i_idx)
                        break
            # Update running indices
            where_running = _np.array(list(set(where_running) - set(where_to_drop)))
            # Stop criteria
            if len(where_running) == 0:
                break
        labels[confidences < self.threshold_confidence] = -1
        # Clear unconfident caches
        for i_idx, label in enumerate(labels):
            idx = idx_centroids[i_idx]
            if label == -1:
                del self._filtrations[idx]
            # Clear aggregated counts cache once their confidences are determined,
            # whether positive or not.
            self.cache_aggregated_counts[idx, :] = 0

        return (
            confidences,
            labels,
            idx_centroids,
        )

    def run_preMapping(
        self,
        n_parallel: int = 1000,
        skip_2nd: bool = False,
    ) -> None:
        """Sometimes 2nd round premapping can be time costing. Use skip_2nd=True to skip it."""
        self._firstRound_preMapping(n_parallel=n_parallel)
        if skip_2nd:
            self.adata_spatial.obsm["confidence_premapping2"] = self.adata_spatial.obsm[
                "confidence_premapping1"
            ].copy()
            self.adata_spatial.obs["cell_type_premapping2"] = _np.argmax(
                self.adata_spatial.obsm["confidence_premapping2"],
                axis=1,
            )
        else:
            self._secondRound_preMapping()
        self._premapped = True
        return

    def run_segmentation(
        self,
        n_spots_add_per_step: int = 9,
        n_parallel: int = 100,
        coverage_to_stop: float = 0.8,
        max_iter: int = 200,
        traverse_all_spots: bool = False,
        verbose: bool = True,
        warnings: bool = False,
        print_summary: bool = True,
    ):
        """Segments the spots into single cells. Spots to query are selected
        `self.n_parallel` in a batch. If `traverse_all_spots`, spots that are
        not queried even if they are positively masked will go through query
        in future rounds; otherwise, once spots are positively masked or queried,
        they will not be queried anymore.
        Updates self.filtrations, self.sampleIds_new, self.confidences_new, and
        self.classes_new.
        """
        assert self._premapped, "Must .run_preMapping() first!"
        confident_count = 0
        class_count: dict[int, int] = dict()
        queried_spotIds = set()
        for i_iter in range(max_iter):
            if verbose:
                _tqdm.write(f"Iter {i_iter+1}:")
            available_spots: list[int] = list(
                (
                    set(_np.arange(self.adata_spatial.shape[0]))
                    if traverse_all_spots
                    else set(self.unmasked_spotIds)
                )
                - queried_spotIds
            )
            if len(available_spots) == 0:
                _tqdm.write("All spots queried.")
                break
            idx_centroids: _NDArray[_np.int_] = _np.random.choice(
                a=available_spots,
                size=min(n_parallel, len(available_spots)),
                replace=False,
            )
            queried_spotIds |= set(idx_centroids)
            confs, labels, _ = self._buildFiltration_addSpotsUntilConfident(
                idx_centroids=idx_centroids,
                n_spots_add_per_step=n_spots_add_per_step,
                verbose=warnings,
            )
            confident_count += (confs >= self.threshold_confidence).sum()
            for label in labels:
                if label == -1:
                    continue
                class_count[label] = class_count.get(label, 0) + 1
            if verbose:
                _tqdm.write(
                    f"Cell {idx_centroids[0]}, ... | Confidence: {confs[0]:.3e}, ... | Confident total: {confident_count} | class: {labels[0]}, ..."
                )
                _tqdm.write(f"Classes total: {class_count}")
            coverage = (self.mask_newIds > -1).sum() / len(self.mask_newIds)
            if verbose:
                _tqdm.write(f"Coverage: {coverage*100:.2f}%")
            if coverage >= coverage_to_stop:
                break
        else:
            if warnings:
                _tqdm.write(f"Reaches max_iter {max_iter}!")
        if verbose:
            _tqdm.write("Done.")
        if print_summary:
            _tqdm.write(
                f"""--- Summary ---
Queried {len(queried_spotIds)} spots, of which {confident_count} made up confident single cells.
Classes total: {class_count}
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
        if (not force) and (
            not isinstance(self.cache_singleCellAnnData, _UndefinedType)
        ):
            return self.cache_singleCellAnnData
        sc_X = []
        raw_X = _to_array(self.adata_spatial.X)
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

    def get_type_name_by_id(self, index: int) -> str:
        if index == -1:
            return "Undefined"
        return self.local_classifier._classes[index]

    def get_spatial_classes(
        self,
        return_string: bool = False,
    ) -> _NDArray[_np.int_] | _NDArray[_np.str_]:
        """Get an array of integers, corresponding to class ids of each old sample (spot). -1 for unassigned. Or if `return_string` is `True`, return an array of
        class names, and 'Undefined' for unassigned."""
        res: _NDArray[_np.int_] = _np.zeros(
            shape=(self.adata_spatial.shape[0],), dtype=int
        )
        for i_sample in range(len(res)):
            # First query the filtrations.keys()
            if i_sample in self._filtrations.keys():
                new_id = i_sample
            else:  # Thereafter, query the mask
                new_id = self.mask_newIds[i_sample]
            if new_id == -1:
                res[i_sample] = -1
                continue
            new_class = self._classes_new[new_id]
            res[i_sample] = new_class
        if return_string:
            return _np.array(
                [self.get_type_name_by_id(id_cls) for id_cls in res], dtype=str
            )
        return res

    def run_plotClasses(self, **plot_kwargs):
        """Plot classes of each spot. Classes sorted alphabetically."""
        import seaborn as sns

        spatial_classes: _NDArray[_np.str_] = self.get_spatial_classes(
            return_string=True
        )
        hue_order: _NDArray[_np.str_] = _np.sort(_np.unique(spatial_classes))
        if "Undefined" in hue_order:
            i_undefined: int = _np.where(hue_order == "Undefined")[0][0]
            hue_order = _np.append(_np.delete(hue_order, i_undefined), "Undefined")
        return sns.scatterplot(
            x=self.adata_spatial.obs["x"].values,
            y=self.adata_spatial.obs["y"].values,
            hue=spatial_classes,
            hue_order=hue_order,
            **plot_kwargs,
        )

    def run_plotNewIds(self):
        """Plot last assigned ids."""
        new_ids = self.mask_newIds
        import seaborn as sns

        return sns.scatterplot(
            x=self.adata_spatial.obs["x"].values,
            y=self.adata_spatial.obs["y"].values,
            hue=new_ids.astype(_np.str_),
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
    dist_matrix: _dok_matrix = ckdtree.sparse_distance_matrix(
        other=ckdtree,
        max_distance=radius_local,
        p=2,
        output_type="dok_matrix",
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
