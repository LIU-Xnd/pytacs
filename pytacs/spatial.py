from scanpy import AnnData as _AnnData
import numpy as _np
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.spatial import distance_matrix as _distance_matrix
from .classifier import LocalClassifier as _LocalClassifier
from .utils import radial_basis_function as _rbf

from .utils import _UNDEFINED

class SpatialHandler:
    """A spatial handler to produce filtrations, integrate spots into single cells,
     and estimate their confidences."""
    def __init__(self,
                adata_spatial: _AnnData,
                local_classifier: _LocalClassifier,
                threshold_adjacent: float = 1.5,
                max_spots_per_cell: int = 81,
                scale_rbf: float = 1.):
        self.adata_spatial = adata_spatial
        self.threshold_adjacent = threshold_adjacent
        self.local_classifier = local_classifier
        
        self.max_spots_per_cell = max_spots_per_cell
        self.scale_rbf = scale_rbf
        
        self.__filtrations = dict()
        # dict {idx_centroid: [idx_centroid, idx_spot_level1, idx_spot_level2, ...]}
        
        self.__mask_newIds = _np.full((self.adata_spatial.X.shape[0],), fill_value=-1, dtype=int)
        # mask on each old sample. -1 for not assigned; otherwise the new id.
        
        self.__classes_new = dict()
        self.__confidences_new = dict()
        
        self.cache_distance_matrix: _np.ndarray = _UNDEFINED
        self.cache_singleCellAnnData: _AnnData = _UNDEFINED
        return None
    
    @property
    def threshold_confidence(self) -> float:
        return self.local_classifier.threshold_confidence
    @property
    def has_negative_control(self) -> bool:
        return self.local_classifier.has_negative_control
    @property
    def filtrations(self) -> dict:
        copy_ = dict()
        for k, v in self.__filtrations.items():
            copy_[k] = v.copy()
        return copy_
    @property
    def mask_newIds(self) -> _np.ndarray:
        return self.__mask_newIds.copy()
    @property
    def masked_spotIds(self) -> _np.ndarray:
        """Already positively masked spot ids."""
        return _np.where(self.mask_newIds > -1)[0]
    @property
    def unmasked_spotIds(self) -> _np.ndarray:
        return _np.where(self.mask_newIds == -1)[0]
    @property
    def sampleIds_new(self) -> _np.ndarray:
        return _np.sort(_np.array(list(set(list(_np.unique(self.mask_newIds))) - {-1})))
    @property
    def classes_new(self) -> dict:
        return self.__classes_new.copy()
    @property
    def confidences_new(self) -> dict:
        return self.__confidences_new.copy()

    def __repr__(self) -> str:
        return f"""--- Spatial Handler (pytacs) ---
- adata_spatial: {self.adata_spatial}
- threshold_adjacent: {self.threshold_adjacent}
- local_classifier: {self.local_classifier}
    + threshold_confidence: {self.threshold_confidence}
    + has_negative_control: {self.has_negative_control}
- max_spots_per_cell: {self.max_spots_per_cell}
- scale_rbf: {self.scale_rbf}
- filtrations: {len(self.filtrations)} fitted
- single-cell segmentation:
    + new samples: {len(self.sampleIds_new)}
    + AnnData: {self.cache_singleCellAnnData}
--- --- --- --- --- ---
"""
        
    def clear_cache(self):
        self.cache_distance_matrix = _UNDEFINED
        self.cache_singleCellAnnData = _UNDEFINED
        return None
        
    def _compute_distance_matrix(self):
        points = self.adata_spatial.obs[['x', 'y']].values
        self.cache_distance_matrix = _distance_matrix(points, points)
        return None
        
    def _find_adjacentOfOneSpot_spotIds(
        self,
        idx_this_spot: int
    ) -> _np.ndarray:
        """Find all adjacent spots, including self."""
        if self.cache_distance_matrix is _UNDEFINED:
            self._compute_distance_matrix()
        distances = self.cache_distance_matrix[idx_this_spot,:]
        idxs_adjacent = _np.where(distances <= self.threshold_adjacent)[0]
        return idxs_adjacent

    def _find_adjacentOfManySpots_spotIds(
        self,
        filtration: list[int]
    ) -> _np.ndarray:
        """Find all adjacent spots of a list of indices of spots (called filtration),
         excluding selves and already positively masked spots."""
        assert len(filtration) > 0
        adj_spots = []
        for spot in filtration:
            adj_spots += list(self._find_adjacentOfOneSpot_spotIds(spot))
        return _np.array(list(
            (set(adj_spots) - set(filtration)) - set(list(self.masked_spotIds))
        ))

    def _buildFiltration_addOneSpot(
        self,
        idx_centroid: int,
        verbose: bool = True,
    ) -> int:
        """Randomly (with rbf probs) adds one spot for the cell
         centered at idx_centroid.

        Filtration list includes idx_centroid itself.
        
        Update the self.filtrations,
         and returns the added spot index (-1 for not added)."""
        loc_centroid = self.adata_spatial.obs[['x','y']].values[idx_centroid,:]
        self.__filtrations[idx_centroid] = self.__filtrations.get(idx_centroid, [idx_centroid])
        if len(self.__filtrations[idx_centroid]) >= self.max_spots_per_cell:
            if verbose:
                print(f"Warning: reaches max_spots_per_cell {self.max_spots_per_cell}")
            return -1
        # Find adjacent candidate spots
        idxs_adjacent = self._find_adjacentOfManySpots_spotIds(self.__filtrations[idx_centroid])
        if len(idxs_adjacent) == 0:
            if verbose:
                print(f"Warning: no adjacent spots found! Check threshold_adjacent {self.threshold_adjacent}")
            return -1
        # Calculate the probs
        probs = []
        for idx in idxs_adjacent:
            loc_adj = self.adata_spatial.obs[['x','y']].values[idx,:]
            probs.append(_rbf(loc_adj, loc_centroid, scale=self.scale_rbf))
        probs = _np.array(probs)
        probs /= _np.sum(probs)
        # Select one randomly
        idx_selected = _np.random.choice(idxs_adjacent, p=probs)
        # Update the filtration
        self.__filtrations[idx_centroid].append(idx_selected)
        return idx_selected

    def _aggregate_spots_given_filtration(
        self,
        filtration: list[int],
    ) -> _np.ndarray:
        """Returns a 1d-array of counts of genes."""
        idxs_filtration = _np.array(filtration)
        return self.adata_spatial.X.toarray()[idxs_filtration,:].sum(axis=0)

    def _compute_confidence_of_level(
        self,
        idx_centroid: int,
        level: int = None,
    ) -> _np.ndarray:
        """Calculate confidence of filtrations[idx_centroid][:level+1]"""
        self.__filtrations[idx_centroid] = self.__filtrations.get(idx_centroid, [idx_centroid])
        if level is None:
            filtration_this_level = self.filtrations[idx_centroid][:]
        else:
            filtration_this_level = self.filtrations[idx_centroid][:level+1]
        probas = self.local_classifier.predict_proba(
            X=_np.array([list(self._aggregate_spots_given_filtration(filtration_this_level))]),
            genes=self.adata_spatial.var.index,
        )[0,:]
        # Remove the NegativeControl proba
        # print(probas)
        if self.has_negative_control:
            probas = probas[:-1]
        return probas
    
    def _buildFiltration_addSpotsUntilConfident(
        self,
        idx_centroid: int,
        verbose: bool = True,
    ) -> tuple[float, int, int]:
        """Find many spots centered at idx_centroid that are confidently in a class.
        
        Update the self.filtrations, update the self.mask_newIds, self.confidences_new,
         self.classes_new,
         and returns the (confidence, class_id, new_sampleId).
         If reaches max_spots_per_cell and still not confident, returns the
         (confidence, -1, and idx_centroid)."""
        label: int = _UNDEFINED
        confidence: float = _UNDEFINED
        for _ in range(self.max_spots_per_cell):
            probas = self._compute_confidence_of_level(idx_centroid, level=None)
            label = _np.argmax(probas)
            confidence = probas[label]
            if confidence >= self.threshold_confidence:
                self.__mask_newIds[_np.array(self.filtrations[idx_centroid])] = idx_centroid
                self.__classes_new[idx_centroid] = label
                self.__confidences_new[idx_centroid] = confidence
                break
            idx_added = self._buildFiltration_addOneSpot(idx_centroid, verbose=verbose)
            if idx_added == -1:
                label = -1
                break
        else:
            label = -1
        if label == -1:
            # Clear filtrations that are not confident
            del self.__filtrations[idx_centroid]
        return (confidence, label, idx_centroid)

    # Need to be careful with input idx_centroid - you don't want to
    # input idx_centroid that is already positively masked.

    def run_segmentation(
        self,
        coverage_to_stop: float = 0.8,
        max_iter: int = 200,
        verbose: bool = True,
        warnings: bool = False,
        print_summary: bool = True,
    ):
        """Segments the spots into single cells. Seed spots are selected randomly and sequentially.
        Updates self.sampleIds_new, self.confidences_new, self.classes_new."""
        confident_count = 0
        class_count = dict()
        for i_iter in range(max_iter):
            if verbose and i_iter % 5 == 0:
                print(f'Iteration {i_iter+1}:')
            available_spots = self.unmasked_spotIds
            if len(available_spots) == 0:
                print("All spots queried. Done.")
                return None
            ix_centroid = _np.random.choice(available_spots)
            if verbose and i_iter % 5 == 0:
                print(f"Querying spot {ix_centroid} ...")
            conf, label, _ = self._buildFiltration_addSpotsUntilConfident(ix_centroid, verbose=warnings)
            if conf >= self.threshold_confidence:
                confident_count += 1
                class_count[label] = class_count.get(label, 0) + 1
            if verbose and i_iter % 5 == 0:
                print(f"Spot {ix_centroid} | confidence: {conf*100:.3f}% | confident total: {confident_count} | class: {label}")
                print(f"Classes total: {class_count}")
            coverage = (self.mask_newIds>-1).sum() / len(self.mask_newIds)
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
            print(f"""--- Summary ---
Queried {max_iter} spots (with replacement), of which {confident_count} made up confident single cells.
Classes total: {class_count}
Coverage: {coverage*100:.2f}%
--- --- --- --- ---
""")
        return None
                

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
            sc_X.append(list(raw_X[_np.array(self.filtrations[ix_new]),:].sum(axis=0)))
        sc_adata = _AnnData(
            X=_csr_matrix(sc_X),
            obs=self.adata_spatial.obs.copy().iloc[self.sampleIds_new],
            var=self.adata_spatial.var.copy()
        )
        sc_adata.obs['confidence'] = 0.
        for ix_new in self.sampleIds_new:
            sc_adata.obs.loc[str(ix_new), 'confidence'] = self.confidences_new[ix_new]
        # Save cache
        if cache:
            self.cache_singleCellAnnData = sc_adata
        return sc_adata

    def run_plotClasses(self):
        pass

    def run_plotNewIds(self):
        pass
        