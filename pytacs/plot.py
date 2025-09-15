import numpy as _np
from dataclasses import dataclass as _dataclass
from scipy.spatial.ckdtree import cKDTree as _cKDTree
from tqdm import tqdm as _tqdm
import seaborn as _sns
import matplotlib.pyplot as _plt
from .types import (
    _NumberType,
    _Nx2ArrayType,
    _1DArrayType,
    _csr_matrix,
    _NDArray,
)

rcParams_setupStr = '''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme('paper', 'white')
plt.rcParams['figure.dpi'] = 900.0
plt.rcParams['scatter.edgecolors'] = 'none'
plt.rcParams['figure.figsize'] = [4,3]
plt.rcParams['savefig.format'] = 'jpg'
plt.rcParams['savefig.format'] = 'jpg'
'''


@_dataclass
class SpAnnPoints:
    """
    Spatial Annotated Points.

    Attrs:
        coordinates (_Nx2ArrayType): a 2d-array of x, y of locations of points in space.

        masks (_1DArrayType): a 1d-array of labels of points. Points of different labels are
        considered to be of different objects.
    """

    coordinates: _Nx2ArrayType
    masks: _1DArrayType

    def __post_init__(self):
        assert len(self.masks) == self.coordinates.shape[0]
        assert self.coordinates.shape[1] == 2
        self._n_points: int = self.coordinates.shape[0]
        self._xrange: tuple[_NumberType, _NumberType] = (
            self.coordinates[:, 0].min(),
            self.coordinates[:, 0].max(),
        )
        self._yrange: tuple[_NumberType, _NumberType] = (
            self.coordinates[:, 1].min(),
            self.coordinates[:, 1].max(),
        )
        return


def get_boundaries(
    ann_points: SpAnnPoints,
    nbhd_radius: _NumberType | None = None,
    verbose: bool = True,
) -> _NDArray[_np.bool_]:
    """
    Get boundaries of many objects.

    Args:
        ann_points (SpAnnPoints): points in 2d-space with annotations of their belonging objects.

        nbhd_radius (_NumberType | None, optional): the range within which points are considered as
        neighbors. If neighbors of a point include different object annotations, this point
        is regarded as a boundary point. If `None`, roughly estimates the radius by calculating
        the density of the points.

    Return:
        ndarray: 1d-array of booleans with shape corresponding to `ann_points`. If a point is
        boundary point, its corresponding entry is `True`, otherwise `False`.
    """
    if nbhd_radius is None:
        area: _NumberType = (ann_points._xrange[1] - ann_points._xrange[0]) * (
            ann_points._yrange[1] - ann_points._yrange[0]
        )
        assert area > 0
        # Area occupied by a point on average
        inv_density: _NumberType = area / ann_points._n_points
        # Estimate nbhd_radius according to S=pi*r^2 -> r = sqrt(S/pi)
        nbhd_radius: _NumberType = 2 * _np.sqrt(inv_density / _np.pi)
        # Increase it a bit
        nbhd_radius *= 1.05
    assert isinstance(nbhd_radius, _NumberType)

    # Construct spatial graph
    ckdtree: _cKDTree = _cKDTree(
        data=ann_points.coordinates,
    )
    dist_matrix: _csr_matrix = _csr_matrix(
        ckdtree.sparse_distance_matrix(
            other=ckdtree,
            max_distance=nbhd_radius,
            p=2,
        )
    )
    assert isinstance(dist_matrix, _csr_matrix)
    boundary_masks: _1DArrayType = _np.zeros(
        shape=(ann_points._n_points,),
        dtype=bool,
    )
    # Just implement a for-loop so far
    if verbose:
        _itor = _tqdm(
            range(dist_matrix.shape[0]),
            desc="Finding boundaries",
            ncols=60,
        )
    else:
        _itor = range(dist_matrix.shape[0])
    for i_point in _itor:
        nbors: _1DArrayType = dist_matrix.getrow(i_point).nonzero()[1]
        if len(nbors) == 0:
            continue
        masks_nbors_unique: _1DArrayType = _np.unique(ann_points.masks[nbors])
        if len(masks_nbors_unique) > 1:
            boundary_masks[i_point] = True
            continue
        if masks_nbors_unique[0] != ann_points.masks[i_point]:
            boundary_masks[i_point] = True
            continue
    return boundary_masks


def plot_boundaries(
    coordinates: _Nx2ArrayType,
    boundaries_flags: _1DArrayType | None = None,
):
    """Plot boundary points.

    Args:
        coordinates (_Nx2ArrayType): coordinates of points. 1) If `boundaries_flags` is `None`,
        then this parameter should be only boundary points; 2) otherwise, this should be
        all raw points.

        boundaries_flags (_1DArrayType): boolean flags of boundary points if given, that is,
        `coordinates[boundaries_flags, :]` are boundary points.
    """
    if boundaries_flags is not None:
        coordinates = coordinates[boundaries_flags, :]
    return _sns.scatterplot(
        x=coordinates[:, 0],
        y=coordinates[:, 1],
        s=1,
    )


def plot_boundaries_on_grids(
    coordinates: _Nx2ArrayType,
    boundaries_flags: _1DArrayType,
    grid_density: int = 1000,
):
    """Plot boundary points (on a gridmesh).

    Args:
        coordinates (_Nx2ArrayType): coordinates of points. This should be
        all raw points for reference of grid anchors.

        boundaries_flags (_1DArrayType): boolean flags of boundary points if given, that is,
        `coordinates[boundaries_flags, :]` are boundary points.

        grid_density (int, optional): number of grids along the longest axis.
    """
    assert isinstance(boundaries_flags, _np.ndarray)
    assert grid_density >= 1
    if grid_density > 3000:
        _tqdm.write(f"Warning: {grid_density=} > 3000 might be too large for RAM!")
    xrange: tuple[_NumberType, _NumberType] = (
        coordinates[:, 0].min(),
        coordinates[:, 0].max(),
    )
    yrange: tuple[_NumberType, _NumberType] = (
        coordinates[:, 1].min(),
        coordinates[:, 1].max(),
    )
    xlen: _NumberType = xrange[1] - xrange[0]
    ylen: _NumberType = yrange[1] - yrange[0]
    assert xlen > 0
    assert ylen > 0
    if xlen >= ylen:
        nx: int = grid_density
        ny: int = max(1, int(_np.round(nx * ylen / xlen)))
    else:
        ny: int = grid_density
        nx: int = int(_np.round(ny * xlen / ylen))
    xs: _1DArrayType = _np.linspace(
        start=xrange[0],
        stop=xrange[1],
        num=nx,
        endpoint=True,
        dtype=float,
    )
    ys: _1DArrayType = _np.linspace(
        start=yrange[0],
        stop=yrange[1],
        num=ny,
        endpoint=True,
        dtype=float,
    )
    from itertools import product

    anchor_points: _Nx2ArrayType = _np.array(
        list(
            product(
                xs,
                ys,
            )
        )
    )
    del xs
    del ys
    # Infer radius
    area: _NumberType = xlen * ylen
    assert area > 0
    # Area occupied by a point on average
    inv_density: _NumberType = area / coordinates.shape[0]
    # Estimate nbhd_radius according to S=pi*r^2 -> r = sqrt(S/pi)
    nbhd_radius: _NumberType = 2 * _np.sqrt(inv_density / _np.pi)
    # Increase it a bit
    nbhd_radius *= 1.2

    ckdtree_anchors = _cKDTree(
        data=anchor_points,
    )
    ckdtree_points = _cKDTree(
        data=coordinates,
    )
    dist_matrix: _csr_matrix = _csr_matrix(
        ckdtree_anchors.sparse_distance_matrix(
            other=ckdtree_points,
            max_distance=nbhd_radius,
            p=2,
        )
    )
    anchor_masks: _1DArrayType = _np.zeros(
        shape=(anchor_points.shape[0],),
        dtype=_np.bool_,
    )
    for i_anchor in _tqdm(
        range(anchor_points.shape[0]),
        desc="Plotting anchors..",
        ncols=60,
    ):
        ix_nonzeros: _1DArrayType = dist_matrix.getrow(i_anchor).nonzero()[1]
        if len(ix_nonzeros) == 0:
            continue
        entries_nonzeros: _1DArrayType = (
            dist_matrix[i_anchor, ix_nonzeros].toarray().reshape(-1)
        )
        ix_min: int = ix_nonzeros[_np.argmin(entries_nonzeros)]
        is_bd: bool = boundaries_flags[ix_min]
        if is_bd:
            anchor_masks[i_anchor] = is_bd

    return _sns.scatterplot(
        x=anchor_points[anchor_masks, 0],
        y=anchor_points[anchor_masks, 1],
        s=1,
    )


def pie(cell_types: _np.ndarray, title: str = 'Cell Type Proportions', show: bool = True):
    from collections import Counter

    cell_types = dict(Counter(cell_types))
    labels = list(cell_types.keys())
    counts = list(cell_types.values())
    buff = sorted(zip(labels, counts))
    labels = [x[0] for x in buff]
    counts = [x[1] for x in buff]
    _plt.pie(counts, labels=labels, autopct='%1.1f%%')
    _plt.title(title)
    _plt.axis('equal')
    if show:
        _plt.show()
        return
    else:
        return _plt.gcf()

def calculate_proportions(
    cell_types: _np.ndarray,
) -> tuple[list[str], list[float]]:
    """
    Calculate the proportion of each unique label in a 1D array of cell types.

    Parameters
    ----------
    cell_types : np.ndarray
        A 1D numpy array (or array-like) containing categorical cell type labels (e.g., strings).
        Repeated entries indicate higher counts of that cell type.

    Returns
    -------
    labels : list of str
        Sorted list of unique cell type labels.

    proportions : list of float
        Corresponding proportions of each label, in the same order as `labels`. 
        Each value is normalized to sum to 1.

    Example
    -------
    >>> calculate_proportions(np.array(['T', 'B', 'T', 'NK']))
    (['B', 'NK', 'T'], [0.25, 0.25, 0.5])
    """

    from collections import Counter
    from typing import NamedTuple
    class CellTypeProportions(NamedTuple):
        labels: list[str]
        proportions: list[float]

    cell_types = dict(Counter(cell_types))
    labels = list(cell_types.keys())
    counts = list(cell_types.values())
    buff = sorted(zip(labels, counts))
    labels = [x[0] for x in buff]
    counts = [x[1] for x in buff]
    n_total = sum(counts)
    props = [c/n_total for c in counts]
    return CellTypeProportions(
        labels=labels,
        proportions=props,
    )

def plot_stacked_barplot(
        proportions,
        sample_labels=None,
        type_labels=None,
        figsize=(8, 5),
        show=True,
):
    """
    Plot a stacked barplot from a matrix of proportions.

    Parameters
    ----------
    proportions : array-like of shape (n_samples, n_types)
        A 2D array where each row corresponds to a sample, and each column corresponds to 
        a proportion of a specific type (e.g., cell type proportions).

    sample_labels : list of str, optional
        Labels for each sample (x-axis). If None, samples will be labeled as "Sample 1", "Sample 2", etc.

    type_labels : list of str, optional
        Labels for each type (legend). If None, types will be labeled as "Type 1", "Type 2", etc.

    figsize : tuple, optional (default: (8, 5))
        Size of the matplotlib figure.

    Returns
    -------
    None
        The function displays a matplotlib stacked bar plot.
    """
    proportions = _np.asarray(proportions)
    n_samples, n_types = proportions.shape

    if sample_labels is None:
        sample_labels = [f'Sample {i+1}' for i in range(n_samples)]

    if type_labels is None:
        type_labels = [f'Type {i+1}' for i in range(n_types)]

    x = _np.arange(n_samples)
    bottom = _np.zeros(n_samples)

    fig, ax = _plt.subplots(figsize=figsize)

    for i in range(n_types):
        ax.bar(x, proportions[:, i], bottom=bottom, label=type_labels[i])
        bottom += proportions[:, i]

    ax.set_xticks(x)
    ax.set_xticklabels(sample_labels, rotation=45, ha='right')
    ax.legend(title='Type', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    if show:
        _plt.show()
    else:
        return _plt.gcf()

def plot_stacked_barplot_circular(
        proportions,
        sample_labels=None,
        type_labels=None,
        figsize=(5, 5),
        show=True,
):
    """
    Plot a stacked circular barplot from a matrix of proportions.

    Parameters
    ----------
    proportions : array-like of shape (n_samples, n_types)
        A 2D array where each row corresponds to a sample, and each column corresponds to 
        a proportion of a specific type (e.g., cell type proportions).

    sample_labels : list of str, optional
        Labels for each sample (x-axis). If None, samples will be labeled as "Sample 1", "Sample 2", etc.

    type_labels : list of str, optional
        Labels for each type (legend). If None, types will be labeled as "Type 1", "Type 2", etc.

    figsize : tuple, optional (default: (5, 5))
        Size of the matplotlib figure.

    Returns
    -------
    None
        The function displays a matplotlib stacked bar plot.
    """
    proportions = _np.asarray(proportions)
    n_samples, n_types = proportions.shape

    if sample_labels is None:
        sample_labels = [f'Sample {i+1}' for i in range(n_samples)]

    if type_labels is None:
        type_labels = [f'Type {i+1}' for i in range(n_types)]

    x = _np.linspace(0, 2*_np.pi, n_samples, endpoint=False)
    bottom = _np.zeros(n_samples)

    fig, ax = _plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)

    for i in range(n_types):
        ax.bar(x, proportions[:, i], bottom=bottom, width=2*_np.pi/n_samples, edgecolor="white", linewidth=0.5, label=type_labels[i])
        bottom += proportions[:, i]

    ax.set_xticks(x)
    ax.set_xticklabels(sample_labels)
    ax.legend(title='Type', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    if show:
        _plt.show()
    else:
        return _plt.gcf()


def clr_transform(x):
    x = _np.array(x)
    # Multiplicative replacement if zeros exist (optional step)
    if _np.any(x == 0):
        x = _np.where(x == 0, 1e-6, x)
    
    geometric_mean = _np.exp(_np.mean(_np.log(x)))
    return _np.log(x / geometric_mean)


_RGBType = tuple[int, int, int] | str # e.g., '1fff2b'
def color_cells(
    spannpoints: SpAnnPoints,
    n_colors: int | list[_RGBType] = 6,
):
    """
    Color cells for visualization.
    """
    pass
    
    # TODO