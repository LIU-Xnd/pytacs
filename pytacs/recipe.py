"""
Integrated recipes for enhanced user friendliness.

Usage:

```
from pytacs import recipe as rc

# Load ref data
ref = rc.read_h5ad('h5ad/snRNA_mouse_demo.h5ad')
rc.prepare_ref(ref)

# Train classifier
clf = rc.prepare_clf(ref, threshold=0.75, filepath_out='clf.dill')
# If saved before, can load
clf = rc.read_clf('clf.dill')

# Load trx data
trx = rc.read_h5ad('h5ad/spRNA_mouse_demo.h5ad)
rc.prepare_trx(trx)

# Bin trx if you wish
trx = rc.prepare_bin(trx, binsize=6)

# Build spatial graph
rc.prepare_graph(trx, radius=10)

# Start annotation
rc.run_annotate(trx, clf, radius=1.5, n_iterations=4)

# Start segmentation
rc.run_segment(trx)

# Get cell-level AnnData
trx_cells = rc.get_cells(trx)
```
"""

def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(message)
    return

from pathlib import Path as _Path
def _mkdirs(path_string: _Path | str):
    path_string: _Path = _Path(path_string)
    path_string.parent.mkdir(parents=True, exist_ok=True)
    return

from scanpy import (
    read_h5ad,
    AnnData as _AnnData,
)

from numpy import (
    ndarray as _ndarray,
)

from .data import (
    _csr_matrix,
    _reinit_index,
    binX as _binX,
)

from .classifier import (
    SVM as _SVM,
)

import dill as _dill

from .spatial import (
    spatial_distances as _spatial_distances,
    rw_aggregate as _rw_aggregate,
    ctrbin_cellseg_parallel as _cellseg,
    aggregate_spots_to_cells_parallel as _aggregate_spots,
    extract_celltypes_full as _ext_ct,
    extract_cell_sizes_full as _ext_cs,
    SpTypeSizeAnnCntMtx as _AnnCnt,
)

def prepare_ref(
    ref: _AnnData,
    cell_type_obs_name: str | None = None,
    filepath_out: str | None = None,
    verbose: bool = True,
) -> None:
    """Check essential fields for ref AnnData and modify them if necessary.
    
    Args:
        cell_type_obs_name (str): the name of obs column that stores cell types.
        If differs from `cell_type`, modify to that (old column remains).

        filepath_out (Path): the filepath to save the modified AnnData. Suffix: h5ad.
    
    Return:
        None. Modifications are taken inplace.
    """
    _log('Check ref.X types.', verbose)
    if isinstance(ref.X, _csr_matrix | _ndarray):
        _log(f'{type(ref.X)=}. Pass.', verbose)
    else:
        _log(f'{type(ref.X)=}. Try to convert to csr.', verbose)
        ref.X = _csr_matrix(ref.X)
    
    _log(f'{ref.X.dtype=}', verbose)
    
    _log(f'Check ref.obs[] cell_type column.', verbose)
    if cell_type_obs_name is not None:
        assert cell_type_obs_name in ref.obs_keys()
        ref.obs['cell_type'] = ref.obs[cell_type_obs_name].copy()
    else:
        assert 'cell_type' in ref.obs_keys()

    _log('Reinit indices.', verbose)
    _reinit_index(ref)
    
    if filepath_out is not None:
        filepath_out: str = str(filepath_out)
        if not filepath_out.endswith('.h5ad'):
            filepath_out += '.h5ad'
        _mkdirs(filepath_out)
        _log(f'Writing to {filepath_out}.')
        ref.write_h5ad(filepath_out, compression='gzip')

    return

def prepare_trx(
    trx: _AnnData,
    spatial_coordinates_obsm_name: str | None = None,
    filepath_out: str | None = None,
    verbose: bool = True,
) -> None:
    """Check essential fields for spatial transcriptomics AnnData and modify them if necessary.
    
    Args:
        spatial_coordinates_obsm_name (str): the name of obsm key that stores
        spatial coordinates (Nx2 Array). If differs from `spatial`, modify to it
        (old remains).

        filepath_out (Path): the filepath to save the modified AnnData. Suffix: h5ad.
    
    Return:
        None. Modifications are taken inplace.
    """
    _log('Check trx.X types.', verbose)
    if isinstance(trx.X, _csr_matrix | _ndarray):
        _log(f'{type(trx.X)=}. Pass.', verbose)
    else:
        _log(f'{type(trx.X)=}. Try to convert to csr.', verbose)
        trx.X = _csr_matrix(trx.X)
    
    _log(f'{trx.X.dtype=}', verbose)
    
    _log(f'Check trx.obsm[] spatial coordinates.', verbose)
    if spatial_coordinates_obsm_name is not None:
        assert spatial_coordinates_obsm_name in trx.obsm_keys()
        trx.obsm['spatial'] = trx.obsm[spatial_coordinates_obsm_name].copy()
    else:
        assert 'spatial' in trx.obsm_keys()
    _log(f'Preview: {trx.obsm['spatial'][:5]=}', verbose)

    _log('Reinit indices.', verbose)
    _reinit_index(trx)
    
    if filepath_out is not None:
        filepath_out: str = str(filepath_out)
        if not filepath_out.endswith('.h5ad'):
            filepath_out += '.h5ad'
        _mkdirs(filepath_out)
        _log(f'Writing to {filepath_out}.', verbose)
        trx.write_h5ad(filepath_out, compression='gzip')

    return

def prepare_bin(
    trx: _AnnData,
    binsize: int = 6,
    filepath_out: str | None = None,
    delete_old: bool = True,
    verbose: bool = True,
) -> _AnnData:
    """
    Bin trx to grids of binsize to enhance signal.
    
    :param trx: prepared trx AnnData.
    :type trx: _AnnData
    :param binsize: size of bin.
    :type binsize: int
    :param delete_old: remove old trx object from memory.
    :type binsize: bool
    :param filepath_out: If given, save resulted trx. Suffix: h5ad.
    :type filepath_out: str | None
    :return: Binned trx.
    :rtype: AnnData
    """
    assert 'spatial' in trx.obsm_keys()
    binned: _AnnData = _binX(trx, binsize=binsize)
    if filepath_out is not None:
        filepath_out = str(filepath_out)
        if not filepath_out.endswith('.h5ad'):
            filepath_out += '.h5ad'
        _mkdirs(filepath_out)
        _log(f'Saving file to {filepath_out}', verbose)
        binned.write_h5ad(filepath_out, compression='gzip')
    if delete_old:
        _log('Releasing memory.', verbose)
        del trx
    return binned

def prepare_graph(
    trx: _AnnData,
    radius: float | int = 8,
    filepath_out: str | None = None,
    verbose: bool = True,
) -> None:
    """
    Prepare spatial graph.
    
    :param trx: trx AnnData.
    :type trx: _AnnData
    :param radius: maximum radius of connection. (Critial for memory cost.)
    :type radius: float | int
    :param filepath_out: If given, save resulted h5ad file. Suffix: h5ad.
    :type filepath_out: str | None
    """
    if filepath_out is not None:
        filepath_out = str(filepath_out)
        if not filepath_out.endswith('.h5ad'):
            filepath_out += '.h5ad'
        _mkdirs(filepath_out)
    _spatial_distances(trx, max_spatial_distance=radius, verbose=verbose)
    if filepath_out is not None:
        _log(f'Save to {filepath_out}', verbose)
        trx.write_h5ad(filepath_out, compression='gzip')
    return

def prepare_clf(
    ref: _AnnData,
    threshold: float = 0.75,
    filepath_out: str | None = None,
    verbose: bool = True,
) -> _SVM:
    """
    Train a classifier using ref.
    
    :param ref: ref AnnData.
    :type ref: _AnnData
    :param threshold: confidence threshold.
    :type threshold: float
    :param filepath_out: If given, save model to file.
    :type filepath_out: str | None
    :return: Trained clf.
    :rtype: SVM
    """
    if filepath_out is not None:
        filepath_out = str(filepath_out)
        _mkdirs(filepath_out)
    clf = _SVM(threshold_confidence=threshold)
    clf.fit(ref)
    if filepath_out is not None:
        _log(f'Save clf to {filepath_out}', verbose)
        with open(filepath_out, 'wb') as f:
            _dill.dump(clf, f)
    return clf

def read_clf(
    filepath: str,
) -> _SVM:
    with open(filepath, 'rb') as f:
        clf = _dill.load(f)
    return clf

def run_annotate(
    trx: _AnnData,
    clf: _SVM,
    radius: float = 1.5,
    n_iterations: int = 4,
    filepath_out: str | None = None,
    verbose: bool = True,
) -> None:
    """
    DeTACH's first part: annotate cell types and sizes.
    
    :param trx: spatial trx AnnData
    :type trx: _AnnData
    :param clf: DeTACH classifier
    :type clf: _SVM
    :param radius: radius of neighborhood
    :type radius: float
    :param n_iterations: number of iterations
    :type n_iterations: int
    """
    assert 'max_spatial_distance' in trx.uns_keys()
    if filepath_out is not None:
        filepath_out = str(filepath_out)
        if not filepath_out.endswith('.h5ad'):
            filepath_out += '.h5ad'
        _mkdirs(filepath_out)
    aggres = _rw_aggregate(
        trx,
        clf,
        max_iter=n_iterations,
        nbhd_radius=radius,
        max_propagation_radius=trx.uns['max_spatial_distance'],
        mode_prune='proportional',
        verbose=verbose,
    )
    trx.obs['cell_type_detach'] = _ext_ct(aggres)
    trx.obs['cell_size_detach'] = _ext_cs(aggres, size_undefined=9)
    if filepath_out is not None:
        _log(f'Writing to {filepath_out}', verbose)
        trx.write_h5ad(filepath_out, compression='gzip')
    return

def run_segment(
    trx: _AnnData,
    filepath_out: str | None = None,
    n_workers: int = 40,
    verbose: bool = True,
) -> None:
    """
    DeTACH's part 2: segment cells.
    
    :param trx: annotated trx.
    :type trx: _AnnData
    """
    if filepath_out is not None:
        filepath_out = str(filepath_out)
        if not filepath_out.endswith('.h5ad'):
            filepath_out += '.h5ad'
        _mkdirs(filepath_out)
    trx.obs['cell_id_detach'] = _cellseg(
        ann_count_matrix=_AnnCnt(
            count_matrix=_csr_matrix(trx.X),
            spatial_distances=trx.obsp['spatial_distances'],
            cell_types=trx.obs['cell_type_detach'].values,
            cell_sizes=trx.obs['cell_size_detach'].values,
        ),
        spatial_coordinates=trx.obsm['spatial'],
        attitude_to_undefined='exclusive',
        n_workers=n_workers,
        verbose=verbose,
    )
    _reinit_index(trx, '__tmp_indices')
    del trx.obs['__tmp_indices']
    if filepath_out is not None:
        _log(f'Writing to {filepath_out}', verbose)
        trx.write_h5ad(filepath_out, compression='gzip')
    return

def get_cells(
    trx: _AnnData,
    filepath_out: str | None = None,
    n_workers: int = 20,
    verbose: bool = True,
) -> _AnnData:
    if filepath_out is not None:
        filepath_out = str(filepath_out)
        if not filepath_out.endswith('.h5ad'):
            filepath_out += '.h5ad'
        _mkdirs(filepath_out)
    ret: _AnnData = _aggregate_spots(trx, 'cell_id_detach', 'cell_type_detach', n_workers=n_workers, verbose=verbose)
    if filepath_out is not None:
        _log(f'Writing to {filepath_out}', verbose)
        ret.write_h5ad(filepath_out, compression='gzip')
    return ret

