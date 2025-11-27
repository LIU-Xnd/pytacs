from argparse import ArgumentParser as AP
from pathlib import Path
from . import recipe as rc
def _dedent(s: str) -> str:
    return s.strip().replace(' '*4, '').replace('\n', ' ')

parser = AP(
    description=_dedent("""DeTACH: Image-free Cell Annotator and Segmentor
        for high-resolution spatial transcriptomics."""),
)
subparsers = parser.add_subparsers(
    title='subcommands',
    dest='command',
    required=True,
)
# Check file integrity and prepare
p_prep = subparsers.add_parser(
    'prep',
    help=_dedent("""Check integrity and prepare input files,
        i.e., single-cell RNA reference (ref) and high-resolution spatial transcriptomics
        (trx). If everything is satisfied, a classifier is trained and saved, and
        the trx is also updated with spatial distance matrix. Otherwise, warning is
        raised."""),
)
p_prep.add_argument(
    '-f', '--ref',
    type=str,
    required=True,
    help=_dedent("""Reference single-cell RNA-seq h5ad filepath.""")
)
p_prep.add_argument(
    '--cell-type-obs-name', dest='obsname',
    required=False,
    type=str,
    default='cell_type',
    help=_dedent("""Column name of cell type in ref.obs. Default: cell_type."""),
)
p_prep.add_argument(
    '-x', '--trx',
    required=True,
    help=_dedent("""High-res spatial trx h5ad filepath.""")
)
p_prep.add_argument(
    '--coordinates-obsm-name', dest='obsmname',
    required=False,
    type=str,
    default='spatial',
    help=_dedent("""Key name of coordinates in trx.obsm. Default: spatial."""),
)
p_prep.add_argument(
    '-o', '--outdir',
    required=True,
    help=_dedent("""directory to save output files, i.e.,
        prepared trx file (trx.h5ad) and trained clf (clf.dill)."""),
)
p_prep.add_argument(
    '-b', '--binsize',
    required=False,
    type=int,
    default=0,
    help=_dedent("""If given, bin trx into grids of this size, and overwrite
        the output trx file (trx.h5ad). Note that the spatial unit also changes,
        e.g., (6.0, 8.0) -> (3, 4) if binsize=2. Integers only.
        """),
)
p_prep.add_argument(
    '-d', '--distance',
    required=False,
    type=float,
    default=8.0,
    help=_dedent("""Maximum distance for building spatial distance matrix.
        Note: if binned, uses the spatial unit after binning. Default: 8."""),
)
p_prep.add_argument(
    '-t', '--threshold',
    type=float,
    required=False,
    default=0.75,
    help=_dedent("""Confidence threshold for training clf. Default: 0.75"""),
)
def prep(a):
    assert Path(a.ref).exists()
    assert Path(a.trx).exists()
    rc._mkdirs(a.outdir)
    print('Check ref file ..')
    ref = rc.read_h5ad(a.ref)
    rc.prepare_ref(ref, a.obsname)
    print('Train clf ..')
    rc.prepare_clf(ref, threshold=a.threshold, filepath_out=Path(a.outdir)/'clf.dill')
    del ref
    print('Check trx file ..')
    trx = rc.read_h5ad(a.trx)
    rc.prepare_trx(trx, a.obsmname)
    if a.binsize>0:
        trx = rc.prepare_bin(trx, binsize=a.binsize)
    rc.prepare_graph(trx, radius=a.distance, filepath_out=Path(a.outdir)/'trx.h5ad')
    return
p_prep.set_defaults(func=prep)

# Run annotation and segmentation
p_run = subparsers.add_parser(
    'run',
    help=_dedent("""Annotate and segment trx based on trained clf."""),
)
p_run.add_argument(
    '-c', '--clf',
    type=str,
    required=True,
    help=_dedent("""Filepath to classifier trained on ref.""")
)
p_run.add_argument(
    '-x', '--trx',
    required=True,
    help=_dedent("""Prepared high-res spatial trx h5ad filepath.""")
)
p_run.add_argument(
    '-o', '--outdir',
    required=True,
    help=_dedent("""directory to save output files, i.e.,
        annotated trx file (trx.h5ad) and segmented cell-level trx file (cells.h5ad)."""),
)
p_run.add_argument(
    '-r', '--radius',
    required=False,
    type=float,
    default=1.5,
    help=_dedent("""Neighborhood radius.
        Note: if binned, uses the spatial unit after binning. Default: 1.5."""),
)
p_run.add_argument(
    '-t', '--iterations', dest='iterations',
    type=int,
    required=False,
    default=4,
    help=_dedent("""Maximum iterations. Default: 4"""),
)
def run(a):
    assert Path(a.clf).exists()
    assert Path(a.trx).exists()
    rc._mkdirs(a.outdir)
    print('Load files ..')
    clf = rc.read_clf(a.clf)
    trx = rc.read_h5ad(a.trx)
    rc.run_annotate(trx, clf, radius=a.radius, n_iterations=a.iterations)
    rc.run_segment(trx, filepath_out=Path(a.outdir)/'trx.h5ad')
    rc.get_cells(trx, filepath_out=Path(a.outdir)/'cells.h5ad')
    return
p_run.set_defaults(func=run)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)
