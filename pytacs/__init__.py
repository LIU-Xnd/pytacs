"""pytacs: Python Topology-Aware Cell Segmentation"""

__author__ = "Liu, Xindong"
__version__ = "2025.9.10"  # alpha

from .utils import (
    chunk_spatial,
    reinit_index,
    read_from_csv,
    write_to_csv,
)
from .data import (
    AnnDataPreparer,
    binX,
    annotate_mt,
    annotate_ribosomal,
    merge_gene_version,
    scale_genes,
    downsample_cells,
    compare_umap,
    sort_by_coords,
)
from .classifier import (
    SVM,
    # GaussianNaiveBayes,
    # QProximityClassifier,
    # CosineSimilarityClassifier,
    # JaccardClassifier,
)
from .spatial import (
    rw_aggregate,
    rw_aggregate_sequential,
    extract_celltypes_full,
    extract_cell_sizes_full,
    cluster_spatial_domain,
    spatial_distances,
    spatial_distances_sequential,
    spatial_distances_sequential_lossless,
    spatial_distances_knn,
    spatial_distances_knn_sequential,
    spatial_connectivities_knn,
    combined_connectivities,
    SpatialTypeAnnCntMtx,
    celltype_refined_bin,
    ctrbin_cellseg,
    ctrbin_cellseg_parallel,
    SpTypeSizeAnnCntMtx,
    aggregate_spots_to_cells,
    aggregate_spots_to_cells_parallel,
    NucleiMasks,
    vonoroi_indices,
    align_coords,
    transfer_label,
)
from .plot import (
    pie,
    plot_stacked_barplot,
    plot_stacked_barplot_circular,
    SpAnnPoints,
    get_boundaries,
    plot_boundaries,
    plot_boundaries_on_grids,
)

# TODO: Add recipe module for user-friendliness
