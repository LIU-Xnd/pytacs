# pytacs: Python Topology-Aware Cell-Type Spotting -
#  an improved version of TopACT (https://gitlab.com/kfbenjamin/topact)
#  implementation.

# Author: Liu X., 2024.12

# Novelty so far:
# - Self-discovered prior knowledge;
# - Improved cell shape approximation;
# - Independency from imaging segmentation information;
# - Improved local classifier strategy - higher inferencing accuracy;


__version__ = "1.7.4"

from .data import AnnDataPreparer, downsample_cells, compare_umap
from .classifier import (
    SVM,
    GaussianNaiveBayes,
    QProximityClassifier,
    CosineSimilarityClassifier,
    JaccardClassifier,
)
from .spatial import rw_aggregate, extract_celltypes_full, cluster_spatial_domain
