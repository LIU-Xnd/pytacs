# pytacs: Python Topological Automatic Cell Segmentation -
#  an improved version of TopACT (https://gitlab.com/kfbenjamin/topact)
#  implementation.

# Author: Liu X., 2024.12

# Updates v1.1.0:
# Plan:
# - Try to merge imaging data for a
# image-based local classifier (CellPose + TopACT)
#   + Find ways to generate confidence from imaging
#    Maybe the highest IOU?
#   + Estimate cell size (n_spots) from imaging
#   + Python implemented CellPose API
#   + Strategy of adding spots: learning from imaging

# Future:

# - Improve run_getSingleCellAnnData()
#    + Add: Coordinate remapping
#    + Add: Shape smoothing (if a spot is surrounded by spots of the same
#    class, then make it of that class, too)
# - Improve documentations
# - The spatial coordinates by convention should have been saved in `.obsm`
#  because `'x'` and `'y'` are related. But in this tool they are expected to be
#  put separately as columns in `.obs`. This might be a break of convention that
#  needs addressing in the future.
# - Add params normalize, log1p, on_PCs, n_PCs to classifier.SVM.
# - Write __repr__ for _LocalClassifier and its child classes.

__version__ = "1.1.0"

from .data import AnnDataPreparer
from .classifier import SVM, GaussianNaiveBayes, QProximityClassifier
from .spatial import SpatialHandler
