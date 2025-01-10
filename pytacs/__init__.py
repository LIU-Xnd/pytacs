# pytacs: Python Topological Automatic Cell Segmentation -
#  an improved version of TopACT (https://gitlab.com/kfbenjamin/topact)
#  implementation.

# Author: Liu X., 2024.12

# Updates v1.0.5:
# - [v] Add param standardize_PCs (bool) to QProximityClassifier to scale
#  different PC dimensions for proximity balls to work better.
# - [v] Add param n_spots_add_per_step to SpatialHandler to speed up.

# Future:
# - Add params normalize, log1p, on_PCs, n_PCs to classifier.SVM.
# - Write __repr__ for _LocalClassifier and its child classes.
# - Improve run_getSingleCellAnnData()
#    + Add: Coordinate remapping
#    + Add: Shape smoothing (if a spot is surrounded by spots of the same
#    class, then make it of that class, too)
# - Improve documentations
# - Image based local classifier (CellPose + TopACT)
# - The spatial coordinates by convention should have been saved in `.obsm`
#  because `'x'` and `'y'` are related. But in this tool they are expected to be
#  put separately as columns in `.obs`. This might be a break of convention that
#  needs addressing in the future.

__version__ = "1.0.5"

from .data import AnnDataPreparer
from .classifier import SVM, GaussianNaiveBayes, QProximityClassifier
from .spatial import SpatialHandler
