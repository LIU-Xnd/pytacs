# pytacs: Python Topological Automatic Cell Segmentation -
#  an improved version of TopACT (https://gitlab.com/kfbenjamin/topact)
#  implementation.

# Author: Liu X., 2024.12

# Updates v1.0.4:
# - To address effect of outliers from snRNA-seq data,
#  we propose the "q-proximity confidence metric".
#  The proximity radius R(i,q) of class i
#  and of quantile q (0 < q < 1) is defined as the minimal r where the ball
#  centered at class i's mean vector \mu_i with radius r, Ball(\mu_i, r),
#  contains a proportion of q points in class i.
#  And the q-proximity of point x to
#  class i, prox_q(x,i), is defined as the cardinality of the intersection
#  of Ball(x, R(i,q)) with the set of all points in class i, X_i, that is,
#  cardinality(Ball(x, R(i,q)) & X_i), divided by q * cardinality(X_i),
#  i.e., prox_q(x,i) := cardinality(Ball(x, R(i,q)) & X_i) / (q*cardinality(X_i)).
#  To avoid potential cases where prox_q > 1, we use capped prox_q(x,i) :=
#  min(1, prox_q(x,i)) as the confidence metric.
#   + This metric does not assume any prior distribution of data and can
#    robustly measure the confidence of identity of cells.
#   + The corresponding classifier is call QProxmityClassifier.

# Future:
# - Add param standardize_PCs (bool) to QProximityClassifier to scale
#  different PC dimensions for proximity balls to work better.
# - Add param n_cells_add_per_step to SpatialHandler to speed up.
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

__version__ = "1.0.4"

from .data import AnnDataPreparer
from .classifier import SVM, GaussianNaiveBayes, QProximityClassifier
from .spatial import SpatialHandler
