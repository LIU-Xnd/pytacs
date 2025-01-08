# pytacs: Python Topological Automatic Cell Segmentation -
#  an improved version of TopACT (https://gitlab.com/kfbenjamin/topact)
#  implementation.

# Author: Liu X., 2024

# Updates v1.0.3:
# - GaussianNaiveBayes has many versions in terms of
#  probs calculation (relative probs, two-tail multiplied probs, and
#  this version's average probs). The former two versions of probs
#  are not stable when
#  all the many genes (features) are used and when there are dropouts.
#  Because one single outlier feature (e.g. a dropout) can cause the
#  resulting two-tail (cumulative) prob to be nearly zero. And so will
#  the relative probs be a lot less accurate. To address this issue, we
#  can consider using the average prob of features instead of the
#  cumulative one as the confidence metric to avoid the effect of outlier
#  features, which leads to this version's
#  GaussianNaiveBayes_AveProbs. We will integrate all the three versions of
#  GaussianNaiveBayes local classifiers into one class with an optional
#  version parameter.
#  Moreover, we should consider using the PCs (or some of the embedded
#  components of the features) instead of all the features (genes) as
#  predictors, which leads to this version's
#  GaussianNaiveBayes(on_PCs=True) option.

# Future:
# - (Expected in v1.0.4) To address effect of outliers from snRNA-seq data,
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
# 
# - Improve run_getSingleCellAnnData()
#    + Add: Coordinate remapping []
#    + Add: Shape smoothing [] (if a spot is surrounded by spots of the same
#    class, then make it of that class, too)
# - Improve documentations []
# - Image based local classifier (CellPose + TopACT) []
# - The spatial coordinates by convention should have been saved in `.obsm`
#  because `'x'` and `'y'` are related. But in this tool they are expected to be
#  put separately as columns in `.obs`. This might be a break of convention that
#  needs addressing in the future. []

__version__ = "1.0.3"

from .data import AnnDataPreparer
from .classifier import SVM, GaussianNaiveBayes
from .spatial import SpatialHandler
