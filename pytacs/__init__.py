# pytacs: Python Topological Automatic Cell Segmentation -
#  an improved version of TopACT (https://gitlab.com/kfbenjamin/topact)
#  implementation.

# Author: Liu X., 2024.12

# Novelty so far:
# - Self-discovered prior knowledge;
# - Improved cell shape approximation;
# - Independency from imaging segmentation information;
# - Improved local classifier strategy - higher inferencing accuracy;

# Updates v1.1.0:
# TACSA - Topological Automatic Cell Segmentation with Autopilot
#  or just TACS, Topological Autopiloting Cell Segmentation?
# Plan:
# - Two-round pre-mapping for inference prior knowledge
#   + 1st-round: spot inference probability
#   + 2nd-round: grouping spots at inference prob to
#    form inference unit; regenerate inference probs
#   based on units.
#   + Use this inference probs for probs of adding
#    next spots.


# SUSPENDED v1.1.0 (beta):
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
# - Write __repr__ for _LocalClassifier and its child classes.

__version__ = "1.2.2"

from .data import AnnDataPreparer
from .classifier import (
    SVM,
    GaussianNaiveBayes,
    QProximityClassifier,
    CosineSimilarityClassifier,
)
from .spatial import SpatialHandler
