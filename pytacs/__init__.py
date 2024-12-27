# pytacs: Python Topological Automatic Cell Segmentation -
#  an improved version of TopACT (https://gitlab.com/kfbenjamin/topact)
#  implementation.

# Author: Liu X., 2024

# TODO v0.9.3:
# - Improve run_getSingleCellAnnData()
#    + Add: Coordinate remapping []
#    + Add: Shape smoothing [] (if a spot is surrounded by spots of the same class, then make it of that class, too)
# - Add README [v]
# - Add requirements.txt [v]
# - Improve documentations []

# Updates v0.9.4:
# - Index implicitly tranformed to str [v, but still considering changing this]
#  Note that all "tidied" indexes ("integers") are actually strings. So call them by
#  adata.obs.loc['0'], .loc['1'], etc.
# - Test dependency on python 3.12.2, passed. [v]

__version__ = '0.9.4'

from .classifier import LocalClassifier
from .spatial import SpatialHandler
from .data import AnnDataPreparer
