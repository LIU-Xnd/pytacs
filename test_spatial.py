assert __name__ == "__main__", "This module is not meant to be imported!"

from pytacs.classifier import SVM
from pytacs.data import AnnDataPreparer
from pytacs.spatial import SpatialHandlerParallel
import scanpy as sc

print("Step 1. Prepare the snRNA-seq and spRNA-seq data")
data_preper = AnnDataPreparer(
    sn_adata=sc.read_h5ad("snRNA_mouse_demo.h5ad"),
    sp_adata=sc.read_h5ad("spRNA_mouse_demo.h5ad"),
)
print(data_preper)

print("Step 2. Train a local classifier")
clf = SVM(threshold_confidence=0.75)
clf.fit(sn_adata=data_preper.sn_adata)
print(clf)
print(f"{data_preper.sp_adata.obs=}")
print("Test SpatialHandlerAutopilot")
sph = SpatialHandlerParallel(
    adata_spatial=data_preper.sp_adata,
    local_classifier=clf,
    threshold_adjacent=3,
    threshold_delta_n_features=0,
    max_spots_per_cell=20,
    n_parallel=50,
)
print(sph)
sph.run_preMapping()
sph.run_segmentation(3, warnings=True)
print(sph)
