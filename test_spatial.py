assert (__name__ == '__main__'
        ), 'This module is not meant to be imported!'

from pytacs.classifier import QProximityClassifier, SVM
from pytacs.data import AnnDataPreparer
from pytacs.spatial import SpatialHandlerAutopilot
import scanpy as sc

print("Step 1. Prepare the snRNA-seq and spRNA-seq data")
data_preper = AnnDataPreparer(
    sn_adata=sc.read_h5ad('snRNA_mouse_demo.h5ad'),
    sp_adata=sc.read_h5ad('spRNA_mouse_demo.h5ad'),
)
print(data_preper)

print("Step 2. Train a local classifier")
clf = SVM(
    threshold_confidence=0.75
)
clf.fit(
    sn_adata=data_preper.sn_adata
)
print(clf)

print("Test SpatialHandlerAutopilot")
sph = SpatialHandlerAutopilot(
    adata_spatial=data_preper.sp_adata,
    local_classifier=clf,
    threshold_adjacent=1.2,
    multiplier_sameClass=3
)
print(sph)

sph._firstRound_preMapping()
print("After first round:")
print(f"{sph.adata_spatial.obs=} \n {sph.adata_spatial.obsm['confidence_premapping1']=}")