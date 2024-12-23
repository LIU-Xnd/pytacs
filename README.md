# Pytacs - Python-implemented Topological Automatic Cell Segmentation

**(Still under construction ...)**

A tool for segmenting/integrating sub-cellular spots in spatial transcriptomics into single-cellular spots.
Ideas are based on (Benjamin et al., 2024)'s work TopACT (see https://gitlab.com/kfbenjamin/topact).
But Pytacs has improved it in several ways:

1. The shape of predicted cells are diverse rather than a rectangle/grid;
2. Negative-control samples are introduced for better performance of local classifiers;
3. Provides a more generalized input-output protocol (based on h5ad format / scanpy object), and users can get integrated single-cell ST output data (in h5ad format) conveniently.

## Requirements
```
# python == 3.10.15
numpy == 1.26.4
pandas == 1.5.3
scanpy == 1.9.6
scikit-learn == 1.5.1
scipy == 1.13.1
```
Could install by `$ pip install -r requirements.txt`.

(Still under construction...)

## Usage
```{python}
import pytacs as ts

# Step 1. Prepare the snRNA-seq and spRNA-seq data
data_prep = ts.AnnDataPreparer(sn_adata, sp_adata)
data_prep.simulate_negative_control()
data_prep.normalize()

# Step 2. Train a local classifer
clf = ts.LocalClassifier()
clf.fit(data_prep.sn_adata_withNegativeControl)

# Step 3. Integrate spatial spots into single-cell spots
sph = ts.SpatialHandler(data_prep.sp_adata, clf)
sph.run_segmentation()

# Get the integrated single-cell ST data
single_cell = sph.run_getSingleCellAnnData()
```

## Demo
[demo.ipynb](./demo.ipynb)

## Issues
The spatial coordiates by convention are saved in `.obsm`
because `'x'` and `'y'` are interrelated. But in this tool they are expected to be
put separately as columns in `.obs`. This might be a break of convention that
needs addressing in the future.
