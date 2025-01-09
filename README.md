# Pytacs - Python-implemented Topological Automatic Cell Segmentation

**(Still under construction ...)**

A tool for segmenting/integrating sub-cellular spots in spatial transcriptomics into single-cellular spots.
Ideas are based on (Benjamin et al., 2024)'s work TopACT (see https://gitlab.com/kfbenjamin/topact).
But Pytacs has improved it in several ways:

1. The shape of predicted cells are diverse rather than a rectangle/grid;
2. Provides more types of local classifiers, including Gaussian Naive Bayes Model and q-Proximity Confidence Model;
3. New strategies are introduced to build a local classifier;
4. Negative-control samples are introduced for better performance of local classifier (especially SVM);
5. Provides a more generalized input-output protocol (based on h5ad format / scanpy object), and users can get integrated single-cell ST output data (in h5ad format) conveniently.

## Requirements
```
# python == 3.12.2
numpy == 1.26.4
pandas == 2.2.3
scanpy == 1.10.4
scikit-learn == 1.5.2
scipy == 1.14.1
```
Could install by `$ pip install -r requirements.txt`.

## Usage
```{Python}
import pytacs as ts

# Step 1. Prepare the snRNA-seq and spRNA-seq data
data_prep = ts.AnnDataPreparer(sn_adata, sp_adata)

# Step 2. Train a local classifier
clf = ts.GaussianNaiveBayes()
clf.fit(data_prep.sn_adata)

# Step 3. Integrate spatial spots into single-cell spots
sph = ts.SpatialHandler(data_prep.sp_adata, clf)
sph.run_segmentation()

# Get the integrated single-cell ST data
single_cell = sph.run_getSingleCellAnnData()
```

## Demo
[demo.ipynb](./demo.ipynb)
