# Basic Usage of `pytacs`:

## Step 1. Prepare your data

```{Python}
import pytacs as tx  # Our tools
import scanpy as sc  # A popular single-cell analysis tool

# Step 1. Prepare snRNA-seq and ST data
data_preper = tx.AnnDataPreparer(
	sn_adata=sc.read_h5ad("your-snRNA-seq-data.h5ad"),
    sp_adata=sc.read_h5ad("your-ST-data.h5ad"),
)
print(data_preper)
```

Output:

```
--- AnnDataPreparer (pytacs) ---
- sn_adata: AnnData object with n_obs × n_vars = 30308 × 42563
    obs: 'batch', 'n_counts', 'mt_proportion', 'old_index', 'cell_type'
    uns: 'annotation_colors', 'batch_colors', 'log1p', 'neighbors', 'pca', 'umap', 'cell_type_colors'
    obsm: 'X_pca', 'X_umap'
    varm: 'PCs'
    layers: 'log1p'
    obsp: 'connectivities', 'distances'
- sp_adata: AnnData object with n_obs × n_vars = 9216 × 42563
    obs: 'x', 'y', 'cell_type_groundtruth', 'cell_belong_groundtruth', 'old_index'
- sn_adata_withNegativeControl: _UNDEFINED
- normalized: False
--- --- --- --- ---
```

## Step 2. Train a local classifier

```{Python}
# Step 2. Train a local classifier
clf = tx.SVM()
clf.fit(data_preper.sn_adata)
```

## Step 3. Handle the spatial transcriptome

```{Python}
# Step 3. Create a spatial handler
sph = tx.SpatialHandlerParallel(
	adata_spatial=data_preper.sp_adata,
    local_classifier=clf,
    n_parallel=1000,  # mem for time
)
print(sph)
```

Output:

```
--- Spatial Handler Autopilot Parallel (pytacs) ---
- adata_spatial: AnnData object with n_obs × n_vars = 9216 × 42563
    obs: 'x', 'y', 'cell_type_groundtruth', 'cell_belong_groundtruth', 'old_index'
- threshold_adjacent: 1.2
- local_classifier: <pytacs.classifier.SVM object at 0x7f6181298fb0>
    + threshold_confidence: 0.75
    + has_negative_control: False
- max_spots_per_cell: 50
- scale_rbf: 1.0
- pre-mapped: False
- n_parallel: 1000
- filtrations: 0 fitted
- single-cell segmentation:
    + new samples: 0
    + AnnData: _UNDEFINED
--- --- --- --- --- ---
```

```{Python}
# Premapping for context information
sph.run_preMapping()
```

Output:

```
Running first round premapping ...
Running second round premapping ...
100%|██████████| 9216/9216 [00:03<00:00, 2974.48it/s]Done.
```

```{Python}
# Segmentation
sph.run_segmentation(
	verbose=False,
	print_summary=True,
)
```

Output:

```
--- Summary ---
Queried 4148 spots, of which 4148 made up confident single cells.
Classes total: {5: 1167, 0: 1104, 4: 48, 1: 399, 6: 946, 2: 397, 7: 12, 3: 75}
Coverage: 100.00%
--- --- --- --- ---
```

## Step 4. Get resulting data or plot cell-type mapping

Get resulting single-cell data by `sph.run_getSingleCellAnnData()`. Get resulting cell types by `sph.get_spatial_classes()`, or plot them by `sph.run_plotClasses()`. Full information of cell-spots mapping is stored in the dictionary `sph.filtrations`, where each key is cell_id, and corresponding value is a list of spot_ids of whom the cell is made up.