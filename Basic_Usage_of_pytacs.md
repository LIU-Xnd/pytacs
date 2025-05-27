# Basic Usage of `pytacs`:

## Step 1. Prepare your data

```Python
import pytacs as tax  # Our tools
import scanpy as sc  # A popular single-cell analysis tool

# Step 1. Prepare snRNA-seq and ST data
data_preper = tax.AnnDataPreparer(
	sn_adata=sc.read_h5ad("your-snRNA-seq-data.h5ad"),
    sp_adata=sc.read_h5ad("your-ST-data.h5ad"),
)
# # Filter for highly variable genes, if you want
# data_preper.filter_highly_variable_genes()
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
    obs: 'cell_type_groundtruth', 'cell_belong_groundtruth', 'old_index'
    osbm: 'spatial'
--- --- --- --- ---
```

## Step 2. Train a local classifier

```Python
# Step 2. Train a local classifier
clf = tax.SVM()
clf.fit(data_preper.sn_adata)
```

## Step 3. Aggregate the spatial transcriptome

```Python
>>> agg_res = tax.rw_aggregate(
    st_anndata=data_prep.sp_adata,
    classifier=clf,
    max_iter=20,
    steps_per_iter=3,
    nbhd_radius=2.4,
    max_propagation_radius=10.,
    mode_metric='inv_dist',
    mode_embedding='pc',
    mode_aggregation='unweighted',
    n_pcs=50,
)
>>> ct_full = extract_celltypes_full(agg_res)

# Plot the celltypes
>>> import seaborn as sns
>>> sns.scatterplot(
    x=data_prep.sp_adata.obsm['spatial'][:,0],
    y=data_prep.sp_adata.obsm['spatial'][:,1],
    hue=ct_full,
)
# Get refined binned pseudo-single-cell spatial transcriptomics 
>>> ann_mtx = tax.SpTypeSizeAnnCntMtx(
    count_matrix,
    spatial_coords,
    cell_types,
    cell_sizes,
)
>>> ann_mtx_sc = tax.ctrbin_cellseg_parallel(
    ann_mtx,
)
```