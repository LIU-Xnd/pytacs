# Basic Usage of `pytacs`:

(Updated in 2025.9.12)

## Step 1. Prepare your data (Old Fashioned)

```Python
import pytacs  # Our tool
import scanpy as sc  # A popular single-cell analysis tool

# Step 1. Prepare snRNA-seq and ST data
data_preper = tax.AnnDataPreparer(
	sn_adata=sc.read_h5ad("your-snRNA-seq-data.h5ad"),
    sp_adata=sc.read_h5ad("your-ST-data.h5ad"),
)
# # Filter for highly variable genes, if you want
# data_preper.filter_highly_variable_genes()
print(data_preper)

sn_adata = data_preper.sn_adata
sp_adata = data_preper.sp_adata
```

Output:

```
--- AnnDataPreparer (pytacs) ---
- sn_adata: AnnData object with n_obs × n_vars = 30308 × 42563
    obs: 'old_index', 'cell_type'
- sp_adata: AnnData object with n_obs × n_vars = 9216 × 42563
    obs: 'old_index'
    obsm: 'spatial'
--- --- --- --- ---
```

## Step 1. Prepare your data (New Fashioned)

```Python
import pytacs  # Our tool
import scanpy as sc  # A popular single-cell analysis tool

# Step 1. Prepare snRNA-seq and ST data
sn_adata=sc.read_h5ad("your-snRNA-seq-data.h5ad"),
sp_adata=sc.read_h5ad("your-ST-data.h5ad"),
# make sure spatial coordinates saved in .obsm['spatial']

# You can pre-bin your spatial data for faster computation and better performance
sp_adata = pytacs.binX(sp_adata, binsize=9) # 9 um

# Reinit index for Pytacs compatibility
pytacs.reinit_index(sn_adata)
pytacs.reinit_index(sp_adata)
```

## Step 2. Train a local classifier

```Python
# Step 2. Train a local classifier
clf = pytacs.SVM()
clf.fit(sn_adata)
```

## Step 3. Aggregate the spatial transcriptome

```Python
aggres = pytacs.rw_aggregate( # Or use pytacs.rw_aggregate_sequential for low mem
    st_anndata=sp_adata,
    classifier=clf,
    max_iter=4,
    steps_per_iter=1,
    nbhd_radius=2.4,
    max_propagation_radius=5,
    mode_metric='inv_dist',
    mode_embedding='pc',
    mode_aggregation='unweighted',
    mode_prune='proportional',
)

sp_adata.obs['cell_type_pytacs'] = pytacs.extract_celltypes_full(aggres)
sp_adata.obs['cell_size_pytacs'] = pytacs.extract_cell_sizes_full(aggres)

# Get type-refined pseudo-single-cell spatial transcriptomics 

sp_adata.obs['cell_id_pytacs'] = pytacs.ctrbin_cellseg(
    ann_count_matrix=pytacs.SpTypeSizeAnnCntMtx(
        count_matrix=sp_adata.X,
        spatial_distances=sp_adata.obsp['spatial_distances'],
        cell_types=sp_adata.obs['cell_type_pytacs'],
        cell_sizes=sp_adata.obs['cell_size_pytacs'],
    ),
    attitude_to_undefined='exclusive',
    allow_reassign=False,
)
# Till now, sp_adata is well-annotated spot-level anndata

pseudo_cell_adata = pytacs.aggregate_spots_to_cells_parallel(
    sp_adata,
    n_workers=40,
)
# Till now, annotated pseudo-cell anndata is generated.
```

## Step 4. Save results
```Python
sp_adata.write_h5ad('Your/output/path/for/spot-level.h5ad', compression='gzip')
pseudo_cell_adata.write_h5ad('Your/output/path/for/single-cell', compression='gzip')

```