# Pytacs - Python-implemented Topology-Aware Cell-type Spotting

```
Copyright (C) 2025 Xindong Liu

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
```

A tool for segmenting/integrating sub-cellular spots in high-resolution spatial
transcriptomics into single-cellular spots and cell-type mapping.

Ideas are inspired by (Benjamin et al., 2024)'s work TopACT
(see https://gitlab.com/kfbenjamin/topact).
But Pytacs has improved it in several ways:

1. The shape of predicted cells are diverse rather than a rectangle/grid, rendering hopefully higher accuracy;
2. Random-Walk-based aggregation strategy with comparative computational speed, making it more
"topology-aware", and rendering hopefully higher accuracy especially at cell boundaries;
3. Bootstrap downsampling strategy is adopted for generating ref scRNA-seq, making it
more flexible in terms of ad-hoc cell-type mapping and novel cell-type detection.

## Requirements
This package is expected to be released on PyPi soon. By then, it could be simply
installed by `pip install pytacs` (the package name yet might change).

For developers, requirements (at develop time) are listed in
`requirements.in` (initial dependencies), `requirements.txt` (full dependencies)
and `requirements.tree.txt` (for a tree view).

For developers using Poetry,
the dependencies lock file is `poetry.lock` and the project information
including main dependencies is listed in `pyproject.toml`. 


## Usage

For detailed usage, see [Basic_Usage_of_pytacs.md](./Basic_Usage_of_pytacs.md)

```Python
>>> import pytacs as tax

# Step 1. Prepare the snRNA-seq and spRNA-seq data
>>> data_prep = tax.AnnDataPreparer(sn_adata, sp_adata)

# Step 2. Train a local classifier
>>> clf = tax.SVM()
>>> clf.fit(data_prep.sn_adata)

# Step 3. Integrate spatial spots into pseudo-single-cell-level spots
>>> agg_res = tax.rw_aggregate(
    st_anndata=data_prep.sp_adata,
    classifier=clf,
    max_iter=20,
    steps_per_iter=3,
    nbhd_radius=2.4,
    trim_proportion=0.5,
    max_propagation_radius=10.,
    mode_metric='cosine',
    mode_embedding='pc',
    mode_aggregation='unweighted',
    n_pcs=100,
)
>>> ct_full = extract_celltypes_full(agg_res)

# Plot the celltypes
>>> import seaborn as sns
>>> sns.scatterplot(
    x=data_prep.sp_adata.obsm['spatial'][:,0],
    y=data_prep.sp_adata.obsm['spatial'][:,1],
    hue=ct_full,
)
```

## Demo
(Deprecated)

[Demo](./data/demo/demo.ipynb)
