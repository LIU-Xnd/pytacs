# Pytacs - Python-implemented Topologiy-Aware Cell-type Spotting

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
2. Provides more types of local classifiers, including Gaussian Naive Bayes Model and
our q-Proximity Confidence Model;
3. New strategies are introduced to build a local classifier;
4. Provides a more popular input-output protocol (i.e., h5ad format), and users can
get integrated single-cell ST output data in h5ad format conveniently.

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

```{Python}
>>> import pytacs as tx

# Step 1. Prepare the snRNA-seq and spRNA-seq data
>>> data_prep = tx.AnnDataPreparer(sn_adata, sp_adata)

# Step 2. Train a local classifier
>>> clf = tx.GaussianNaiveBayes()
>>> clf.fit(data_prep.sn_adata)

# Step 3. Integrate spatial spots into single-cell spots
>>> sph = tx.SpatialHandlerParallel(data_prep.sp_adata, clf)
>>> sph.run_preMapping()
>>> sph.run_segmentation()

# Get the integrated single-cell ST data
>>> single_cell = sph.run_getSingleCellAnnData()
```

## Demo
[Demo](./data/demo/demo.ipynb)
