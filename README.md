# Pytacs - Python-implemented Topology-Aware Cell Segmentation

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

1. The shape of predicted cells can be diverse rather than a rectangle/grid, rendering higher accuracy;
2. Random-Walk-based aggregation strategy with comparative computational speed, making it more
"topology-aware", and rendering higher accuracy especially at cell boundaries;

## Requirements
This package is released on PyPi now! It could be simply
installed by `pip install pytacs` (the package name yet might change).

For conda users,

```Bash
conda create -n pytacs python=3.12 -y
conda activate pytacs
pip install pytacs
```

For python3 users, first make sure your python is
of version 3.12, and then in your working directory,

```Bash
python -m venv pytacs
source pytacs/bin/activate
python -m pip install pytacs
```

For developers, requirements (at develop time) are listed in
`requirements.in` (initial dependencies), `requirements.txt` (full dependencies)
and `requirements.tree.txt` (for a tree view).

For developers using Poetry,
the dependencies lock file is `poetry.lock` and the project information
including main dependencies is listed in `pyproject.toml`. 

To use it for downstream analysis in combination with Squidpy, it is recommended to use a seperate virtual environment to install Squidpy.

## Usage

In the future, there will be a well-prepared `recipe` module for users to use conveniently.

For detailed usage, see [Basic_Usage_of_pytacs.md](./Basic_Usage_of_pytacs.md)

## Demo

[Demo](./data/demo/demo.ipynb)
