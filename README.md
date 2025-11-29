# Pytacs - Python-implemented Topology-Aware Cell Segmentation

Note: this is a prototype for DeTACH (https://pypi.org/project/pydetach/). Pytacs was initially a Python package but now supports command-line usage.

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
transcriptomics into single-cellular spots and cell-type mapping leveraging paired scRNA-seq.


## Requirements
It could be simply
installed by `pip install pytacs`.

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

(Updated in 2025.11.27)

Pytacs now is packed as a commandline tool:

See help:

```
$ python -m pytacs -h
```

If you are interested in using Pytacs as a python package, we recommend using pytacs.recipe module.
See docstring in pytacs.recipe.

## Demo

[Demo](./data/demo/demo.ipynb)
