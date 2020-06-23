<img src="docs/Images/Magritte_logo.png" alt="logo" width="360"/>

[![Documentation Status](https://readthedocs.org/projects/magritte/badge/?version=latest)](https://magritte.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/UCL/Magritte.svg?token=j3NNTbFLxGaJNsSoKgCz&branch=master)](https://travis-ci.com/UCL/Magritte)
---

This is the repository of the Magritte main library. Magritte is a modern open-source
software library for 3D radiative transfer. It is a deterministic ray-tracer using a
formal solver that currently focusses on line radiative transfer. See our first
Magritte paper ([arXiv](https://arxiv.org/pdf/1912.08445.pdf),
[MNRAS](https://doi.org/10.1093/mnras/stz3557)) for more details.

Magritte can either be used as a C++ library or as a Python package.

## Installation
First, download the dependencies and configure Magritte using
```bash
bash install.sh
```
Then, create an anaconda environment from the environment file with
```bash
conda env create -f conda_environment.yml
```
Afterwards, activate the environment you just created with
```bash
conda activate magritte
```
Now Magritte can be build using
```bash
bash build.sh
```
This builds the library in the `/bin` folder. Note that if you try to build Magritte from outside the `magritte_env` conda environment compilation might fail or the generated library might not work in python due to version mismatches. Therefore as a general rule: **always build and use Magritte from within the magritte_env conda environment**. To use Magritte's python interface, you should include the binary folder folder in your python path e.g. by including
```python
from sys import path
path.append("path/to/Magritte/bin")
```
To use the setup tools and the mesher, you should also include
```python
path.append("path/to/Magritte/setup")
path.append("path/to/Magritte/mesher")
```

You can have a look at the `build.sh` script for further configuration options.

## Issues
Please report any issues [here](https://github.com/UCL/Magritte/issues).


## Dependencies
* `CMake`, for building;
* `Eigen`, for linear algebra;
* `pybind11`, for interfacing with python;
* `Grid_SIMD`, for vectorisation;
* `cuda`, for GPU acceleration.
