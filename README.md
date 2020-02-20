# Magritte
----------

[![Build Status](https://travis-ci.com/UCL/Magritte.svg?token=j3NNTbFLxGaJNsSoKgCz&branch=master)](https://travis-ci.com/UCL/Magritte)


## Install
----------

First, download the dependencies and configure Magritte using
```
$ bash configure.sh
```
Then, create an anaconda environment from the environment file with
```
$ conda env create -f magritte_conda_environment.yml
```
Afterwards, activate the environment you just created with
```
$ conda activate magritte_env
```
Now Magritte can be build using
```
$ bash build.sh
```


magritte_conda_environment contains the default packages plus:
* `healpy`, for uniform discretisations of a sphere (to create uniform rays);
* `h5py`, for reading and writing HDF5 files;
* `bokeh`, for visualusations;
* `jupyter`, for working in notebooks.


### Dependencies
----------------

* `CMake`, for building;
* `Eigen`, for linear algebra;
* `pybind11`, for interfacing with python;
* `Grid_SIMD`, for vectorisation;
* `cuda`, for GPU acceleration.
