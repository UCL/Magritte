# Magritte
----------

[![Build Status](https://travis-ci.com/UCL/Magritte.svg?token=j3NNTbFLxGaJNsSoKgCz&branch=master)](https://travis-ci.com/UCL/Magritte)


## Install
----------

Assuming Anaconda (or Miniconda) it creates a conda environment (magritte_env) containing:
* `healpy`, for uniform discretisations of a sphere (to create uniform rays);
* `h5py`, for reading and writing HDF5 files;
* `bokeh`, for visualusations;
* `jupyter`, for working in notebooks.


### Dependencies
----------------

* `CMake`, for building;
* `Eigen`, for linear algebra;
* `pybind11`, for interfacing with python;
* `Grid_SIMD`, for vectorisation.
