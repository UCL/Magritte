#! /bin/bash

# Configuration for Linux

# Get dependencies
cd dependencies
bash get_dependencies.sh

# Create and activate magritte conda environment
conda env create -f magritte_conda_environment.yml
conda activate magritte_env
