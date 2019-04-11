#! /bin/bash

# Get Eigen
wget http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz
# Extract only the Eigen headers
tar -zxvf 3.3.7.tar.gz eigen-eigen-323c052e1731/Eigen/ --directory Eigen --strip-components=1
# Remove tar ball
rm 3.3.7.tar.gz

# Get pybind11
git clone git@github.com:pybind/pybind11.git

# Clone Grid-SIMD
git clone git@github.com:Magritte-code/Grid-SIMD.git
