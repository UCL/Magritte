#! /bin/bash

# Remove everything
rm -rf Eigen
rm -rf pybind11
rm -rf Grid-SIMD
rm -rf scorep

# Get Eigen
wget http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz
# Extract only the Eigen headers
tar -zxvf 3.3.7.tar.gz eigen-eigen-323c052e1731/Eigen/ --directory Eigen --strip-components=1
# Remove tar ball
rm 3.3.7.tar.gz

# Get pybind11
wget https://github.com/pybind/pybind11/archive/v2.2.4.tar.gz
# Extract whole directory
tar -zxvf v2.2.4.tar.gz
# Rename the folder
mv pybind11-2.2.4 pybind11
# Remove tar ball
rm v2.2.4.tar.gz

# Get Grid-SIMD
wget https://github.com/Magritte-code/Grid-SIMD/archive/master.zip
# Extract whole directory
unzip master.zip
# Rename the folder
mv Grid-SIMD-master Grid-SIMD
# Remove zip file
rm master.zip

# Get Score-P
wget https://www.vi-hps.org/cms/upload/packages/scorep/scorep-5.0.tar.gz
# Extract the whole directory
tar -zxvf scorep-5.0.tar.gz
# Create scorep folder
mkdir scorep
# Rename the folder
mv scorep-5.0 scorep
# Remove zip file
rm scorep-5.0.tar.gz
