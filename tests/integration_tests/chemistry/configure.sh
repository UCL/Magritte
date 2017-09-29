#! /bin/bash



# Make sure 3D-RT is configured

cd ../../../
./configure.sh
cd tests/integration_tests/chemistry


# Create the Makefile

cmake .

# Execute the Makefile

make


# Clean up

rm cmake_install.cmake

rm CMakeCache.txt
