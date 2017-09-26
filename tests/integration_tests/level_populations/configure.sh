#! /bin/bash



# Make sure 3D-RT is configured

cd ../../../
./configure.sh
cd tests/integration_tests/level_populations


# Create the Makefile

cmake .



# Clean up

rm cmake_install.cmake

rm CMakeCache.txt
