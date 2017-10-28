#! /bin/bash



# Create and execute the Makefile for the setup

cd setup
cmake .
make



# Execute the setup for Magritte

./setup.exe
cd ..



# Create the Makefile for Magritte

cmake .



# Clean up

rm cmake_install.cmake

rm CMakeCache.txt
