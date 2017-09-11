#! /bin/bash



# Compile the pre_setup and create the Makefile for setup

cd setup
g++ -o pre_setup.exe pre_setup.cpp
cmake .
cd ..



# Create the Makefile

cmake .



# Clean up

rm cmake_install.cmake

rm CMakeCache.txt
