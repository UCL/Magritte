#!/bin/bash


# Produce an input grid

#python input/grid_1D.py



# Make sure the stack is large enough to hold the grid

# The command below sets the stack size to 128 Mb

ulimit -S -s 131072


# Setup the code

g++ -o setup src/setup.cpp
./setup


# Compile and make an executable

cd CMake
make
cd ..



export OMP_NUM_THREADS=1

./3D-RT

#cd output
#python levelplot.py
#cd ..