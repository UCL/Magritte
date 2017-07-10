#!/bin/bash

cd input
python grid_1D.py
cd ..

export OMP_NUM_THREADS=1

icc -g -fopenmp -o 3DRT src/3DRT.cpp -lm

./3DRT

#cd output
#python levelplot.py
#cd ..
