#!/bin/bash

cd input
python grid_1D.py
cd ..

export OMP_NUM_THREADS=1

./3D-RT

#cd output
#python levelplot.py
#cd ..
