#!/bin/bash


#  an input grid

cd input
python grid_1D_regular.py 101
cd ..


# Setup species and rates

#cd src/MakeRates
#python make_rates.py reactionFile=../../data/rates_reduced.d speciesFile=../../data/species_reduced.d outputPrefix=1 sortSpecies=False logForm=False fileFormat=Rate95 codeFormat=C
#cd ../..


# Compile and make an executable

#make




#export OMP_NUM_THREADS=1

#./3D-RT

#cd output
#python levelplot.py
#cd ..
