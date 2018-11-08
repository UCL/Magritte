#! /bin/bash

# Move into the folder given as the first argument
cd $1

# Create and move into directory "build"
mkdir build
cd build

# Run Cmake for Magritte, to produce Makefile
CC=gcc CXX=g++ cmake -DGRID_INSTALL_FOLDER=$HOME/Codes/GridInst/ $HOME/Dropbox/Astro/Magritte/modules/Lines

# Run Makefile to comile Magritte
make -j4
