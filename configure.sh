#! /bin/bash



# Create and execute the Makefile for the setup

cd setup
cmake .
make
cd ..


# Create the Makefile for Magritte

cmake .
