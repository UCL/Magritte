#! /bin/bash


# Create and execute Makefile for setup
cd pySetup
cmake .
make
cd ..


# Create Makefile for Magritte
cmake .


# Get Magritte_folder
Magritte_folder="$(pwd)/"
