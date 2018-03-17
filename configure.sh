#! /bin/bash


# Create and execute the Makefile for the setup
cd pySetup
cmake .
make
cd ..


# Create the Makefile for Magritte
cmake .
