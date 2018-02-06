#! /bin/bash



echo " "
echo "--------------"
echo "run_test_setup"
echo "--------------"
echo " "


# Copy the parameters.hpp file to the main directory

cp ../tests/$1parameters.hpp ../parameters.hpp


# Create and execute the Makefile for the setup

cmake .
make


# Get the current date and time to label the output files

date_stamp=`date +%y-%m-%d_%H:%M`;

output_directory="tests/$1/output/files/${date_stamp}_output/";


echo "#define OUTPUT_DIRECTORY \"output/files/${date_stamp}_output/\"" > directories.hpp


# Make the rate_equation file for the chemistry, based on parameters.hpp

python make_rates.py


# Execute setup

./setup.exe


# Make a directory for the output and the plots

mkdir "../$output_directory"

mkdir "../$output_directory/plots/"


# Copy the input parameters to the output file

cp ../parameters.hpp ../$output_directory/parameters.hpp


echo " "
echo "--------------"
echo "--------------"
echo " "
