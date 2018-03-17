#! /bin/bash


echo " "
echo "------------"
echo "run_setup.sh"
echo "------------"
echo " "


# Copy the parameters.hpp file to the main directory

cp $1parameters.hpp ../src/parameters.hpp


# Get the current date and time to label the output files

date_stamp=`date +%y-%m-%d_%H:%M`;

output_directory="$1output/files/${date_stamp}_output/";

echo "#define OUTPUT_DIRECTORY \"$output_directory\"" > directories.hpp
echo "#define PROJECT_FOLDER \"$1\"" >> directories.hpp


# Make the rate_equation file for the chemistry, based on parameters.hpp

python makeRates.py


# Create and execute the Makefile for the setup

cmake .
make


# Execute setup

python setup.py


# Make a directory for the output and the plots

mkdir "$output_directory"

mkdir "$output_directory/plots/"


# Copy the input parameters to the output file

cp ../src/parameters.hpp $output_directory/parameters.hpp


echo " "
echo "------------"
echo "------------"
echo " "
