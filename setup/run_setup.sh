#! /bin/bash


# Get the current date and time to label the output files

date_stamp=`date +%Y-%m-%d_%H:%M:%S`;

output_directory="output/files/${date_stamp}_output/";

echo "std::string OUTPUT_DIRECTORY = \"$output_directory\";" > outputdirectory.hpp


# Make the rate_equation file for the chemistry, based on parameters.hpp

python make_rates.py


# Make a directory for the output and the plots

mkdir "../$output_directory"

mkdir "../$output_directory/plots/"


# Copy the input parameters to the output file

cp ../parameters.hpp ../$output_directory/parameters.hpp
