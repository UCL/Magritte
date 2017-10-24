#! /bin/bash


date_stamp=`date +%Y-%m-%d_%H:%M:%S`;

output_directory="output/files/${date_stamp}_output/";

echo "std::string OUTPUT_DIRECTORY = \"$output_directory\";" > outputdirectory.hpp

# Make the setup file, execute if make was succesful and make a directory for the output

make PARAMETERS_FILE=../parameters.txt && ./setup.exe && mkdir "../$output_directory"
