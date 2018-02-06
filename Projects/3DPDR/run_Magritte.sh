#! /bin/bash

project_folder="$(pwd)/"

Magritte_folder="/home/frederik/Dropbox/Astro/Magritte"


# Go to Magritte folder
cd $Magritte_folder

# Make Magritte
make PROJECT_FOLDER=$project_folder

# Execute Magritte
./Magritte.exe
