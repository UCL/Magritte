#! /bin/bash

project_folder="$(pwd)/"

Magritte_folder="/home/frederik/Dropbox/Astro/Magritte"


# Go to Magritte folder
cd $Magritte_folder

# Make Magritte
make PROJECT_FOLDER=$project_folder

# Go back to project folder
cd $project_folder
