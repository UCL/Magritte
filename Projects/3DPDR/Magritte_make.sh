#! /bin/bash

# Current folder is project folder
project_folder="$(pwd)/"

# Specify folder where magritte is located
Magritte_folder="/home/frederik/Dropbox/Astro/Magritte"


# Go to Magritte folder
cd $Magritte_folder

# Make Magritte, specifying project folder
make PROJECT_FOLDER=$project_folder

# Go back to project folder
cd $project_folder
