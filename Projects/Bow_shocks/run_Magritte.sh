#! /bin/bash

project_folder="$(pwd)/"

Magritte_folder="/home/frederik/Dropbox/Astro/Magritte/"


cd $Magritte_folder

make PROJECT_FOLDER=$project_folder
