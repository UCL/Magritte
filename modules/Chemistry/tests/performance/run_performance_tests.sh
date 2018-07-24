#! /bin/bash

for parameter_file in parameters/parameters_*.hpp
do

  echo " "
  echo " "
  echo " "
  echo Running with input from $parameter_file
  echo " "
  echo " "

  cp $parameter_file ../../parameters.hpp

  cd ../..
  make
  ./Magritte.exe
  cd tests/performance

done
