#! /bin/bash

for parameter_file in parameters/parameters*.hpp
do

  echo Running with input from $parameter_file

  cp $parameter_file ../../parameters.hpp

  cd ../..
  make
  ./Magritte.exe
  cd tests/performance

done
