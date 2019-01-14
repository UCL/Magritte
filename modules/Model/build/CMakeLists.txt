cmake_minimum_required (VERSION 2.8.12)

# Set project title
project (Magritte)

# Point to the include directory
include_directories (../include)

# Link to pybind11
add_subdirectory (/home/frederik/software/pybind11-2.2.4
  ../include/pybind11)

# Point library binary file to the bin directory
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/../bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/../bin)

# pybind11 command to create the python module in bin directory
pybind11_add_module (pyMagritte
  ../src/pyBindings.cpp
  ../src/model.cpp
  ../src/temperature.cpp)

# Add C++ library
add_library (Magritte
  ../src/model.cpp
  ../src/temperature.cpp)
