# CMakeLists.txt for Python Python Bindings
###########################################

# Point to the include directory
include_directories (../include)

# Link to pybind11
add_subdirectory (/home/frederik/software/pybind11-2.2.4  pybind11)

# pybind11 command to create the python module in bin directory
pybind11_add_module (pyMagritte
  bindings.cpp
  ../src/input.cpp)
#    ../src/model.cpp
#  ../src/cells.cpp
#  ../src/rays.cpp
#  ../src/species.cpp
#  ../src/frequencies.cpp
#  ../src/temperature.cpp
#  ../src/heapsort.cpp)
