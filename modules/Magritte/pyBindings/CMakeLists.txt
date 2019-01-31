# CMakeLists.txt for Python Python Bindings
###########################################

# Point to the include directory
include_directories (../include)

# Link to pybind11
add_subdirectory (/home/frederik/software/pybind11-2.2.4  pybind11)

# pybind11 command to create the python module in bin directory
pybind11_add_module (pyMagritte
  bindings.cpp
  ../src/io.cpp
  ../src/io_text.cpp
  ../src/model.cpp
  ../src/cells.cpp
  ../src/rays.cpp
#  ../src/species.cpp
#  ../src/frequencies.cpp
  ../src/temperature.cpp)
#  ../src/heapsort.cpp)

add_library (PyIo
  ../src/io.cpp
  io_Python.cpp)
target_link_libraries (PyIo PRIVATE
  pybind11::embed)

add_executable (example
  main.cpp)
target_link_libraries (example
  PyIo)
