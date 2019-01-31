
# CMakeLists.txt for Magritte's source
######################################

# Point to the include directory
include_directories (../include)

# Create Magritte library
add_library (Magritte
  io.cpp
  io_text.cpp
  model.cpp
  cells.cpp
  rays.cpp
  species.cpp
  linedata.cpp
#  frequencies.cpp
  temperature.cpp)
#  heapsort.cpp)
