# CMakeLists.txt for Python Python Bindings
###########################################

# Point to the include directory
include_directories (../)
include_directories (../../dependencies)
include_directories (../../dependencies/pybind11/include)

# Set variables in configuration file
set (MAGRITTE_MPI_PARALLEL false)

# If Grid-SIMD is used
if    (GRID_SIMD)
  # Include headers
  include_directories (../../dependencies/Grid-SIMD/include)
  # Set compiler flags
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GRID_COMPILER_FLAGS}")
  # Set variable in configuration file
  set (MAGRITTE_GRID_SIMD true)
else  (GRID_SIMD)
  # Set variable in configuration file
  set (MAGRITTE_GRID_SIMD false)
endif (GRID_SIMD)

# Write configuration file
configure_file (${CMAKE_SOURCE_DIR}/src/configure.hpp.in
                ${CMAKE_SOURCE_DIR}/src/configure.hpp   )

# pybind11 command to create the python module in bin directory
pybind11_add_module (magritte
  pyBindings.cpp
  ../Model/model.cpp
  ../Model/parameters.cpp
  ../Model/Geometry/geometry.cpp
  ../Model/Geometry/Cells/cells.cpp
  ../Model/Geometry/Rays/rays.cpp
  ../Model/Geometry/Boundary/boundary.cpp
  ../Model/Chemistry/chemistry.cpp
  ../Model/Chemistry/Species/species.cpp
  ../Model/Lines/lines.cpp
  ../Model/Lines/LineProducingSpecies/lineProducingSpecies.cpp
  ../Model/Lines/LineProducingSpecies/Quadrature/quadrature.cpp
  ../Model/Lines/LineProducingSpecies/Linedata/linedata.cpp
  ../Model/Lines/LineProducingSpecies/Linedata/CollisionPartner/collisionPartner.cpp
  ../Model/Radiation/radiation.cpp
  ../Model/Radiation/Frequencies/frequencies.cpp
  ../Model/Thermodynamics/thermodynamics.cpp
  ../Model/Thermodynamics/Temperature/temperature.cpp
  ../Model/Thermodynamics/Turbulence/turbulence.cpp
  ../Functions/heapsort.cpp
  ../Simulation/simulation.cpp
  ../Simulation/Raypair/raypair.cpp
  ../Simulation/Image/image.cpp)

if    (GRID_SIMD)
  target_link_libraries (magritte PRIVATE GridSIMD)
endif (GRID_SIMD)

# pybind11 command to create the python module in bin directory
pybind11_add_module (ioMagritte
  ioBindings.cpp
  ../Io/io.cpp
  ../Io/cpp/io_cpp_text.cpp
  ../Io/python/io_python.cpp)
