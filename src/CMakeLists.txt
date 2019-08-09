# CMakeLists.txt for Magritte's source
######################################


# Headers are included in this folder and in the dependencies
include_directories (.)
include_directories (../dependencies)


# If MPI is used
if    (MPI_PARALLEL)
  # Look for package
  find_package (MPI REQUIRED)
  # Include headers
  include_directories (SYSTEM ${MPI_INCLUDE_PATH})
  # Set variable in configuration file
  set (MAGRITTE_MPI_PARALLEL true)
else  (MPI_PARALLEL)
  # Set variable in configuration file
  set (MAGRITTE_MPI_PARALLEL false)
endif (MPI_PARALLEL)


# If Grid-SIMD is used
if    (GRID_SIMD)
  # Include headers
  include_directories (../dependencies/Grid-SIMD/include)
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
  ${CMAKE_SOURCE_DIR}/src/configure.hpp)

# Create Magritte library
add_library (_Magritte
  Io/io.cpp
  Io/cpp/io_cpp_text.cpp
  Model/model.cpp
  Model/parameters.cpp
  Model/Geometry/geometry.cpp
  Model/Geometry/Cameras/cameras.cpp
  Model/Geometry/Cells/cells.cpp
  Model/Geometry/Rays/rays.cpp
  Model/Geometry/Boundary/boundary.cpp
  Model/Chemistry/chemistry.cpp
  Model/Chemistry/Species/species.cpp
  Model/Lines/lines.cpp
  Model/Lines/LineProducingSpecies/lineProducingSpecies.cpp
  Model/Lines/LineProducingSpecies/Quadrature/quadrature.cpp
  Model/Lines/LineProducingSpecies/Linedata/linedata.cpp
  Model/Lines/LineProducingSpecies/Linedata/CollisionPartner/collisionPartner.cpp
  Model/Radiation/radiation.cpp
  Model/Radiation/Frequencies/frequencies.cpp
  Model/Thermodynamics/thermodynamics.cpp
  Model/Thermodynamics/Temperature/temperature.cpp
  Model/Thermodynamics/Turbulence/turbulence.cpp
  Functions/heapsort.cpp
  Simulation/simulation.cpp
  Simulation/Raypair/raypair.cpp
  Simulation/Image/image.cpp)


if    (GRID_SIMD)
  target_link_libraries (_Magritte   GridSIMD)
endif (GRID_SIMD)

if    (MPI_PARALLEL)
  target_link_libraries (_Magritte   ${MPI_C_LIBRARIES})
endif (MPI_PARALLEL)


if    (PYTHON_IO)
  # Include pybind11
  include_directories (../dependencies/pybind11/include)
  # Create library for python io
  add_library (PyIo
    Io/io.cpp
    Io/python/io_python.cpp)
  # Link embedding
  target_link_libraries (PyIo PRIVATE   pybind11::embed)
endif (PYTHON_IO)
