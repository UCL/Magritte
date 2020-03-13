# CMakeLists.txt for Magritte's source
######################################


# Headers are included in this folder and in the dependencies
include_directories (.)
include_directories (../dependencies)


# If MPI is used
if    (MPI_PARALLEL)
  message (STATUS "Getting ready for MPI...")
  # Look for package
  find_package (MPI REQUIRED)
  # Include headers
  include_directories (SYSTEM ${MPI_INCLUDE_PATH})
  # Set variable in configuration file
  set (MAGRITTE_MPI_PARALLEL true)
else  (MPI_PARALLEL)
  message (STATUS "NOT ready for MPI...")
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

if    (GPU_ACCELERATION)
  set (MAGRITTE_GPU_ACCELERATION true)
else  (GPU_ACCELERATION)
  set (MAGRITTE_GPU_ACCELERATION false)
endif (GPU_ACCELERATION)

# Write configuration file
configure_file (${CMAKE_SOURCE_DIR}/src/configure.hpp.in
                ${CMAKE_SOURCE_DIR}/src/configure.hpp   )

# Create Magritte library
if    (GPU_ACCELERATION)
  # Add CUDA package
  find_package (CUDA REQUIRED)
  # Create the library
  cuda_add_library (Magritte
    Io/cpp/io_cpp_text.cpp
    Model/model.cpp
    Model/parameters.cpp
    Model/Geometry/geometry.cpp
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
    Simulation/simulation.cu
    Simulation/Raypair/raypair.cpp
    Simulation/Raypair/rayblock.cpp
    Simulation/Raypair/raypair.cu
#    Simulation/Raypair/rayblock.cu
    Simulation/Image/image.cpp)
  # Set library properties
  set_target_properties (Magritte PROPERTIES PREFIX "gpu")
else  (GPU_ACCELERATION)
  add_library (Magritte
    Io/cpp/io_cpp_text.cpp
    Model/model.cpp
    Model/parameters.cpp
    Model/Geometry/geometry.cpp
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
    Simulation/Raypair/rayblock.cpp
    Simulation/Image/image.cpp)
  # Set library properties
  set_target_properties (Magritte PROPERTIES PREFIX "cpu")
endif (GPU_ACCELERATION)

if    (GRID_SIMD)
  target_link_libraries (Magritte   GridSIMD)
endif (GRID_SIMD)

if    (MPI_PARALLEL)
  target_link_libraries (Magritte   ${MPI_C_LIBRARIES})
endif (MPI_PARALLEL)

if    (PYTHON_IO)
  # Include pybind11
  include_directories (../dependencies/pybind11/include)
  # Create library for python io
  add_library (PyIo Io/python/io_python.cpp)
  # Link embedding
  target_link_libraries (PyIo PRIVATE pybind11::embed)
  # Link into Magritte
  target_link_libraries (Magritte   PyIo)
endif (PYTHON_IO)

# Add directory containing Magritte's Python bindings
if    (PYTHON_BINDINGS)
  add_subdirectory (bindings)
endif (PYTHON_BINDINGS)

# Add directory containing Magritte's examples
add_subdirectory (examples)
