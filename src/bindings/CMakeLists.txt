# CMakeLists.txt for Python Python Bindings
###########################################

# Write the python library to the magritte python package
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${BINARY_DIRECTORY}/../magritte)

# Add Python package
find_package (PythonLibs REQUIRED)

# Include directories
include_directories (../)
include_directories (../../dependencies)
include_directories (../../dependencies/pybind11/include)
include_directories (${PYTHON_INCLUDE_DIRS})

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

# Add CUDA raypair solver
if    (GPU_ACCELERATION)
  # Set variable in configuration file
  set (MAGRITTE_GPU_ACCELERATION true)
  # Write configuration file
  configure_file (${CMAKE_SOURCE_DIR}/src/configure.hpp.in
                  ${CMAKE_SOURCE_DIR}/src/configure.hpp   )
  # Add CUDA package
  find_package (CUDA REQUIRED)
  # Add CUDA library
  cuda_add_library (core SHARED
    pyBindings.cpp
    ../Io/cpp/io_cpp_text.cpp
    ../Io/python/io_python.cpp
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
    ../Simulation/Solver/cpu/cpu_solver.cpp
#    ../Simulation/Solver/simd/simd_solver.cpp
    ../Simulation/Solver/gpu/gpu_solver.cu
    ../Simulation/simulation.cu
    ../Simulation/Raypair/raypair.cpp
#    ../Simulation/Rayblock/rayBlock_v.cpp
#    ../Simulation/Raypair/raypair.cu
#    ../Simulation/Raypair/rayblock.cpp
#    ../Simulation/Raypair/rayblock.cu
    ../Simulation/Image/image.cpp)
  # Link Python libraries
  #target_link_libraries (core ${PYTHON_LIBRARIES})

else  (GPU_ACCELERATION)

  # Set variable in configuration file
  set (MAGRITTE_GPU_ACCELERATION false)
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
  pybind11_add_module (core SHARED
    pyBindings.cpp
    ../Io/cpp/io_cpp_text.cpp
    ../Io/python/io_python.cpp
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
    ../Simulation/Solver/cpu/cpu_solver.cpp
#    ../Simulation/Solver/simd/simd_solver.cpp
    ../Simulation/Raypair/raypair.cpp
#    ../Simulation/Rayblock/rayBlock_v.cpp
    #    ../Simulation/Raypair/rayblock.cpp
    ../Simulation/Image/image.cpp)
  # Link GRID_SIMD libraries if required
  if    (GRID_SIMD)
    target_link_libraries (core PRIVATE GridSIMD)
  endif (GRID_SIMD)

endif (GPU_ACCELERATION)

# Set library properties
set_target_properties (core PROPERTIES PREFIX "")
set_target_properties (core PROPERTIES SUFFIX ".so")

# Link MPI libraries
if    (MPI_PARALLEL)
  target_link_libraries (core PRIVATE ${MPI_C_LIBRARIES})
#  target_link_libraries (core PRIVATE mpi)
endif (MPI_PARALLEL)

