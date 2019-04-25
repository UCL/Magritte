# CMakeLists.txt for Magritte
#############################

# Set minimun required CMake version
cmake_minimum_required (VERSION 2.8.12)

# Set project title
project (Magritte)

# Set version number
set (MAGRITTE_VERSION_MAJOR 0)
set (MAGRITTE_VERSION_MINOR 0)

# Set build type to Release to set all optimisation flags
set (CMAKE_BUILD_TYPE Release)


# Build options
option (PYTHON_IO
  "Handling all io through python (slow)" OFF)
option (PYTHON_BINDINGS
  "Python front-end?"                     OFF)
option (OMP_PARALLEL
  "OpenMP parallelisation required?"      OFF)
option (MPI_PARALLEL
  "MPI parallelisation required?"         OFF)
option (GRID_SIMD
  "GRID SIMD vectorisation required?"     OFF)


# Write all binary files to the bin directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_SOURCE_DIR}/bin)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  ${CMAKE_SOURCE_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY
  ${CMAKE_SOURCE_DIR}/bin)


# Add compiler and linker flags
set (CMAKE_C_FLAGS
  "${CMAKE_C_FLAGS}          -mcmodel=large -std=c++11")
set (CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS}        -mcmodel=large -std=c++11")
set (CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} -mcmodel=large -std=c++11")


# Add OpenMP if required
if (OMP_PARALLEL)
  # Look for package
  find_package (OpenMP REQUIRED)
  # Set compiler and linker flags
  set (CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  # Set variable in configuration file
  set (MAGRITTE_OMP_PARALLEL true)
else (OMP_PARALLEL)
  # Set variable in configuration file
  set (MAGRITTE_OMP_PARALLEL false)
endif (OMP_PARALLEL)


# Add directory containing Magritte's source
add_subdirectory (src)

# Add directory containing Magritte's tests
add_subdirectory (tests/unit_tests)

# Add directory containing Magritte's examples
add_subdirectory (examples)


# Link to pybind11 directory if required
if (PYTHON_BINDINGS   OR   PYTHON_IO)
  add_subdirectory (dependencies/pybind11 pybind11)
endif (PYTHON_BINDINGS   OR   PYTHON_IO)

# Add directory containing Magritte's Python bindings
if (PYTHON_BINDINGS)
  add_subdirectory (bindings)
endif (PYTHON_BINDINGS)
