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
#set (CMAKE_BUILD_TYPE Relwithdebinfo)
#set (CMAKE_BUILD_TYPE Debug)
#set (CUDA_NVCC_FLAGS " -g -G")
set (CUDA_NVCC_FLAGS "-x cu -O5")

# Build options
option (PYTHON_IO        "Handling all io through python (slow)" OFF)
option (PYTHON_BINDINGS  "Python front-end?"                     OFF)
option (OMP_PARALLEL     "OpenMP parallelisation required?"      OFF)
option (MPI_PARALLEL     "MPI parallelisation required?"         OFF)
option (GRID_SIMD        "GRID SIMD vectorisation required?"     OFF)
option (PERF_ANALYSIS    "Build for performance analysis"        OFF)
option (GPU_ACCELERATION "Build using the CUDA GPU solver"       OFF)

# Set variable for use in the source code (defined in configure.hpp)
if    (PYTHON_IO)
  set (MAGRITTE_PYTHON_IO  true)
else  (PYTHON_IO)
  set (MAGRITTE_PYTHON_IO  false)
endif (PYTHON_IO)

if    (GRID_SIMD)
  set (MAGRITTE_GRID_SIMD true)
else  (GRID_SIMD)
  set (MAGRITTE_GRID_SIMD false)
endif (GRID_SIMD)

# Set directory for binaries (and provide a separate one for instrumented binaries for performance testing)
if    (PERF_ANALYSIS)
  set (BINARY_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/tests/performance)
else  (PERF_ANALYSIS)
  set (BINARY_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
endif (PERF_ANALYSIS)

# Write all binary files to the bin directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${BINARY_DIRECTORY})
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${BINARY_DIRECTORY})
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${BINARY_DIRECTORY})

# Use C++11
set (CMAKE_C_FLAGS          "${CMAKE_C_FLAGS}          -std=c++11")
set (CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS}        -std=c++11")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -std=c++11")

# Set large memory model
set (CMAKE_C_FLAGS          "${CMAKE_C_FLAGS}          -mcmodel=large")
set (CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS}        -mcmodel=large")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mcmodel=large")

# Add OpenMP if required
if    (OMP_PARALLEL)
  # Look for package
  find_package (OpenMP REQUIRED)
  # Set compiler and linker flags
  set (CMAKE_C_FLAGS          "${CMAKE_C_FLAGS}          ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS}        ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  # Set variable in configuration file
  set (MAGRITTE_OMP_PARALLEL true)
else  (OMP_PARALLEL)
  # Set variable in configuration file
  set (MAGRITTE_OMP_PARALLEL false)
endif (OMP_PARALLEL)

if    (GRID_SIMD)
  # Add directory containing Grid-SIMD's source
  add_subdirectory (dependencies/Grid-SIMD)
  # Set Grid compiler flags
  set (GRID_COMPILER_FLAGS "-mavx -mf16c")
endif (GRID_SIMD)

# Link to pybind11 directory if required
if    (PYTHON_BINDINGS OR PYTHON_IO)
  add_subdirectory (dependencies/pybind11 pybind11)
endif (PYTHON_BINDINGS OR PYTHON_IO)

# Add directory containing Magritte's source
add_subdirectory (src)

# Add directory containing Magritte's tests
# add_subdirectory (tests/unit)
