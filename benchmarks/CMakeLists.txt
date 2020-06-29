# CMakeLists.txt for examples
#############################

# Write all binary files to the bin directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${BINARY_DIRECTORY}/benchmarks)

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

# If GPU acceleration
if    (GPU_ACCELERATION)
  set (MAGRITTE_GPU_ACCELERATION true)
else  (GPU_ACCELERATION)
  set (MAGRITTE_GPU_ACCELERATION false)
endif (GPU_ACCELERATION)

# Write configuration file
configure_file (${CMAKE_SOURCE_DIR}/src/configure.hpp.in
                ${CMAKE_SOURCE_DIR}/src/configure.hpp   )

# Include directory containing source
include_directories (../src)
include_directories (../dependencies)

# Create executable for examples

add_executable (bm1_radiation bm1_radiation.cpp)
add_executable (bm2_levelpops bm2_levelpops.cpp)

target_link_libraries (bm1_radiation Magritte)
target_link_libraries (bm2_levelpops Magritte)

# Link MPI libraries
if    (MPI_PARALLEL)
  target_link_libraries (bm1_radiation ${MPI_C_LIBRARIES})
  target_link_libraries (bm2_levelpops ${MPI_C_LIBRARIES})
endif (MPI_PARALLEL)
