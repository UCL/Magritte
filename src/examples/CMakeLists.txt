# CMakeLists.txt for examples
#############################

# Write all binary files to the bin directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${BINARY_DIRECTORY}/examples)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${BINARY_DIRECTORY}/examples)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${BINARY_DIRECTORY}/examples)

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
include_directories (../)
include_directories (../../dependencies)

# Create executable for examples

#add_executable        (example_1_cpu.exe example_1.cpp    )
#target_link_libraries (example_1_cpu.exe Magritte         )

#add_executable        (example_2_cpu.exe example_2_cpu.cpp)
#target_link_libraries (example_2_cpu.exe Magritte         )

#add_executable        (example_3_cpu.exe example_3.cpp    )
#target_link_libraries (example_3_cpu.exe Magritte         )

#add_executable        (example_4_cpu.exe example_4.cpp    )
#target_link_libraries (example_4_cpu.exe Magritte         )

#add_executable        (example_5_cpu.exe example_5.cpp    )
#target_link_libraries (example_5_cpu.exe Magritte         )

add_executable        (example_8_gpu.exe example_8_gpu.cpp)
target_link_libraries (example_8_gpu.exe Magritte         )

if    (GPU_ACCELERATION)

  find_package (CUDA REQUIRED)

#  cuda_add_executable   (example_1_gpu.exe example_1.cpp    )
#  target_link_libraries (example_1_gpu.exe Magritte         )

#  cuda_add_executable   (example_2_gpu.exe example_2_gpu.cpp)
#  target_link_libraries (example_2_gpu.exe Magritte         )

#  cuda_add_executable   (example_3_gpu.exe example_3.cpp    )
#  target_link_libraries (example_3_gpu.exe Magritte         )

#  cuda_add_executable   (example_4_gpu.exe example_4.cpp    )
#  target_link_libraries (example_4_gpu.exe Magritte         )

#  cuda_add_executable   (example_5_gpu.exe example_5.cpp    )
#  target_link_libraries (example_5_gpu.exe Magritte         )

#  cuda_add_executable   (example_6_gpu.exe example_6_gpu.cpp)
#  target_link_libraries (example_6_gpu.exe Magritte         )

#  cuda_add_executable   (example_7_gpu.exe example_7_gpu.cpp)
#  target_link_libraries (example_7_gpu.exe Magritte         )

#  cuda_add_executable   (example_8_gpu.exe example_8_gpu.cpp)
#  target_link_libraries (example_8_gpu.exe Magritte         )

endif (GPU_ACCELERATION)


# Link MPI libraries
if    (MPI_PARALLEL)
#  target_link_libraries (example_1_cpu.exe   ${MPI_C_LIBRARIES})
#  target_link_libraries (example_2_cpu.exe   ${MPI_C_LIBRARIES})
#  target_link_libraries (example_3_cpu.exe   ${MPI_C_LIBRARIES})
#  target_link_libraries (example_4_cpu.exe   ${MPI_C_LIBRARIES})
#  target_link_libraries (example_8_gpu.exe   ${MPI_C_LIBRARIES})
endif (MPI_PARALLEL)
