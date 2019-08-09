# CMakeLists.txt for examples
#############################

# Write all binary files to the bin directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${BINARY_DIRECTORY}/examples)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${BINARY_DIRECTORY}/examples)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${BINARY_DIRECTORY}/examples)

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

# Include directory containing source
include_directories (../src)
include_directories (../dependencies)

# Create executable for examples
add_executable        (example_1.exe   example_1.cpp)
target_link_libraries (example_1.exe   _Magritte    )

add_executable        (example_2.exe   example_2.cpp)
target_link_libraries (example_2.exe  _Magritte     )

add_executable        (example_3.exe   example_3.cpp)
target_link_libraries (example_3.exe  _Magritte     )

add_executable        (example_4.exe   example_4.cpp)
target_link_libraries (example_4.exe  _Magritte     )
