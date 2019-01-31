# CMakeLists.txt for Magritte
#############################

# Set minimun required CMake version
cmake_minimum_required (VERSION 2.8.12)

# Set project title
project (Magritte)

# Write all binary files to the bin directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_SOURCE_DIR}/bin)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  ${CMAKE_SOURCE_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY
  ${CMAKE_SOURCE_DIR}/bin)

# Add directory containing Magritte's source
add_subdirectory (src)

# Add directory containing Magritte's tests
add_subdirectory (tests)

# Add directory containing Magritte's Python bindings
add_subdirectory (pyBindings)
