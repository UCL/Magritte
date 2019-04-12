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

# Add OpenMP
find_package( OpenMP REQUIRED )
if(OPENMP_FOUND)
  set (CMAKE_C_FLAGS
  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -mcmodel=large -std=c++11")
  set (CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -mcmodel=large -std=c++11")
  set (CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -mcmodel=large -std=c++11")
endif()

# Find Python
#find_package (Python)

# Link to pybind11
add_subdirectory (dependencies/pybind11  pybind11)

# Add directory containing Magritte's source
add_subdirectory (src)

# Add directory containing Magritte's Python bindings
add_subdirectory (bindings)

# Add directory containing Magritte's tests
add_subdirectory (tests/unit_tests)

# Add directory containing Magritte's examples
add_subdirectory (examples)
