# CMakeLists.txt for examples
#############################

# Write all binary files to the bin directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY
${CMAKE_SOURCE_DIR}/bin/examples)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY
${CMAKE_SOURCE_DIR}/bin/examples)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY
${CMAKE_SOURCE_DIR}/bin/examples)

# Include directory containing source
include_directories (../src)

# Create executable for examples
add_executable (example_1.exe
  example_1.cpp)
target_link_libraries (example_1.exe
  _Magritte
  PyIo)

add_executable (example_2.exe
  example_2.cpp)
target_link_libraries (example_2.exe
  _Magritte
  PyIo)
