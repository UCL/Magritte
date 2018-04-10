# Specify minimum required CMake version
cmake_minimum_required(VERSION 2.8)

# Specify project name
project(Magritte_test_chemistry)

# Set build type to Debug
set(CMAKE_BUILD_TYPE Debug)

# Add compiler flaggs
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medium -std=c++11 -std=gnu++11")

# Add compiler flags
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS
  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -mcmodel=medium")
  set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -mcmodel=medium")
  set(CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -mcmodel=medium")
endif()

# Execute setup
add_custom_target(MAKE_SETUP ALL
                  COMMAND python setup.py ${CMAKE_CURRENT_BINARY_DIR}/
                  WORKING_DIRECTORY ../../../pySetup)

# Add src directory
add_subdirectory( ${CMAKE_CURRENT_BINARY_DIR}/../../../src/
${CMAKE_CURRENT_BINARY_DIR}/MAGRITTE )

# Create executable to test ray_tracing
add_executable( test_chemistry.exe
test_chemistry.cpp )

# Link necessary libraries
target_link_libraries( test_chemistry.exe
_MAGRITTE )

# Ensure test_ray_tracing.hpp depends on setup command defined above
add_dependencies( test_chemistry.exe
MAKE_SETUP )
