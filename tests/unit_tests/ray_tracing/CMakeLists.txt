# CMakeLists.txt for tests/unit_tests/ray_tracing
# -----------------------------------------------


# Specify minimum required CMake version
cmake_minimum_required(VERSION 2.8)


# Specify project name
project(Magritte_tests)


# Set build type to Release to get optimal performance
set(CMAKE_BUILD_TYPE Debug)


# VTK includes
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})


# Make setup executable and execute setup
add_custom_command(OUTPUT ALL /home/frederik/Dropbox/Astro/Magritte/src/Magritte_config.hpp
                   COMMAND bash run_test_setup.sh unit_tests/ray_tracing
                   WORKING_DIRECTORY ../../../setup
                   DEPENDS parameters.hpp)

add_custom_target(MAKE_SETUP
                  DEPENDS /home/frederik/Dropbox/Astro/Magritte/src/Magritte_config.hpp)


# Add test main and MAGRITTE source code (which is to be tested)
add_library(_ray_tracing
test_main.cpp
../../../setup/setup_data_tools.cpp
../../../src/initializers.cpp
../../../src/read_input.cpp
../../../src/ray_tracing.cpp
../../../src/heapsort.cpp
../../../src/write_output.cpp
../../../src/write_txt_tools.cpp)


# Ensure _ray_tracing depends on setup command defined above
add_dependencies(_ray_tracing MAKE_SETUP)


# Add compiler flags
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


# Add HEALPix library
find_library(HEALPix
             lib_HEALPix.a ../../../src/HEALPix
             DOC "HEALPix library")


# Create executable to test ray_tracing
add_executable(test_ray_tracing.exe test_ray_tracing.cpp)


# Ensure test_ray_tracing.hpp depends on setup command defined above
add_dependencies(test_ray_tracing.exe MAKE_SETUP)


# Link necessary libraries
target_link_libraries(test_ray_tracing.exe _ray_tracing)
target_link_libraries(test_ray_tracing.exe ${HEALPix})
target_link_libraries(test_ray_tracing.exe ${VTK_LIBRARIES})
