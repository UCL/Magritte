# CMakeLists.txt for unit_tests/ray_tracing


# Specify minimum required CMake version
cmake_minimum_required(VERSION 2.8)


# Specify project name
project(Magritte_tests)

# Set build type to Debug
set(CMAKE_BUILD_TYPE Debug)

# Add directories containing tests
add_subdirectory(t1_1D)
