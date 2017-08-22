# CMakeLists.txt for tests/unit_tests/level_population_solver
# -----------------------------------------------------------


# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(3D-RT_tests)


# Resize the stack to 128 Mb (UNIX version)

add_custom_target( RESIZE_STACK ALL
                   COMMAND ulimit -S -s 131072 )


# Add library containing the main

add_library( _level_population_solver   test_main.cpp ../../../src/declarations.hpp ../../../src/definitions.hpp ../../../src/level_population_solver.cpp  )


# Add the Eigen directory so the header files can be found

include_directories(../../../src)


# Create executable to test level_population_solver

add_executable( test_level_population_solver   test_level_population_solver.cpp )


# Link the necessary libraries

target_link_libraries( test_level_population_solver   _level_population_solver )

