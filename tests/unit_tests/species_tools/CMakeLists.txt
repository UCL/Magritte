# CMakeLists.txt for tests/unit_tests/species_tools
# -------------------------------------------------


# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(3D-RT_tests)


# Resize the stack to 128 Mb (UNIX version)

add_custom_target( RESIZE_STACK ALL
                   COMMAND ulimit -S -s 131072 )


# Add the test main and the 3DRT source code (which is to be tested)

add_library( _species_tools test_main.cpp ../../../src/declarations.hpp ../../../src/definitions.hpp ../../../src/species_tools.cpp )


# Create executable to test species_tools

add_executable( test_species_tools   test_species_tools.cpp )


# Link the necessary libraries

target_link_libraries( test_species_tools   _species_tools )

