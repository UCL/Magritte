# CMakeLists.txt for tests/unit_tests/species_tools
# -------------------------------------------------


# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(Magritte_tests)


# Resize the stack to 128 Mb (UNIX version)

add_custom_target( RESIZE_STACK ALL
                   COMMAND ulimit -S -s 131072 )


# Make the setup executable and execute setup

add_custom_target( MAKE_SETUP ALL
                   COMMAND make PARAMETERS_FILE=../tests/unit_tests/species_tools/parameters_for_testing.txt && ./setup
                   WORKING_DIRECTORY ../../../setup )


# Add the test main and the MAGRITTE source code (which is to be tested)

add_library( _species_tools test_main.cpp ../../../src/declarations.hpp ../../../src/definitions.hpp ../../../src/species_tools.cpp )


# Create executable to test species_tools

add_executable( test_species_tools   test_species_tools.cpp )


# Link the necessary libraries

target_link_libraries( test_species_tools   _species_tools )

