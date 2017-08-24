# CMakeLists.txt for tests/unit_tests/reaction_rates
# --------------------------------------------------


# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(3D-RT_tests)


# Resize the stack to 128 Mb (UNIX version)

add_custom_target( RESIZE_STACK ALL
                   COMMAND ulimit -S -s 131072 )


# Make the setup executable and execute setup

add_custom_target( MAKE_SETUP ALL
                   COMMAND make PARAMETERS_FILE=../tests/unit_tests/reaction_rates/parameters_for_testing.txt && ./setup
                   WORKING_DIRECTORY ../../../setup )


# Add library containing the main and the 3DRT source code (which is to be tested)

add_library( _reaction_rates   test_main.cpp ../../../src/declarations.hpp ../../../src/definitions.hpp ../../../src/data_tools.cpp ../../../src/species_tools.cpp ../../../src/read_chemdata.cpp ../../../src/reaction_rates.cpp ../../../src/rate_calculations.cpp ../../../src/rate_calculations_radfield.cpp ../../../src/radfield_tools.cpp ../../../src/spline.cpp )


# Create executable to test rate_calculations_radfield

add_executable( test_reaction_rates   test_reaction_rates.cpp )


# Link the necessary libraries

target_link_libraries( test_reaction_rates   _reaction_rates )

