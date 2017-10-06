# CMakeLists.txt for tests/unit_tests/radfield_tools
# --------------------------------------------------


# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(Magritte_tests)


# Resize the stack to 128 Mb (UNIX version)

add_custom_target( RESIZE_STACK ALL
                   COMMAND ulimit -S -s 131072 )


# Make the setup executable and execute setup

add_custom_target( MAKE_SETUP ALL
                   COMMAND make PARAMETERS_FILE=../tests/unit_tests/radfield_tools/parameters_for_testing.txt && ./setup
                   WORKING_DIRECTORY ../../../setup )


# Add library containing the main and the MAGRITTE source code (which is to be tested)

add_library( _radfield_tools   test_main.cpp ../../../src/declarations.hpp ../../../src/definitions.hpp ../../../src/data_tools.cpp ../../../src/read_chemdata.cpp ../../../src/radfield_tools.cpp ../../../src/spline.cpp )


# Create executable to test calc_reac_rates_rad

add_executable( test_radfield_tools   test_radfield_tools.cpp )


# Link the necessary libraries

target_link_libraries( test_radfield_tools   _radfield_tools )

