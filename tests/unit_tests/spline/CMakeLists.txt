# CMakeLists.txt for tests/unit_tests/spline
# ------------------------------------------


# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(Magritte_tests)


# Resize the stack to 128 Mb (UNIX version)

add_custom_target( RESIZE_STACK ALL
                   COMMAND ulimit -S -s 131072 )


# Add the test main and the MAGRITTE source code (which is to be tested)

add_library( _spline test_main.cpp ../../../src/spline.cpp )


# Create executable to test spline

add_executable( test_spline   test_spline.cpp )


# Link the necessary libraries

target_link_libraries( test_spline   _spline )

