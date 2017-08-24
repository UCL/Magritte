# CMakeLists.txt for tests/integration_tests/chemistry
# ----------------------------------------------------


# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(3D-RT_tests)


# Resize the stack to 128 Mb (UNIX version)

add_custom_target( RESIZE_STACK ALL
                   COMMAND ulimit -S -s 131072 )


# Make the setup executable and execute setup

add_custom_target( MAKE_SETUP ALL
                   COMMAND make PARAMETERS_FILE=../tests/integration_tests/chemistry/parameters_for_testing.txt && ./setup
                   WORKING_DIRECTORY ../../../setup )


# Specify path to SUNDIALS header files

set( SUNDIALS_INC_DIR
     ../../../src/sundials/include
     CACHE STRING "Location of SUNDIALS header files" )


# Set search path for SUNDIALS libraries

set( SUNDIALS_LIB_DIR
     ../../../src/sundials/lib
     CACHE STRING "Location of SUNDIALS library files" )


# Find the SUNDIALS solver's library

find_library( SUNDIALS_SOLVER_LIB
              sundials_cvode ${SUNDIALS_LIB_DIR}
              DOC "CVODE library" )


# Find the NVECTOR library

find_library( SUNDIALS_NVEC_LIB
              sundials_nvecserial ${SUNDIALS_LIB_DIR}
              DOC "NVECTOR library" )


# Set additional libraries

set( SUNDIALS_EXTRA_LIB
     -lm /usr/lib/x86_64-linux-gnu/librt.so
     CACHE STRING "Additional libraries")


# List of all libraries

set( SUNDIALS_LIBS ${SUNDIALS_SOLVER_LIB} ${SUNDIALS_NVEC_LIB} ${SUNDIALS_EXTRA_LIB} )


# Create a sundials library

add_library( _sundials   ../../../src/sundials/rate_equation_solver.cpp )


# Link the necessary libraries

target_link_libraries( _sundials   ${SUNDIALS_LIBS} )



# Add library containing the main and the 3DRT source code (which is to be tested)

add_library( _chemistry   test_main.cpp ../../../src/declarations.hpp ../../../src/definitions.hpp ../../../src/read_input.cpp ../../../src/data_tools.cpp ../../../src/species_tools.cpp ../../../src/read_chemdata.cpp ../../../src/abundances.cpp ../../../src/reaction_rates.cpp ../../../src/rate_calculations.cpp ../../../src/rate_calculations_radfield.cpp ../../../src/radfield_tools.cpp ../../../src/spline.cpp )

target_link_libraries( _chemistry   _sundials )


# Create executable to test rate_calculations_radfield

add_executable( test_chemistry   test_chemistry.cpp )


# Link the necessary libraries

target_link_libraries( test_chemistry   _chemistry )

