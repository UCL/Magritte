# CMakeLists.txt for tests/unit_tests/heating
# -------------------------------------------


# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(Magritte_tests)


# Resize the stack to 128 Mb (UNIX version)

add_custom_target( RESIZE_STACK ALL
                   COMMAND ulimit -S -s 131072 )


# Make the setup executable and execute setup

add_custom_target( MAKE_SETUP ALL
                   COMMAND make PARAMETERS_FILE=../tests/unit_tests/heating/parameters_for_testing.txt && ./setup
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



# Add library containing the main and the MAGRITTE source code (which is to be tested)

add_library( _heating   test_main.cpp ../../../src/declarations.hpp ../../../src/definitions.hpp ../../../src/initializers.cpp ../../../src/read_input.cpp ../../../src/data_tools.cpp ../../../src/species_tools.cpp ../../../src/read_chemdata.cpp ../../../src/chemistry.cpp ../../../src/reaction_rates.cpp ../../../src/calc_reac_rates.cpp ../../../src/calc_reac_rates_rad.cpp ../../../src/radfield_tools.cpp ../../../src/spline.cpp ../../../src/heating.cpp )

target_link_libraries( _heating   _sundials )


# Create executable to test calc_reac_rates_rad

add_executable( test_heating   test_heating.cpp )


# Link the necessary libraries

target_link_libraries( test_heating   _heating )
