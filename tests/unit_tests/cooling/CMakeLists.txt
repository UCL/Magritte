# CMakeLists.txt for tests/unit_tests/cooling
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
                   COMMAND make PARAMETERS_FILE=../tests/unit_tests/cooling/parameters_for_testing.txt && ./setup
                   WORKING_DIRECTORY ../../../setup )


# Add the Eigen directory so the header files can be found

include_directories(../../../src)


# Add compiler flags

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
  set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()


# Add HEALPix library

find_library( HEALPix
              lib_HEALPix.a ../../../src/HEALPix
              DOC "HEALPix library" )


# Add library containing the main and the MAGRITTE source code (which is to be tested)

add_library( _cooling   test_main.cpp ../../../src/declarations.hpp ../../../src/definitions.hpp ../../../src/initializers.cpp ../../../src/read_input.cpp ../../../src/data_tools.cpp ../../../src/setup_data_structures.cpp ../../../src/species_tools.cpp ../../../src/read_chemdata.cpp ../../../src/read_linedata.cpp ../../../src/create_healpixvectors.cpp ../../../src/ray_tracing.cpp ../../../src/heapsort.cpp ../../../src/level_populations.cpp ../../../src/radiative_transfer.cpp ../../../src/calc_C_coeff.cpp ../../../src/level_population_solver.cpp ../../../src/cooling.cpp ../../../src/exact_feautrier.cpp ../../../src/write_output.cpp )


# Create executable to test calc_reac_rates_rad

add_executable( test_cooling   test_cooling.cpp )


# Link the necessary libraries

target_link_libraries( test_cooling   _cooling ${HEALPix} )
