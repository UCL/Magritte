# CMakeLists.txt for tests/unit_tests/exact_feautrier
# ---------------------------------------------------


# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(3D-RT_tests)


# Resize the stack to 128 Mb (UNIX version)

add_custom_target( RESIZE_STACK ALL
                   COMMAND ulimit -S -s 131072 )


# Make the setup executable and execute setup

add_custom_target( MAKE_SETUP ALL
                   COMMAND make PARAMETERS_FILE=../tests/unit_tests/exact_feautrier/parameters_for_testing.txt && ./setup
                   WORKING_DIRECTORY ../../../setup )


# Add library containing the main and the 3DRT source code (which is to be tested)

add_library( _exact_feautrier   test_main.cpp ../../../src/declarations.hpp ../../../src/definitions.hpp ../../../src/initializers.cpp ../../../src/read_input.cpp ../../../src/create_healpixvectors.cpp ../../../src/ray_tracing.cpp ../../../src/heapsort.cpp ../../../src/exact_feautrier.cpp )


# Add compiler flags

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
  set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()


# Add  HEALPix library

find_library( HEALPix
              lib_HEALPix.a ../../../src/HEALPix
              DOC "HEALPix library" )


# Create executable to test exact_feautrier

add_executable( test_exact_feautrier   test_exact_feautrier.cpp )


# Link the necessary libraries

target_link_libraries( test_exact_feautrier   _exact_feautrier   ${HEALPix} )

