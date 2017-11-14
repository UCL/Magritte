# CMakeLists.txt for tests/unit_tests/feautrier
# ---------------------------------------------------


# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(Magritte_tests)


# Resize the stack to 128 Mb (UNIX version)

# add_custom_target( RESIZE_STACK ALL
#                    COMMAND ulimit -S -s 131072 )


# Make the setup executable and execute setup

add_custom_command( OUTPUT /home/frederik/Dropbox/Astro/Magritte/src/sundials/rate_equations.cpp
                    COMMAND bash run_test_setup.sh unit_tests/feautrier
                    WORKING_DIRECTORY /home/frederik/Dropbox/Astro/Magritte/setup
                    DEPENDS /home/frederik/Dropbox/Astro/Magritte/tests/unit_tests/feautrier/parameters.hpp )

add_custom_target( MAKE_SETUP
                   DEPENDS /home/frederik/Dropbox/Astro/Magritte/src/sundials/rate_equations.cpp )



# Add library containing the main and the MAGRITTE source code (which is to be tested)

add_library( _feautrier
test_main.cpp
../../../src/initializers.cpp
../../../src/read_input.cpp
../../../src/create_healpixvectors.cpp
../../../src/ray_tracing.cpp
../../../src/heapsort.cpp
../../../src/feautrier.cpp
../../../src/write_output.cpp )


# Ensure that all source code depends on the setup command defined above

add_dependencies( _feautrier MAKE_SETUP )


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


# Create executable to test feautrier

add_executable( test_feautrier   test_feautrier.cpp )


# Link the necessary libraries

target_link_libraries( test_feautrier   _feautrier   ${HEALPix} )
