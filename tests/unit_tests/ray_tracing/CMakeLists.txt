# CMakeLists.txt for tests/unit_tests/ray_tracing
# -----------------------------------------------


# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(Magritte_tests)


# Set build type to Release to get optimal performance

set(CMAKE_BUILD_TYPE Debug)


# VTK includes

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})


# Resize the stack to 128 Mb (UNIX version)

# add_custom_target( RESIZE_STACK ALL
#                    COMMAND ulimit -S -s 131072 )


# Make the setup executable and execute setup

add_custom_command( OUTPUT ../../../src/declarations.hpp
                    COMMAND make PARAMETERS_FILE=../tests/unit_tests/ray_tracing/parameters.txt && ./setup.exe
                    WORKING_DIRECTORY ../../../setup
                    DEPENDS parameters.txt )

add_custom_target( MAKE_SETUP
                   DEPENDS ../../../src/declarations.hpp  )


# Add the test main and the MAGRITTE source code (which is to be tested)

add_library( _ray_tracing
test_main.cpp
../../../src/declarations.hpp
../../../src/definitions.hpp
../../../src/initializers.cpp
../../../src/read_input.cpp
../../../src/ray_tracing.cpp
../../../src/heapsort.cpp
../../../src/write_output.cpp )


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


# Create executable to test ray_tracing

add_executable( test_ray_tracing   test_ray_tracing.cpp )


# Link the necessary libraries

target_link_libraries( test_ray_tracing   _ray_tracing ${HEALPix} )
