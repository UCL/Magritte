# CMakeLists.txt for Magritte
# ---------------------------



# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify projact name

project(Magritte)


# Set build type to Release to get optimal performance

set(CMAKE_BUILD_TYPE Release)


# Specify the version of the project

set(MAGRITTE_VERSION_MAJOR 0)
set(MAGRITTE_VERSION_MINOR 1)



# Resize the stack to 128 Mb (UNIX version)

#add_custom_target( RESIZE_STACK ALL
#                   COMMAND ulimit -S -s 131072 )


# Make the setup executable and execute setup

add_custom_command( OUTPUT src/sundials/rate_equations.cpp
                    COMMAND bash run_setup.sh
                    WORKING_DIRECTORY setup
                    DEPENDS parameters.hpp )

add_custom_target( MAKE_SETUP
                   DEPENDS src/sundials/rate_equations.cpp )


# Specify the subdirectory containing the source code

add_subdirectory(src)


# VTK includes

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})


# Add compiler flags

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  set( CMAKE_C_FLAGS
  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -mcmodel=medium" )
  set( CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -mcmodel=medium" )
  set( CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -mcmodel=medium" )
endif()


# Create executable for Magritte main code

add_executable( Magritte.exe   src/Magritte.cpp )


# Ensure that Magritte depends on the setup command defined above

add_dependencies( Magritte.exe   MAKE_SETUP )


# Link the library of the /src directory to the Magritte main code

target_link_libraries( Magritte.exe   _MAGRITTE )
