# CMakeLists.txt for 3D-RT (main directory)
# -----------------------------------------



# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify projact name

project(3D-RT)


# Specify the version of the project

set(3DRT_VERSION_MAJOR 0)
set(3DRT_VERSION_MINOR 0)


# Resize the stack to 128 Mb (UNIX version)

add_custom_target( RESIZE_STACK ALL
                   COMMAND ulimit -S -s 131072 )


# Make the setup executable and execute setup

add_custom_target( MAKE_SETUP ALL
                   COMMAND make PARAMETERS_FILE=../parameters.txt && ./setup
                   WORKING_DIRECTORY /home/frederik/Dropbox/Astro/3D-RT/setup )


# Specify the subdirectory containing the source code

add_subdirectory(src)


# Add compiler flags

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
  set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()


# Create executable for 3D-RT main code

add_executable( 3D-RT   src/3D-RT.cpp )


# Link the library of the /src directory to the 3D-RT main code

target_link_libraries( 3D-RT   _3DRT )
