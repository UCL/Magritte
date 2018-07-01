# CMakeLists.txt for Magritte: Levels
# -----------------------------------

# Specify the minimum required CMake version
cmake_minimum_required( VERSION 2.8 )

# Specify projact name
project( Magritte_Lines )

# Specify project version
set( MAGRITTE_VERSION_MAJOR 0 )
set( MAGRITTE_VERSION_MINOR 1 )

#SET( CMAKE_CXX_COMPILER "/opt/intel/compilers_and_libraries_2017/linux/bin/intel64/icc" )

# Set build type to Release to get optimal performance
set( CMAKE_BUILD_TYPE Debug )

option(INTEL_CC "Use the Intel compiler." OFF)

if (INTEL_CC)
  find_program(CMAKE_C_COMPILER NAMES icc)
  find_program(CMAKE_CXX_COMPILER NAMES icpc)
  find_program(CMAKE_AR NAMES xiar)
  find_program(CMAKE_LINKER NAMES xild)

  if (CMAKE_C_COMPILER MATCHES CMAKE_C_COMPILER-NOTFOUND OR
      CMAKE_CXX_COMPILER MATCHES CMAKE_CXX_COMPILER-NOTFOUND OR
      CMAKE_AR MATCHES CMAKE_AR-NOTFOUND OR
      CMAKE_LINKER MATCHES CMAKE_LINKER-NOTFOUND)
    message(FATAL_ERROR "Cannot find Intel compiler.  You may need to run `. /opt/intel/bin/compilervars.sh intel64'")
  endif ()
endif (INTEL_CC)

# Add compiler flags for C++11
set( CMAKE_CXX_FLAGS
"${CMAKE_CXX_FLAGS} -std=c++11 -std=gnu++11")

#Add MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# Add compiler flags for OpenMP
find_package( OpenMP REQUIRED )
if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS
  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
  set( CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()

# Create documentation
# add_custom_target( DOCUMENTATION ALL
# COMMAND doxygen Doxyfile
# WORKING_DIRECTORY docs )


# Run setup script (if parameters.hpp changed)
# add_custom_target(MAKE_SETUP ALL
#                   COMMAND python setup.py $(PROJECT_FOLDER)
#                   WORKING_DIRECTORY pySetup)

# Specify subdirectories containing source code and tests
add_subdirectory( src )
add_subdirectory( tests )
add_subdirectory( examples )
