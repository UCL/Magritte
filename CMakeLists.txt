# CMake file for 3D-RT
#------------------------



# Specify the minimum required CMake version

cmake_minimum_required (VERSION 2.8)



# Specify projact name

project(3D-RT)



# Specify the version of the project

set(3DRT_VERSION_MAJOR 0)
set(3DRT_VERSION_MINOR 0)



# Specify the subdirectory containing the source code

add_subdirectory(src)



# Add compiler flags

FIND_PACKAGE(OpenMP REQUIRED)

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



# Add executable
ADD_EXECUTABLE(3D-RT src/3D-RT.cpp)
TARGET_LINK_LIBRARIES(3D-RT 3DRT_library)

