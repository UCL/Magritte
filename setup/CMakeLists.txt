# CMakeLists.txt for Magritte setup
# ---------------------------------


# Specify the minimum required CMake version
cmake_minimum_required(VERSION 2.8)


# VTK includes
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})


# Add compiler flags
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS
  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -mcmodel=medium")
  set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -mcmodel=medium")
  set(CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -mcmodel=medium")
endif()


# Create setup library
add_library( _setup
setup_data_structures.cpp
setup_healpixvectors.cpp
setup_initializers.cpp
setup_data_tools.cpp)


# Add the HEALPix directory
add_subdirectory(HEALPix)


# Create executable for setup
add_executable(setup.exe setup.cpp)


# Link the setup library to the executable
target_link_libraries(setup.exe _setup)
target_link_libraries(setup.exe _HEALPix)
target_link_libraries(setup.exe ${VTK_LIBRARIES})
