# CMakeLists.txt for Magritte: RadiativeTransfer (/src directory)
# ----------------------------------------------------------------

# Add compiler flags
find_package( OpenMP REQUIRED )
if(OPENMP_FOUND)
  set( CMAKE_C_FLAGS
  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -mcmodel=large" )
  set( CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -mcmodel=large" )
  set( CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -mcmodel=large" )
endif()

# Add HEALPix subdirectory
add_subdirectory(HEALPix)

# Include this directory to find Eigen
include_directories(.)

# Create a Radiative Transfer library
add_library( _RADIATIVE_TRANSFER
rays.hpp
cells.hpp
medium.cpp
radiation.cpp
RadiativeTransfer.hpp
interpolation.cpp
set_up_ray.hpp
solve_ray.cpp )

# Link HEALPix library
target_link_libraries( _RADIATIVE_TRANSFER
_HEALPix )

# Set linker language
set_target_properties( _RADIATIVE_TRANSFER PROPERTIES LINKER_LANGUAGE CXX )

