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
add_subdirectory( HEALPix )

# Include directories for Eigen library
include_directories( . )

#Include directories for Lines library
include_directories( ../.. )

# Create a Radiative Transfer library
add_library( _RADIATIVE_TRANSFER
constants.hpp
rays.hpp
cells.hpp
species.cpp
temperature.cpp
frequencies.cpp
lines.cpp
scattering.cpp
radiation.cpp
profile.cpp
RadiativeTransfer.hpp
set_up_ray.hpp
solve_ray.cpp
timer.hpp
heapsort.cpp
interpolation.cpp )

# Link libraries
target_link_libraries( _RADIATIVE_TRANSFER
_HEALPix
_LINES )

# Set linker language
set_target_properties( _RADIATIVE_TRANSFER PROPERTIES LINKER_LANGUAGE CXX )
