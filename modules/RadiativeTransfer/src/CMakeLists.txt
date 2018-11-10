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

# Uncomment the lines below if you want to build from RadiativeTransfer directory
#add_subdirectory( ../../Lines
#  ../../Lines )

# Include directories for Eigen library
include_directories( . )

#Include directories for Lines library
include_directories( ../.. )

# Set folder where Grid is instaled
#set( GRID_INSTALL_FOLDER "/home/frederik/Codes/Grid/install/" )

# Extract compiler flags for Grid
execute_process( COMMAND ${GRID_INSTALL_FOLDER}bin/grid-config --cxxflags
	OUTPUT_VARIABLE GRID_CPP_FLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE )
execute_process( COMMAND ${GRID_INSTALL_FOLDER}bin/grid-config --ldflags
	OUTPUT_VARIABLE GRID_LD_FLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE )
execute_process( COMMAND ${GRID_INSTALL_FOLDER}bin/grid-config --libs
	OUTPUT_VARIABLE GRID_LIB_FLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE )

# Include directories for Grid
include_directories( "${GRID_INSTALL_FOLDER}include/Grid/" )
include_directories( "${GRID_INSTALL_FOLDER}include/" )

# Link directory for Grid
link_directories( "${GRID_INSTALL_FOLDER}lib/" )

# Add compiler flags for Grid
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GRID_CPP_FLAGS}" )
set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GRID_LD_FLAGS}" )

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
  image.cpp
  profile.hpp
  RadiativeTransfer.hpp
  set_up_ray.hpp
  solve_ray.hpp
  timer.hpp
  heapsort.cpp
  interpolation.hpp )

# Link libraries
target_link_libraries( _RADIATIVE_TRANSFER
  _HEALPix
  _LINES
  -lGrid
  ${GRID_LIB_FLAGS} )

# Set linker language
set_target_properties( _RADIATIVE_TRANSFER PROPERTIES LINKER_LANGUAGE CXX )
