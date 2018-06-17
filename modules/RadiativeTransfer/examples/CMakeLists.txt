# Add C++11 compiler flags
set( CMAKE_CXX_FLAGS
"${CMAKE_CXX_FLAGS} -std=c++11 -std=gnu++11" )

# Add OpenMP compiler flags
find_package( OpenMP REQUIRED )
if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS
  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
  set( CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()

# Include directories for src
include_directories(../src)

#Include directories for Lines library
include_directories( ../.. )

# Set folder where Grid is instaled
set( GRID_INSTALL_FOLDER "/home/frederik/Codes/Grid/install/" )

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

add_executable( example_solve_ray.exe 
  example_solve_ray.cpp )
target_link_libraries( example_solve_ray.exe
  _RADIATIVE_TRANSFER )

add_executable( example_RadiativeTransfer.exe
  example_RadiativeTransfer.cpp )
target_link_libraries( example_RadiativeTransfer.exe
  _RADIATIVE_TRANSFER )
