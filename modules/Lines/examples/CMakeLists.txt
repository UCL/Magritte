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
include_directories( ../src )

#Include directories for Common library
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

# # Execute setup
# add_custom_target(MAKE_SETUP ALL
#                   COMMAND python setup.py ${CMAKE_CURRENT_BINARY_DIR}/
#                   WORKING_DIRECTORY ../../../../pySetup)

# Add src directory
# add_subdirectory( ${CMAKE_CURRENT_BINARY_DIR}/../src/
# ${CMAKE_CURRENT_BINARY_DIR}/RADIATIVE_TRANSFER )


#add_executable( example_linedata.exe 
#example_linedata.cpp )
#target_link_libraries( example_linedata.exe
#_LEVELS )

add_executable( example_Lines_Lime1.exe 
	example_Lines_Lime1.cpp )
target_link_libraries( example_Lines_Lime1.exe
  _LINES
  _RADIATIVE_TRANSFER
  ${MPI_LIBRARIES} )

add_executable( example_levels.exe 
	example_levels.cpp )
target_link_libraries( example_levels.exe
  _LINES
  _RADIATIVE_TRANSFER
  ${MPI_LIBRARIES} )

# Add MPI flags
if(MPI_COMPILE_FLAGS)
	set_target_properties( example_Lines_Lime1.exe PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
	set_target_properties( example_Lines_Lime1.exe PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()


# Ensure test_ray_tracing.hpp depends on setup command defined above
# add_dependencies( test_ray_tracing.exe
# MAKE_SETUP )
