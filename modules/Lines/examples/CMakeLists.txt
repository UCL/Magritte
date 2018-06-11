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
