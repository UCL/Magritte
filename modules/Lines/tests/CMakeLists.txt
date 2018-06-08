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

# Include directories for Eigen library
include_directories( ../src )

#Include directories for RadiativeTransfer library
include_directories( ../.. )

# # Execute setup
# add_custom_target(MAKE_SETUP ALL
#                   COMMAND python setup.py ${CMAKE_CURRENT_BINARY_DIR}/
#                   WORKING_DIRECTORY ../../../../pySetup)

# Add src directory
# add_subdirectory( ${CMAKE_CURRENT_BINARY_DIR}/../src/
# ${CMAKE_CURRENT_BINARY_DIR}/RADIATIVE_TRANSFER )

# Create main.cpp lib (as advised in Catch docs)
add_library( _TEST_MAIN
main.cpp )

# Create executable to test rays
add_executable( test_linedata.exe
test_linedata.cpp )
target_link_libraries( test_linedata.exe
_LINES
_TEST_MAIN )


# Ensure test_ray_tracing.hpp depends on setup command defined above
# add_dependencies( test_ray_tracing.exe
# MAKE_SETUP )
