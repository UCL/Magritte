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

# Include source directory to find Eigen
include_directories(../src)

# # Execute setup
# add_custom_target(MAKE_SETUP ALL
#                   COMMAND python setup.py ${CMAKE_CURRENT_BINARY_DIR}/
#                   WORKING_DIRECTORY ../../../../pySetup)

# Add src directory
# add_subdirectory( ${CMAKE_CURRENT_BINARY_DIR}/../src/
# ${CMAKE_CURRENT_BINARY_DIR}/RADIATIVE_TRANSFER )

add_executable( example_feautrier.exe 
example_feautrier.cpp )
target_link_libraries( example_feautrier.exe
_RADIATIVE_TRANSFER )

add_executable( example_RadiativeTransfer.exe
example_RadiativeTransfer.cpp )
target_link_libraries( example_RadiativeTransfer.exe
_RADIATIVE_TRANSFER )

# Ensure test_ray_tracing.hpp depends on setup command defined above
# add_dependencies( test_ray_tracing.exe
# MAKE_SETUP )
