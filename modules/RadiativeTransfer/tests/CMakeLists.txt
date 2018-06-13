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

#Include directories for Common library
include_directories( ../../RadiativeTransfer/src )

# # Execute setup
# add_custom_target(MAKE_SETUP ALL
#                   COMMAND python setup.py ${CMAKE_CURRENT_BINARY_DIR}/
#                   WORKING_DIRECTORY ../../../../pySetup)

# Add src directory
# add_subdirectory( ${CMAKE_CURRENT_BINARY_DIR}/../src/
# ${CMAKE_CURRENT_BINARY_DIR}/RADIATIVE_TRANSFER )

# Create test_main.cpp lib (as advised in Catch docs)
add_library( _TEST_RT_MAIN
main.cpp )

# Create executable to test rays
add_executable( test_rays.exe
test_rays.cpp )
target_link_libraries( test_rays.exe
_RADIATIVE_TRANSFER
_TEST_RT_MAIN )

# Create executable to test cells
add_executable( test_cells.exe
test_cells.cpp )
target_link_libraries( test_cells.exe
_RADIATIVE_TRANSFER
_TEST_RT_MAIN )

## Create executable to test ray_tracing
#add_executable( test_ray_tracing.exe
#test_ray_tracing.cpp )
#target_link_libraries( test_ray_tracing.exe
#_RADIATIVE_TRANSFER
#_TEST_RT_MAIN )

# Create executable to test feautrier solver
add_executable( test_set_up_ray.exe
test_set_up_ray.cpp )
target_link_libraries( test_set_up_ray.exe
_RADIATIVE_TRANSFER
_TEST_RT_MAIN )

# Create executable to test feautrier solver
add_executable( test_solve_ray.exe
test_solve_ray.cpp )
target_link_libraries( test_solve_ray.exe
_RADIATIVE_TRANSFER
_TEST_RT_MAIN )

# Create executable to test radiation field
add_executable( test_radiation.exe
test_radiation.cpp )
target_link_libraries( test_radiation.exe
_RADIATIVE_TRANSFER
_TEST_RT_MAIN )

# Create executable to test interpolation
add_executable( test_interpolation.exe
test_interpolation.cpp )
target_link_libraries( test_interpolation.exe
_RADIATIVE_TRANSFER
_TEST_RT_MAIN )

# Create executable to test heapsort
add_executable( test_heapsort.exe
test_heapsort.cpp )
target_link_libraries( test_heapsort.exe
_RADIATIVE_TRANSFER
_TEST_RT_MAIN )

# Create executable to test frequencies
add_executable( test_frequencies.exe
test_frequencies.cpp )
target_link_libraries( test_frequencies.exe
_RADIATIVE_TRANSFER
_TEST_RT_MAIN )

# Create executable to test frequencies
add_executable( test_lines.exe
test_lines.cpp )
target_link_libraries( test_lines.exe
_RADIATIVE_TRANSFER
_TEST_RT_MAIN )

# Create executable to test species
add_executable( test_species.exe
test_species.cpp )
target_link_libraries( test_species.exe
_RADIATIVE_TRANSFER
_TEST_RT_MAIN )

# Create executable to test RadiativeTransfer
add_executable( test_RadiativeTransfer.exe
test_RadiativeTransfer.cpp )
target_link_libraries( test_RadiativeTransfer.exe
_RADIATIVE_TRANSFER
_TEST_RT_MAIN )

# Create executable to test radiative transfer
#add_executable( test_RadiativeTransfer.exe
#test_RadiativeTransfer.cpp )
#target_link_libraries( test_RadiativeTransfer.exe
#_RADIATIVE_TRANSFER
#_TEST_MAIN )

# Ensure test_ray_tracing.hpp depends on setup command defined above
# add_dependencies( test_ray_tracing.exe
# MAKE_SETUP )
