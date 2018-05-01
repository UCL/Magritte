# CMakeLists.txt for Magritte: RadiativeTransfer
# ----------------------------------------------

# Specify the minimum required CMake version
cmake_minimum_required( VERSION 2.8 )

# Specify projact name
project( Magritte_RadiativeTransfer )

# Specify project version
set( MAGRITTE_VERSION_MAJOR 0 )
set( MAGRITTE_VERSION_MINOR 1 )

# Set build type to Release to get optimal performance
set( CMAKE_BUILD_TYPE Release )

# Add compiler flags for C++11
set( CMAKE_CXX_FLAGS
"${CMAKE_CXX_FLAGS} -std=c++11 -std=gnu++11" )

# Add compiler flags for OpenMP
find_package( OpenMP REQUIRED )
if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS
  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
  set( CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()

# Create documentation
add_custom_target( DOCUMENTATION ALL
COMMAND doxygen Doxyfile
WORKING_DIRECTORY docs )


# Run setup script (if parameters.hpp changed)
# add_custom_target(MAKE_SETUP ALL
#                   COMMAND python setup.py $(PROJECT_FOLDER)
#                   WORKING_DIRECTORY pySetup)

# Specify subdirectories containing source code and tests
add_subdirectory( src )
add_subdirectory( tests )

# VTK includes
# find_package(VTK REQUIRED)
# include(${VTK_USE_FILE})


# Create executable for Magritte main code
# add_executable(Magritte.exe src/Magritte.cpp)
# Ensure that Magritte depends on setup command defined above
# add_dependencies(Magritte.exe MAKE_SETUP)
# Link the library of the /src directory to Magritte main code
# target_link_libraries(Magritte.exe _MAGRITTE)
# Link VTK libraries
# target_link_libraries(Magritte.exe ${VTK_LIBRARIES})
