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

# Add RadiativeTransfer subdirectory
add_subdirectory( ../../RadiativeTransfer
../../RadiativeTransfer )

#Include directories for Common and RadiativeTransfer library
include_directories( ../.. )

# Include directories for Eigen library
include_directories( . )

# Create a Levels library
add_library( _LINES
linedata.cpp
levels.cpp
acceleration_Ng.cpp
Lines.hpp )

target_link_libraries( _LINES
_RADIATIVE_TRANSFER )

# Set linker language
set_target_properties( _LINES 
PROPERTIES LINKER_LANGUAGE CXX )

