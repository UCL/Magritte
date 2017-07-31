# CMake file for 3D-RT
#------------------------


# Specify the minimum required CMake version
cmake_minimum_required (VERSION 2.8)


# Specify projact name
project(3D-RT)


# Specify the version of the project
set(3DRT_VERSION_MAJOR 0)
set(3DRT_VERSION_MINOR 0)


MARK_AS_ADVANCED(EXECUTABLE_OUTPUT_PATH LIBRARY_OUTPUT_PATH)


# Set the name of the source to be build
SET(source 3D-RT)


# Specify path to SUNDIALS header files
SET( SUNDIALS_INC_DIR
     /home/frederik/Dropbox/Astro/3D-RT/3D-RT_merged/src/sundials/include
     CACHE STRING "Location of SUNDIALS header files" )


# Set search path for SUNDIALS libraries 
SET( SUNDIALS_LIB_DIR
     /home/frederik/Dropbox/Astro/3D-RT/3D-RT_merged/src/sundials/lib )


# Find the SUNDIALS solver's library
FIND_LIBRARY( SUNDIALS_SOLVER_LIB
              sundials_cvode ${SUNDIALS_LIB_DIR}
              DOC "CVODE library" )


# Find the NVECTOR library
FIND_LIBRARY( SUNDIALS_NVEC_LIB
              sundials_nvecserial ${SUNDIALS_LIB_DIR}
              DOC "NVECTOR library" )


# Set additional libraries
SET( SUNDIALS_EXTRA_LIB
     -lm /usr/lib/x86_64-linux-gnu/librt.so
     CACHE STRING "Additional libraries")

# List of all libraries
SET(SUNDIALS_LIBS ${SUNDIALS_SOLVER_LIB} ${SUNDIALS_NVEC_LIB} ${SUNDIALS_EXTRA_LIB})


# Add compiler flags
FIND_PACKAGE(OpenMP REQUIRED)

if(OPENMP_FOUND)

  message("OPENMP FOUND")

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

endif()


# Add executable
ADD_EXECUTABLE(../${source} ../src/${source}.cpp)
TARGET_LINK_LIBRARIES(../${source} ${SUNDIALS_LIBS})

