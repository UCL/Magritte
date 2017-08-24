# CMakeLists.txt for 3D-RT (/sundials directory)
# ----------------------------------------------



# Specify path to SUNDIALS header files

set( SUNDIALS_INC_DIR
     include
     CACHE STRING "Location of SUNDIALS header files" )


# Set search path for SUNDIALS libraries

set( SUNDIALS_LIB_DIR
     lib
     CACHE STRING "Location of SUNDIALS library files" )


# Find the SUNDIALS solver's library

find_library( SUNDIALS_SOLVER_LIB
              sundials_cvode ${SUNDIALS_LIB_DIR}
              DOC "CVODE library" )


# Find the NVECTOR library

find_library( SUNDIALS_NVEC_LIB
              sundials_nvecserial ${SUNDIALS_LIB_DIR}
              DOC "NVECTOR library" )


# Set additional libraries

set( SUNDIALS_EXTRA_LIB
     -lm /usr/lib/x86_64-linux-gnu/librt.so
     CACHE STRING "Additional libraries")


# List of all libraries

set( SUNDIALS_LIBS ${SUNDIALS_SOLVER_LIB} ${SUNDIALS_NVEC_LIB} ${SUNDIALS_EXTRA_LIB} )


# Create a sundials library

add_library( _sundials   rate_equation_solver.cpp )


# Link the necessary libraries

target_link_libraries( _sundials   ${SUNDIALS_LIBS} )

