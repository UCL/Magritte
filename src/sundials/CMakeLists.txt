# Specify the rate_equations and jacobian file
#SET(rate_equations /home/frederik/Dropbox/Astro/3D-RT/src/sundials/rate_equations.cpp)
#SET(jacobian /home/frederik/Dropbox/Astro/3D-RT/src/sundials/jacobian.cpp)


# Specify path to SUNDIALS header files
SET( SUNDIALS_INC_DIR
     /home/frederik/Dropbox/Astro/3D-RT/src/sundials/include
     CACHE STRING "Location of SUNDIALS header files" )


# Set search path for SUNDIALS libraries
SET( SUNDIALS_LIB_DIR
     /home/frederik/Dropbox/Astro/3D-RT/src/sundials/lib )


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
SET(SUNDIALS_LIBS ${SUNDIALS_SOLVER_LIB} ${SUNDIALS_NVEC_LIB} ${SUNDIALS_EXTRA_LIB} )

add_library(sundials_library rate_equation_solver.cpp rate_equations.cpp jacobian.cpp)
TARGET_LINK_LIBRARIES(sundials_library ${SUNDIALS_LIBS})
