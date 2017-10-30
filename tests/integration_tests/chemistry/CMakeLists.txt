# CMakeLists.txt for tests/integration_tests/chemistry/
# -----------------------------------------------------



# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(Magritte_chemistry_tests)


IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    message("debug mode")
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)


# Resize the stack to 128 Mb (UNIX version)

# add_custom_target( RESIZE_STACK ALL
#                    COMMAND ulimit -S -s 131072 )


# Make the setup executable and execute setup

add_custom_command( OUTPUT /home/frederik/Dropbox/Astro/Magritte/src/sundials/rate_equations.cpp
                    COMMAND bash run_test_setup.sh && echo done
                    WORKING_DIRECTORY /home/frederik/Dropbox/Astro/Magritte/setup
                    DEPENDS /home/frederik/Dropbox/Astro/Magritte/tests/integration_tests/chemistry/parameters.hpp )

add_custom_target( MAKE_SETUP
                   DEPENDS /home/frederik/Dropbox/Astro/Magritte/src/sundials/rate_equations.cpp )


# Add compiler flags

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
  set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()


# Add library containing the main and the MAGRITTE source code (which is to be tested)

add_library( _chemistry
test_main.cpp
../../../src/declarations.hpp
../../../src/definitions.hpp
../../../src/initializers.cpp
../../../src/read_input.cpp
../../../src/species_tools.cpp
../../../src/read_chemdata.cpp
../../../src/create_healpixvectors.cpp
../../../src/ray_tracing.cpp
../../../src/heapsort.cpp
../../../src/chemistry.cpp
../../../src/reaction_rates.cpp
../../../src/calc_reac_rates.cpp
../../../src/calc_reac_rates_rad.cpp
../../../src/radfield_tools.cpp
../../../src/spline.cpp
../../../src/calc_AV.cpp
../../../src/calc_UV_field.cpp
../../../src/calc_column_density.cpp
../../../src/calc_rad_surface.cpp
../../../src/calc_temperature_dust.cpp
../../../src/calc_LTE_populations.cpp
../../../src/write_output.cpp )


# Ensure that all source code depends on the setup command defined above

add_dependencies( _chemistry MAKE_SETUP )


# Specify path to SUNDIALS header files

set( SUNDIALS_INC_DIR
     ../../../src/sundials/include
     CACHE STRING "Location of SUNDIALS header files" )

include_directories(${SUNDIALS_INC_DIR})


# Set search path for SUNDIALS libraries

set( SUNDIALS_LIB_DIR
     ../../../src/sundials/lib
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

add_library( _sundials   ../../../src/sundials/rate_equation_solver.cpp )


# Link the necessary libraries

target_link_libraries( _sundials   ${SUNDIALS_LIBS} )


# Add HEALPix library

find_library( HEALPix
              lib_HEALPix.a ../../../src/HEALPix
              DOC "HEALPix library" )


target_link_libraries( _chemistry   _sundials ${HEALPix} )


# Create executable to test calc_reac_rates_rad

add_executable( test_chemistry.exe   test_chemistry.cpp )


# Link the necessary libraries

target_link_libraries( test_chemistry.exe   _chemistry )
