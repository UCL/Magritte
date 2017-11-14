# CMakeLists.txt for tests/unit_tests
# -----------------------------------


# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(Magritte_tests)


# Resize the stack to 128 Mb (UNIX version)

add_custom_target( RESIZE_STACK ALL
                   COMMAND ulimit -S -s 131072 )


# Make the setup executable and execute setup

add_custom_target( MAKE_SETUP ALL
                   COMMAND make PARAMETERS_FILE=../tests/parameters_for_testing.txt && ./setup
                   WORKING_DIRECTORY /home/frederik/Dropbox/Astro/Magritte/setup )


# Add library containing the main

add_library( _test_main   test_main.cpp  )


# Add compiler flags

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
  set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()


# Add MAGRITTE and HEALPix libraries containing the source code (which is to be tested)

find_library( MAGRITTE
              lib_MAGRITTE.a /home/frederik/Dropbox/Astro/Magritte/src
              DOC "MAGRITTE library" )

find_library( HEALPix
              lib_HEALPix.a /home/frederik/Dropbox/Astro/Magritte/src/HEALPix
              DOC "HEALPix library" )


# Add the Eigen directory so the header files can be found

include_directories(/home/frederik/Dropbox/Astro/Magritte/src)


# Create executable to test feautrier

add_executable( test_feautrier   test_feautrier.cpp )


# Link the necessary libraries

target_link_libraries( test_feautrier   _test_main   ${MAGRITTE}   ${HEALPix} )


# Create executable to test calc_column_density

add_executable( test_calc_column_density   test_calc_column_density.cpp )


# Link the necessary libraries

target_link_libraries( test_calc_column_density   _test_main   ${MAGRITTE}   ${HEALPix} )


# Create executable to test level_population_solver

add_executable( test_level_population_solver   test_level_population_solver.cpp )


# Link the necessary libraries

target_link_libraries( test_level_population_solver   _test_main   ${MAGRITTE} )


# Create executable to test calc_reac_rates_rad

add_executable( test_radfield_tools   test_radfield_tools.cpp )


# Link the necessary libraries

target_link_libraries( test_radfield_tools   _test_main   ${MAGRITTE} )


# Create executable to test ray_tracing

add_executable( test_ray_tracing   test_ray_tracing.cpp )


# Link the necessary libraries

target_link_libraries( test_ray_tracing   _test_main   ${MAGRITTE}   ${HEALPix} )


# Create executable to test species_tools

add_executable( test_species_tools   test_species_tools.cpp )


# Link the necessary libraries

target_link_libraries( test_species_tools   _test_main   ${MAGRITTE} )


# Create executable to test spline

add_executable( test_spline   test_spline.cpp )


# Link the necessary libraries

target_link_libraries( test_spline   _test_main   ${MAGRITTE} )
