# CMakeLists.txt for 3D-RT_tests
# ------------------------------



# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(3D-RT_tests)


# Add compiler flags

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
  set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()


# Add 3DRT and HEALPix libraries containing the source code (which is to be tested)

find_library( 3DRT
              lib_3DRT.a /home/frederik/Dropbox/Astro/3D-RT/src
              DOC "3DRT library" )

find_library( HEALPix
              lib_HEALPix.a /home/frederik/Dropbox/Astro/3D-RT/src/HEALPix
              DOC "HEALPix library" )


# Add subdirectory containing the test source code

add_subdirectory(test_src)


# Resize the stack to 128 Mb (UNIX version)

add_custom_target( RESIZE_STACK ALL
                   COMMAND ulimit -S -s 131072 )


# Make the setup executable and execute setup

add_custom_target( MAKE_SETUP ALL
                   COMMAND make PARAMETERS_FILE=../tests/parameters_for_testing.txt && ./setup
                   WORKING_DIRECTORY /home/frederik/Dropbox/Astro/3D-RT/setup )


# Create executable to test exact_feautrier

add_executable( test_exact_feautrier   test_src/test_exact_feautrier.cpp )


# Link the necessary libraries

target_link_libraries( test_exact_feautrier   _test_main   ${3DRT}   ${HEALPix} )


# Create executable to test column_density_calculator

add_executable( test_column_density_calculator   test_src/test_column_density_calculator.cpp )


# Link the necessary libraries

target_link_libraries( test_column_density_calculator   _test_main   ${3DRT}   ${HEALPix} )


# Create executable to test level_population_solver

add_executable( test_level_population_solver   test_src/test_level_population_solver.cpp )


# Link the necessary libraries

target_link_libraries( test_level_population_solver   _test_main   ${3DRT} )


# Create executable to test rate_calculations_radfield

add_executable( test_radfield_tools   test_src/test_radfield_tools.cpp )


# Link the necessary libraries

target_link_libraries( test_radfield_tools   _test_main   ${3DRT} )


# Create executable to test ray_tracing

add_executable( test_ray_tracing   test_src/test_ray_tracing.cpp )


# Link the necessary libraries

target_link_libraries( test_ray_tracing   _test_main   ${3DRT}   ${HEALPix} )


# Create executable to test species_tools

add_executable( test_species_tools   test_src/test_species_tools.cpp )


# Link the necessary libraries

target_link_libraries( test_species_tools   _test_main   ${3DRT} )


# Create executable to test spline

add_executable( test_spline   test_src/test_spline.cpp )


# Link the necessary libraries

target_link_libraries( test_spline   _test_main   ${3DRT} )
