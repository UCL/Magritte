# CMakeLists.txt for tests/unit_tests
# -----------------------------------


# Specify the minimum required CMake version
cmake_minimum_required(VERSION 2.8)


# Specify project name
project(Magritte_tests)


# Set build type to Release to get optimal performance
set(CMAKE_BUILD_TYPE Debug)


# VTK includes
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})


# Make setup executable and execute setup
# add_custom_command(OUTPUT ALL /home/frederik/Dropbox/Astro/Magritte/src/Magritte_config.hpp
#                    COMMAND bash run_setup.sh /home/frederik/Dropbox/Astro/Magritte/tests/unit_tests/calc_column_density/
#                    WORKING_DIRECTORY ../../../setup)
#
# add_custom_target(MAKE_SETUP
#                   DEPENDS /home/frederik/Dropbox/Astro/Magritte/src/Magritte_config.hpp)


# Make setup executable and execute setup
add_custom_target(MAKE_SETUP ALL
                  COMMAND bash run_setup.sh ${CMAKE_CURRENT_BINARY_DIR}/
                  WORKING_DIRECTORY ../../../setup)


# Add library containing the main
add_library(_calc_column_density
test_main.cpp
../../../setup/setup_data_tools.cpp
../../../src/initializers.cpp
../../../src/read_input.cpp
../../../src/bound.cpp
../../../src/ray_tracing.cpp
../../../src/heapsort.cpp
../../../src/write_output.cpp
../../../src/write_txt_tools.cpp
../../../src/read_chemdata.cpp
../../../src/species_tools.cpp
../../../src/calc_column_density.cpp)


# Ensure _ depends on setup command defined above
add_dependencies(_calc_column_density MAKE_SETUP)


# Add compiler flags
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
  set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()


# HEALPix library
find_library( HEALPix
              lib_HEALPix.a ../../../src/HEALPix
              DOC "HEALPix library" )


# Create executable to test ray_tracing
add_executable(test_calc_column_density.exe test_calc_column_density.cpp)


# Ensure test_ray_tracing.hpp depends on setup command defined above
add_dependencies(test_calc_column_density.exe MAKE_SETUP)


# Link necessary libraries
target_link_libraries(test_calc_column_density.exe _calc_column_density)
target_link_libraries(test_calc_column_density.exe ${HEALPix})
target_link_libraries(test_calc_column_density.exe ${VTK_LIBRARIES})
