# CMakeLists.txt for test
#########################

# Write all binary files to the bin directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_SOURCE_DIR}/bin/tests)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  ${CMAKE_SOURCE_DIR}/bin/tests)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY
  ${CMAKE_SOURCE_DIR}/bin/tests)

# Include directory containing source
include_directories (../src)

# Include directory containing the catch header
include_directories (.)

# Create main lib (as advised in Catch docs, to speed up compilation)
add_library (_TestMain
  main.cpp)


# Create executable to test io
add_executable (test_io_Python.exe
  Io/test_io_Python.cpp)
target_link_libraries (test_io_Python.exe
  PyIo
  _TestMain)

add_executable (test_model.exe
  Model/test_model.cpp)
target_link_libraries (test_model.exe
  PyIo
  _Magritte
  _TestMain)

add_executable (test_parameters.exe
  Model/test_parameters.cpp)
target_link_libraries (test_parameters.exe
  PyIo
  _Magritte
  _TestMain)

add_executable (test_raypair.exe
  Simulation/Raypair/test_raypair.cpp)
target_link_libraries (test_raypair.exe
  PyIo
  _Magritte
  _TestMain)

add_executable (test_lineProducingSpecies.exe
  Model/Lines/LineProducingSpecies/test_lineProducingSpecies.cpp)
target_link_libraries (test_lineProducingSpecies.exe
  PyIo
  _Magritte
  _TestMain)

add_executable (test_lambda.exe
  Model/Lines/LineProducingSpecies/Lambda/test_lambda.cpp)
target_link_libraries (test_lambda.exe
  _Magritte
  _TestMain)

## Create executable to test model
#add_executable (test_model.exe
#  test_model.cpp)
#target_link_libraries (test_model.exe
#  Magritte
#  TestMain)
#
## Create executable to test model
#add_executable (test_rays.exe
#  test_rays.cpp)
#target_link_libraries (test_rays.exe
#  Magritte
#  TestMain)
#
## Create executable to test model
#add_executable (test_cells.exe
#  test_cells.cpp)
#target_link_libraries (test_cells.exe
#  Magritte
#  TestMain)
#
## Create executable to test cells
#add_executable( test_cells.exe
#test_cells.cpp )
#target_link_libraries( test_cells.exe
#_RADIATIVE_TRANSFER
#_TEST_RT_MAIN )
#
#### Create executable to test ray_tracing
###add_executable( test_ray_tracing.exe
###test_ray_tracing.cpp )
###target_link_libraries( test_ray_tracing.exe
###_RADIATIVE_TRANSFER
###_TEST_RT_MAIN )
##
### Create executable to test ray setup
##add_executable( test_set_up_ray.exe
##test_set_up_ray.cpp )
##target_link_libraries( test_set_up_ray.exe
##_RADIATIVE_TRANSFER
##_LINES
##_TEST_RT_MAIN )
##
## Create executable to test feautrier solver
#add_executable( test_solve_ray.exe
#test_solve_ray.cpp )
#target_link_libraries( test_solve_ray.exe
  #_RADIATIVE_TRANSFER
#_TEST_RT_MAIN )
#
### Create executable to test radiation field
##add_executable( test_radiation.exe
##  test_radiation.cpp )
##target_link_libraries( test_radiation.exe
##  _RADIATIVE_TRANSFER
##  _TEST_RT_MAIN )
#
## Create executable to test interpolation
#add_executable( test_interpolation.exe
#test_interpolation.cpp )
#target_link_libraries( test_interpolation.exe
  #_RADIATIVE_TRANSFER
#_TEST_RT_MAIN )
#
## Create executable to test heapsort
#add_executable( test_heapsort.exe
#test_heapsort.cpp )
#target_link_libraries( test_heapsort.exe
  #_RADIATIVE_TRANSFER
#_TEST_RT_MAIN )
#
## Create executable to test frequencies
#add_executable( test_frequencies.exe
#test_frequencies.cpp )
#target_link_libraries( test_frequencies.exe
  #_RADIATIVE_TRANSFER
#_TEST_RT_MAIN )
#
## Create executable to test profile
#add_executable( test_profile.exe
#test_profile.cpp )
#target_link_libraries( test_profile.exe
  #_RADIATIVE_TRANSFER
#_TEST_RT_MAIN )
#
## Create executable to test frequencies
#add_executable( test_lines.exe
#test_lines.cpp )
#target_link_libraries( test_lines.exe
  #_RADIATIVE_TRANSFER
#_TEST_RT_MAIN )
#
## Create executable to test species
#add_executable( test_species.exe
#test_species.cpp )
#target_link_libraries( test_species.exe
#_RADIATIVE_TRANSFER
#_TEST_RT_MAIN )
#
## Create executable to test RadiativeTransfer
#add_executable( test_RadiativeTransfer.exe
#test_RadiativeTransfer.cpp )
#target_link_libraries( test_RadiativeTransfer.exe
#_RADIATIVE_TRANSFER
#_LINES
#_TEST_RT_MAIN )
