# CMakeLists.txt for test
#########################

# Include directory containing source
include_directories (../include)

# Create main lib (as advised in Catch docs, to speed up compilation)
add_library (TestMain
  main.cpp)

# Create executable to test input
add_executable (test_input.exe
  test_input.cpp)
target_link_libraries (test_input.exe
  Magritte
  TestMain)

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
