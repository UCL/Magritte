# CMakeLists.txt for Magritte (/src directory)
# --------------------------------------------


# VTK includes
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# Add compiler flags
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS
  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -mcmodel=medium")
  set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -mcmodel=medium")
  set(CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -mcmodel=medium")
endif()

# Add the two subdirectories in /src
add_subdirectory(HEALPix)
add_subdirectory(sundials)

# Add the folder containing Eigen directory so the header files can be found
include_directories(.)

# Add the folder containing the sundials headers
include_directories(sundials/include)

# Create a library containing all source code in /src except the main (Magritte.cpp)
add_library(_MAGRITTE
../setup/setup_data_tools.cpp
initializers.cpp
chemistry.cpp
species_tools.cpp
read_input.cpp
read_chemdata.cpp
read_linedata.cpp
ray_tracing.cpp
heapsort.cpp
reduce.cpp
bound.cpp
reaction_rates.cpp
calc_reac_rates.cpp
calc_reac_rates_rad.cpp
radfield_tools.cpp
spline.cpp
thermal_balance.cpp
thermal_balance_iteration.cpp
calc_LTE_populations.cpp
level_populations_otf.cpp
cell_level_populations.cpp
lines.cpp
acceleration_Ng.cpp
calc_C_coeff.cpp
sobolev.cpp
cell_sobolev.cpp
radiative_transfer_otf.cpp
cell_radiative_transfer.cpp
feautrier.cpp
cell_feautrier.cpp
level_population_solver.cpp
calc_rad_surface.cpp
calc_column_density.cpp
calc_AV.cpp
calc_UV_field.cpp
calc_temperature_dust.cpp
heating.cpp
cooling.cpp
update_temperature_gas.cpp
write_output.cpp
write_txt_tools.cpp
write_vtu_tools.cpp)


# Link the libraries of the subdirectories to this library
target_link_libraries(_MAGRITTE _HEALPix)
target_link_libraries(_MAGRITTE _sundials)
target_link_libraries(_MAGRITTE ${VTK_LIBRARIES})


# Create a library
add_library(_IO
../setup/setup_data_tools.cpp
ray_tracing.cpp
heapsort.cpp
read_input.cpp
initializers.cpp
write_output.cpp
write_txt_tools.cpp
write_vtu_tools.cpp
reduce.cpp
bound.cpp)

# Link the libraries of the subdirectories to this library
target_link_libraries(_IO _HEALPix)
target_link_libraries(_IO ${VTK_LIBRARIES})


# Create a library
add_library(_O
read_input.cpp
initializers.cpp
write_output.cpp
write_txt_tools.cpp
write_vtu_tools.cpp)

# Link the libraries of the subdirectories to this library
target_link_libraries(_O _HEALPix)
target_link_libraries(_O ${VTK_LIBRARIES})
