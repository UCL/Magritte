# CMakeLists.txt for Magritte (/src directory)
# -----------------------------------------



# Add the two subdirectories in /src

add_subdirectory(HEALPix)
add_subdirectory(sundials)


# Add the folder containing Eigen directory so the header files can be found

include_directories( . )


# Add the folder containing the sundials headers

include_directories( sundials/include )


# Create a library containing all source code in /src except the main (Magritte.cpp)

add_library( _MAGRITTE
initializers.cpp
chemistry.cpp
data_tools.cpp
species_tools.cpp
setup_data_structures.cpp
read_input.cpp
read_chemdata.cpp
read_linedata.cpp
create_healpixvectors.cpp
ray_tracing.cpp
heapsort.cpp
reaction_rates.cpp
calc_reac_rates.cpp
calc_reac_rates_rad.cpp
radfield_tools.cpp
spline.cpp
calc_LTE_populations.cpp
level_populations.cpp
calc_C_coeff.cpp
radiative_transfer.cpp
exact_feautrier.cpp
level_population_solver.cpp
calc_rad_surface.cpp
calc_column_density.cpp
calc_AV.cpp
calc_UV_field.cpp
calc_temperature_dust.cpp
heating.cpp
cooling.cpp
update_temperature_gas.cpp
write_output.cpp )


# Link the libraries of the subdirectories to this library

target_link_libraries( _MAGRITTE   _HEALPix )
target_link_libraries( _MAGRITTE   _sundials )
