# CMakeLists.txt for 3D-RT (/src directory)
# -----------------------------------------



# Add the two subdirectories in /src

add_subdirectory(HEALPix)
add_subdirectory(sundials)


# Add the Eigen directory so the header files can be found

include_directories(/home/frederik/Dropbox/Astro/3D-RT/src)


# Create a library containing all source code in /src except the main (3D-RT.cpp)

add_library( _3DRT initializers.cpp abundances.cpp data_tools.cpp species_tools.cpp setup_data_structures.cpp  read_input.cpp read_chemdata.cpp read_linedata.cpp create_healpixvectors.cpp ray_tracing.cpp heapsort.cpp reaction_rates.cpp rate_calculations.cpp rate_calculations_radfield.cpp radfield_tools.cpp spline.cpp level_populations.cpp calc_C_coeff.cpp radiative_transfer.cpp exact_feautrier.cpp level_population_solver.cpp rad_surface_calculator.cpp column_density_calculator.cpp AV_calculator.cpp UV_field_calculator.cpp dust_temperature_calculation.cpp heating.cpp cooling.cpp update_temperature_gas.cpp write_output.cpp )


# Link the libraries of the subdirectories to this library

target_link_libraries( _3DRT   _HEALPix )
target_link_libraries( _3DRT   _sundials )

