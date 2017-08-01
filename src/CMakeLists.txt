add_subdirectory(HEALPix)
add_subdirectory(sundials)

add_library( 3DRT_library abundance.cpp data_tools.cpp species_tools.cpp setup_data_structures.cpp  read_input.cpp read_chemdata.cpp read_linedata.cpp create_healpixvectors.cpp ray_tracing.cpp heapsort.cpp reaction_rates.cpp rate_calculations.cpp rate_calculations_radfield.cpp spline.cpp level_populations.cpp calc_C_coeff.cpp radiative_transfer.cpp exact_feautrier.cpp level_population_solver.cpp column_density_calculator.cpp UV_field_calculator.cpp write_output.cpp )

TARGET_LINK_LIBRARIES(3DRT_library HEALPix_library)
TARGET_LINK_LIBRARIES(3DRT_library sundials_library)
