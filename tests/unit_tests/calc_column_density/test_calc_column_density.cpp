// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <string>
#include <iostream>

#include "catch.hpp"

#include "../../../parameters.hpp"
#include "../../../src/declarations.hpp"
#include "../../../src/definitions.hpp"

#include "../../../src/initializers.hpp"
#include "../../../src/read_input.hpp"
#include "../../../src/ray_tracing.hpp"
#include "../../../src/species_tools.hpp"
#include "../../../src/read_chemdata.hpp"
#include "../../../src/calc_column_density.hpp"


#define EPS 1.0E-7


TEST_CASE("calc_column_density")
{

  // SET UP TEST
  // ___________


  // Since executables are in directory /tests, we have to change paths

  std::string test_inputfile     = "../../../" + inputfile;
  std::string test_spec_datafile = "../../../" + spec_datafile;


  // Read input file

  CELL cell[NCELLS];

  read_txt_input (test_inputfile, NCELLS, cell);


  // Find neighbors

  initialize_cells (NCELLS, cell);

  find_neighbors (NCELLS, cell);


  // Read species file

  SPECIES species[NSPEC];

  read_species (test_spec_datafile, species);


  // Define and initialize

  double column_tot[NCELLS*NRAYS];   // total column density

  initialize_double_array (NCELLS*NRAYS, column_tot);




  // TESTS
  // _____

  calc_column_density (NCELLS, cell, column_tot, NSPEC-1);


  CHECK( 1==1 );

}
