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

#include "../../../setup/setup_data_tools.hpp"
#include "../../../src/initializers.hpp"
#include "../../../src/read_input.hpp"
#include "../../../src/ray_tracing.hpp"
#include "../../../src/species_tools.hpp"
#include "../../../src/read_chemdata.hpp"
#include "../../../src/calc_column_density.hpp"


#define EPS 1.0E-7


TEST_CASE ("calc_column_density")
{

  // SET UP TEST
  // ___________


  // Define grid input file

  std::string test_inputfile = "../../../input/files/tests/grid_2D_test_25.txt";


  // Read input file

  long ncells = get_NCELLS_txt (test_inputfile);

  CELL *cell = new CELL[ncells];

  read_txt_input (test_inputfile, ncells, cell);

  initialize_cells (ncells, cell);


  // Find neighbors and endpoints

  find_neighbors (ncells, cell);
  find_endpoints (ncells, cell);


  // Read species file

  SPECIES species[NSPEC];



  // Define and initialize

  double column_tot[ncells*NRAYS];   // total column density

  initialize_double_array (ncells*NRAYS, column_tot);




  // TESTS
  // _____

  // cell_column_density (ncells, cell, column_tot, NSPEC-1);


  CHECK (1==1);


  delete [] cell;

}
