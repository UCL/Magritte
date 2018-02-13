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
#include "../../../src/bound.hpp"
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

  std::string test_inputfile = "input/files/grid_2D_test_25.txt";


  /* Layout of the test grid:

     +---> x
     |
     v  00 05 10 15 20
     y  01 06 11 16 21
        02 07 12 17 22
        03 08 13 18 23
        04 09 14 19 24

  */


  // Read input file

  long ncells = get_NCELLS_txt (test_inputfile);

  CELL *cell = new CELL[ncells];

  initialize_cells (ncells, cell);

  read_txt_input (test_inputfile, ncells, cell);


  // Define full grid

  long size_x = 6;
  long size_y = 6;
  long size_z = 0;


  long n_extra = 2*(size_x + size_y);   // number of boundary cells

  long ncells_full = ncells+n_extra;

  CELL *cell_full = new CELL[ncells_full];

  initialize_cells (ncells_full, cell_full);


  // Add boundary

  bound_cube (ncells, cell, cell_full, size_x, size_y, size_z);



  // Find neighbors and endpoints

  find_neighbors (ncells_full, cell_full);
  find_endpoints (ncells_full, cell_full);


  // Read species file

  SPECIES species[NSPEC];



  // Define and initialize

  double *column_tot = new double[ncells_full*NRAYS];   // total column density

  initialize_double_array (ncells_full*NRAYS, column_tot);


  // Fill cells

  for (long c = 0; c < ncells_full; c++)
  {
    for (int spec = 0; spec < NSPEC; spec++)
    {
      cell_full[c].abundance[spec] = 1.0;
    }
  }



  // TESTS
  // _____

  calc_column_density (ncells_full, cell_full, column_tot, NSPEC-1);

  printf("column density tot %lE\n", column_tot[RINDEX(2,1)]/PC);

  printf("column density tot 41 1 %lE\n", column_tot[RINDEX(41,1)]/PC);
  printf("column density tot 41 7 %lE\n", column_tot[RINDEX(41,7)]/PC);

  for (long c = 0; c < ncells_full; c++)
  {
    for (long r = 0; r < NRAYS; r++)
    {
      cell_full[c].ray[r].column = column_tot[RINDEX(c,r)];
    }
  }


  for (long c = 0; c < ncells_full; c++)
  {
    if (cell_full[c].mirror)
    {
      printf("TEST   %ld\n", c);
      for (long r = 0; r < NRAYS; r++)
      {
        cell_full[c].ray[r].column = cell_full[c].ray[mirror_xz[r]].column;
      }
    }
  }


  printf("column density tot 41 1 %lE\n", column_tot[RINDEX(41,1)]/PC);
  printf("column density tot 41 7 %lE\n", column_tot[RINDEX(41,7)]/PC);

  calc_column_density (ncells_full, cell_full, column_tot, NSPEC-1);

  printf("column density tot %lE\n", column_tot[RINDEX(2,1)]/PC);

  CHECK (1==1);


  delete [] cell;

}
