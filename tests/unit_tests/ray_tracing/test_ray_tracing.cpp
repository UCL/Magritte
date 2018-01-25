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
#include "../../../src/Magritte_config.hpp"
#include "../../../src/declarations.hpp"
#include "../../../src/definitions.hpp"

#include "../../../setup/setup_data_tools.hpp"
#include "../../../src/initializers.hpp"
#include "../../../src/read_input.hpp"
#include "../../../src/ray_tracing.hpp"
#include "../../../src/write_txt_tools.hpp"

#define EPS 1.0E-5


TEST_CASE ("Cell structure")
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


  // for (long c = 0; c < ncells; c++)
  // {
  //   printf("%ld\n", cell[c].n_neighbors);
  //
  //   for (long n = 0; n < cell[c].n_neighbors; n++)
  //   {
  //     printf("cell %ld has neighbors %ld\n", c, cell[c].neighbor[n]);
  //   }
  // }
  //
  // write_healpixvectors("");
  //
  // double dZ = 0.0;
  //
  // long origin = 1;
  // long ray    = 1;
  //
  // double Z = 0.0;
  //
  // long current = origin;
  // long next    = next_cell (ncells, cell, origin, ray, &Z, current, &dZ);
  //
  //
  //
  // while (next != ncells)
  // {
  //   printf("current %ld, next %ld, Z %lE\n", current, next, Z);
  //
  //   current = next;
  //   next    = next_cell (ncells, cell, origin, ray, &Z, current, &dZ);
  // }

  //
  // long origin = 0;
  // long ray    = 5;
  //
  // for (long o = 0; o < ncells; o++){
  //   std::cout << cell[o].Z[ray] << "\n";
  //   std::cout << cell[o].endpoint[ray] << "\n";
  // }

  long o   = 8;
  long ray = 1;

    std::cout << cell[o].Z[ray] << "\n";
    std::cout << cell[o].endpoint[ray] << "\n";

  double dZ = 0.0;
  double Z  = 0.0;

    std::cout << previous_cell (ncells, cell, o, ray, &Z, o, &dZ) << "\n";




  CHECK (true);


  delete [] cell;

}
