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


TEST_CASE ("Cell structure for 5x5, 2D grid")
{


  // SET UP TEST
  // ___________


  // Define grid input file

  std::string test_inputfile = "../../../input/files/tests/grid_2D_test_25.txt";


  /* Layout of the test grid:

     00 01 02 03 04
     05 06 07 08 09
     10 11 12 13 14
     15 16 17 18 19
     20 21 22 23 24

  */


  // Read input file

  long ncells = get_NCELLS_txt (test_inputfile);

  CELL *cell = new CELL[ncells];

  initialize_cells (ncells, cell);

  read_txt_input (test_inputfile, ncells, cell);


  // Find neighbors and endpoints

  find_neighbors (ncells, cell);
  find_endpoints (ncells, cell);


  // Check number of neighbors

  CHECK (cell[0].n_neighbors  == 3);
  CHECK (cell[4].n_neighbors  == 3);
  CHECK (cell[20].n_neighbors == 3);
  CHECK (cell[24].n_neighbors == 3);

  CHECK (cell[1].n_neighbors  == 5);
  CHECK (cell[2].n_neighbors  == 5);
  CHECK (cell[3].n_neighbors  == 5);
  CHECK (cell[5].n_neighbors  == 5);
  CHECK (cell[9].n_neighbors  == 5);
  CHECK (cell[10].n_neighbors == 5);
  CHECK (cell[14].n_neighbors == 5);
  CHECK (cell[15].n_neighbors == 5);
  CHECK (cell[19].n_neighbors == 5);
  CHECK (cell[21].n_neighbors == 5);
  CHECK (cell[22].n_neighbors == 5);
  CHECK (cell[23].n_neighbors == 5);

  CHECK (cell[11].n_neighbors == 8);

  CHECK (cell[11].neighbor[0] == 16);
  CHECK (cell[11].neighbor[1] == 17);
  CHECK (cell[11].neighbor[2] == 12);
  CHECK (cell[11].neighbor[3] ==  7);
  CHECK (cell[11].neighbor[4] ==  6);
  CHECK (cell[11].neighbor[5] ==  5);
  CHECK (cell[11].neighbor[6] == 10);
  CHECK (cell[11].neighbor[7] == 15);
  CHECK (cell[11].neighbor[8] ==  8);


  double Z  = 0.0;
  double dZ = 0.0;

  CHECK ( next_cell (ncells, cell, 11, 1, &Z, 11, &dZ) ==  17);



  // for (long c = 0; c < ncells; c++)
  // {
  //   printf("%ld\n", cell[c].n_neighbors);
  //   printf("%lE  %lE  %lE\n", cell[c].x, cell[c].y, cell[c].z);
  //
  //   for (long n = 0; n < cell[c].n_neighbors; n++)
  //   {
  //     printf("cell %ld has neighbors %ld\n", c, cell[c].neighbor[n]);
  //   }
  // }

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

  // long o   = 8;
  // long ray = 1;
  //
  //   std::cout << cell[o].Z[ray] << "\n";
  //   std::cout << cell[o].endpoint[ray] << "\n";
  //
  // double dZ = 0.0;
  // double Z  = 0.0;
  //
  //   std::cout << previous_cell (ncells, cell, o, ray, &Z, o, &dZ) << "\n";
  //



  CHECK (true);


  delete [] cell;

}
