// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <string>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "../../../../src/declarations.hpp"
#include "../../../../src/definitions.hpp"
#include "../../../../src/initializers.hpp"
#include "../../../../src/ray_tracing.hpp"

#define EPS 1.0E-5


TEST_CASE ("Visually inspect 5x5 2D grid: boundary cube")
{


  // SET UP TEST
  // ___________


  // Define grid input file


  /* Layout of the test grid:

     +---> x
     |
     v  00 05 10 15 20
     y  01 06 11 16 21
        02 07 12 17 22
        03 08 13 18 23
        04 09 14 19 24

  */

  // Create rays

  const RAYS rays;


  // Read input file

  long ncells = NCELLS_INIT;

  CELLS Cells (NCELLS_INIT);

  CELLS *cells = &Cells;

  initialize_cells (ncells, cells);

  cells->read_input (inputfile);


  // Find neighbors

  find_neighbors (ncells, cells, rays);


  // Check number of neighbors

  CHECK (cells->n_neighbors[0]  == 3);
  CHECK (cells->n_neighbors[4]  == 3);
  CHECK (cells->n_neighbors[20] == 3);
  CHECK (cells->n_neighbors[24] == 3);

  CHECK (cells->n_neighbors[1]  == 5);
  CHECK (cells->n_neighbors[2]  == 5);
  CHECK (cells->n_neighbors[3]  == 5);
  CHECK (cells->n_neighbors[5]  == 5);
  CHECK (cells->n_neighbors[9]  == 5);
  CHECK (cells->n_neighbors[10] == 5);
  CHECK (cells->n_neighbors[14] == 5);
  CHECK (cells->n_neighbors[15] == 5);
  CHECK (cells->n_neighbors[19] == 5);
  CHECK (cells->n_neighbors[21] == 5);
  CHECK (cells->n_neighbors[22] == 5);
  CHECK (cells->n_neighbors[23] == 5);

  CHECK (cells->n_neighbors[11] == 8);

  CHECK (cells->neighbor[RINDEX(11,0)] == 16);
  CHECK (cells->neighbor[RINDEX(11,1)] == 17);
  CHECK (cells->neighbor[RINDEX(11,2)] == 12);
  CHECK (cells->neighbor[RINDEX(11,3)] ==  7);
  CHECK (cells->neighbor[RINDEX(11,4)] ==  6);
  CHECK (cells->neighbor[RINDEX(11,5)] ==  5);
  CHECK (cells->neighbor[RINDEX(11,6)] == 10);
  CHECK (cells->neighbor[RINDEX(11,7)] == 15);


  long origin = 2;
  long r      = 1;

  double Z  = 0.0;
  double dZ = 0.0;

  long current = origin;
  long next    = next_cell (NCELLS, cells, rays, origin, r, &Z, current, &dZ);

  // printf("current %ld,   next %ld\n", current, next);


  while (next != NCELLS)
  {
    current = next;
    next    = next_cell (NCELLS, cells, rays, origin, r, &Z, current, &dZ);

    // printf("current %ld,   next %ld\n", current, next);
  }

}
