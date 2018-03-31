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
#include "../../../../src/bound.hpp"

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

  cells->initialize ();

  cells->read_input (inputfile);


  // Define full grid

  long size_x = 6;
  long size_y = 6;
  long size_z = 0;

  long n_extra = 2*(size_x + size_y);   // number of boundary cells

  long ncells_full = ncells + n_extra;

  CELLS Cells_full (ncells_full);

  CELLS *cells_full = &Cells_full;

  cells_full->initialize ();


  // Add boundary

  bound_cube (cells, cells_full, size_x, size_y, size_z);


  // Find neighbors and endpoints

  find_neighbors (ncells_full, cells_full, rays);
  find_endpoints (ncells_full, cells_full, rays);


  // Check whether boundary cells

  for (int p = 0; p < ncells_full; p++)
  {
    if (cells_full->boundary[p])
    {
      for (long n = 0; n < cells_full->n_neighbors[p]; n++)
      {
        long neighbor = cells_full->neighbor[RINDEX(p,n)];
        printf("%ld %ld\n", p, neighbor);
      }
    }
  }



  // Check number of neighbors

  CHECK (cells_full->n_neighbors[0]  == 8);
  CHECK (cells_full->n_neighbors[4]  == 8);
  CHECK (cells_full->n_neighbors[20] == 8);
  CHECK (cells_full->n_neighbors[24] == 8);

  CHECK (cells_full->n_neighbors[1]  == 8);
  CHECK (cells_full->n_neighbors[2]  == 8);
  CHECK (cells_full->n_neighbors[3]  == 8);
  CHECK (cells_full->n_neighbors[5]  == 8);
  CHECK (cells_full->n_neighbors[9]  == 8);
  CHECK (cells_full->n_neighbors[10] == 8);
  CHECK (cells_full->n_neighbors[14] == 8);
  CHECK (cells_full->n_neighbors[15] == 8);
  CHECK (cells_full->n_neighbors[19] == 8);
  CHECK (cells_full->n_neighbors[21] == 8);
  CHECK (cells_full->n_neighbors[22] == 8);
  CHECK (cells_full->n_neighbors[23] == 8);

  CHECK (cells_full->n_neighbors[11] == 8);

  CHECK (cells_full->neighbor[RINDEX(11,0)] == 16);
  CHECK (cells_full->neighbor[RINDEX(11,1)] == 17);
  CHECK (cells_full->neighbor[RINDEX(11,2)] == 12);
  CHECK (cells_full->neighbor[RINDEX(11,3)] ==  7);
  CHECK (cells_full->neighbor[RINDEX(11,4)] ==  6);
  CHECK (cells_full->neighbor[RINDEX(11,5)] ==  5);
  CHECK (cells_full->neighbor[RINDEX(11,6)] == 10);
  CHECK (cells_full->neighbor[RINDEX(11,7)] == 15);


  long origin = 2;
  long r      = 1;

  double Z  = 0.0;
  double dZ = 0.0;

  printf("ncells = %ld,   ncells_full = %ld\n", ncells, ncells_full);

  long current = origin;
  long next    = next_cell (ncells_full, cells_full, rays, origin, r, &Z, current, &dZ);

  printf("current %ld,   next %ld\n", current, next);

  int count = 0;

  while (next != ncells_full && (count < 100))
  {
    current = next;
    next    = next_cell (ncells_full, cells_full, rays, origin, r, &Z, current, &dZ);

    printf("current %ld,   next %ld\n", current, next);

    count++;
  }

/*
  NOTE
    There is a problem when a boundary point has non-boundary neighbours which can be projected on the ray. Temporary solution is to use a boundary which is "denser" than the interior grid.
*/


  // long wat = next_cell (ncells_full, cells_full, rays, origin, r, &Z, current, &dZ);
  //
  // printf("%ld\n", wat);

}
