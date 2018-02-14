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
#include "../../../src/reduce.hpp"
#include "../../../src/bound.hpp"
#include "../../../src/ray_tracing.hpp"
#include "../../../src/write_txt_tools.hpp"

#define EPS 1.0E-5


TEST_CASE ("Visually inspect 5x5 2D grid: boundary cube")
{


  // SET UP TEST
  // ___________


  // Define grid input file

  std::string test_inputfile = "input/files/grid_2D_test_25.txt";


  /* Layout of the test grid:

     00 01 02 03 04
     05 06 07 08 09
     10 11 12 13 14
     15 16 17 18 19
     20 21 22 23 24

  */


  // Read input file

  long ncells_init = get_NCELLS_txt (test_inputfile);

  CELL *cell_init = new CELL[ncells_init];

  initialize_cells (ncells_init, cell_init);

  read_txt_input (test_inputfile, ncells_init, cell_init);


  // Define full grid

  long size_x = 6;
  long size_y = 6;
  long size_z = 2;


# if   (DIMENSIONS == 2)

  long n_extra = 2*(size_x + size_y);   // number of boundary cells

# elif (DIMENSIONS == 3)

  long n_extra = 2*(size_x*size_z + size_y*size_z + size_x*size_y + 1);   // number of boundary cells

# endif


  long ncells_full = ncells_init+n_extra;

  CELL *cell_full = new CELL[ncells_full];

  initialize_cells (ncells_full, cell_full);


  // Add boundary

  bound_cube (ncells_init, cell_init, cell_full, size_x, size_y, size_z);


  // Write grid

  // write_grid ("", ncells_init, cell_init);

  // write_grid ("full", ncells_full, cell_full);


  delete [] cell_init;
  delete [] cell_full;

}




TEST_CASE ("Visually inspect 5x4 2D grid: boundary cube")
{

  // SET UP TEST
  // ___________


  // Define grid input file

  std::string test_inputfile = "input/files/grid_2D_test_6.txt";


  /* Layout of the test grid:

     00 01 02 03 04
     05 06 07 08 09
     10 11 12 13 14
     15 16 17 18 19

  */


  // Read input file

  long ncells_init = get_NCELLS_txt (test_inputfile);

  CELL *cell_init = new CELL[ncells_init];

  initialize_cells (ncells_init, cell_init);

  read_txt_input (test_inputfile, ncells_init, cell_init);


  // Define full grid

  long size_x = 3;
  long size_y = 3;
  long size_z = 2;


# if  (DIMENSIONS == 2)

  long n_extra = 2*(size_x + size_y);   // number of boundary cells

# elif (DIMENSIONS == 3)

  long n_extra = 2*(size_x*size_z + size_y*size_z + size_x*size_y + 1);   // number of boundary cells

# endif

  long ncells_full = ncells_init+n_extra;

  CELL *cell_full = new CELL[ncells_full];

  initialize_cells (ncells_full, cell_full);


  // Add boundary

  bound_cube (ncells_init, cell_init, cell_full, size_x, size_y, size_z);


  // Write grid

  // write_grid ("", ncells_init, cell_init);

  // write_grid ("full", ncells_full, cell_full);


  delete [] cell_init;
  delete [] cell_full;

}




TEST_CASE ("Visually inspect grid: boundary circle")
{

  // SET UP TEST
  // ___________


  // Define grid input file

  std::string test_inputfile = "input/files/grid_2D_test_25.txt";


  /* Layout of the test grid:

     00 01 02 03 04
     05 06 07 08 09
     10 11 12 13 14
     15 16 17 18 19
     20 21 22 23 24

  */


  // Read input file

  long ncells_init = get_NCELLS_txt (test_inputfile);

  CELL *cell_init = new CELL[ncells_init];

  initialize_cells (ncells_init, cell_init);

  read_txt_input (test_inputfile, ncells_init, cell_init);


  // Define full grid

  long n_extra = 12*2*2*2*2;   // number of boundary cells

  long ncells_full = ncells_init+n_extra;

  CELL *cell_full = new CELL[ncells_full];

  initialize_cells (ncells_full, cell_full);


  // Add boundary

  bound_sphere (ncells_init, cell_init, cell_full, n_extra);


  // Write grid

  // write_grid ("", ncells_init, cell_init);

  // write_grid ("full", ncells_full, cell_full);


  delete [] cell_init;
  delete [] cell_full;

}




TEST_CASE ("Neighbor structure for 5x5, 2D grid")
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


  // Find neighbors

  find_neighbors (ncells, cell);


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


  long origin = 2;
  long r      = 1;

  double Z  = 0.0;
  double dZ = 0.0;

  long current = origin;
  long next    = next_cell (NCELLS, cell, origin, r, &Z, current, &dZ);

  // printf("current %ld,   next %ld\n", current, next);


  while (next != NCELLS)
  {
    current = next;
    next    = next_cell (NCELLS, cell, origin, r, &Z, current, &dZ);

    // printf("current %ld,   next %ld\n", current, next);
  }


  delete [] cell;

}



TEST_CASE ("Neighbor structure + boundary for 5x5, 2D grid")
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


# if   (DIMENSIONS == 2)

  long n_extra = 2*(size_x + size_y);   // number of boundary cells

# elif (DIMENSIONS == 3)

  long n_extra = 2*(size_x*size_z + size_y*size_z + size_x*size_y + 1);   // number of boundary cells

# endif


  long ncells_full = ncells+n_extra;

  CELL *cell_full = new CELL[ncells_full];

  initialize_cells (ncells_full, cell_full);


  // Add boundary

  bound_cube (ncells, cell, cell_full, size_x, size_y, size_z);



  // Find neighbors and endpoints

  find_neighbors (ncells_full, cell_full);
  find_endpoints (ncells_full, cell_full);




  // TESTS
  // _____


  // Check number of neighbors

  CHECK (cell_full[0].n_neighbors  == 8);
  CHECK (cell_full[4].n_neighbors  == 8);
  CHECK (cell_full[20].n_neighbors == 8);
  CHECK (cell_full[24].n_neighbors == 8);

  CHECK (cell_full[1].n_neighbors  == 8);
  CHECK (cell_full[2].n_neighbors  == 8);
  CHECK (cell_full[3].n_neighbors  == 8);
  CHECK (cell_full[5].n_neighbors  == 8);
  CHECK (cell_full[9].n_neighbors  == 8);
  CHECK (cell_full[10].n_neighbors == 8);
  CHECK (cell_full[14].n_neighbors == 8);
  CHECK (cell_full[15].n_neighbors == 8);
  CHECK (cell_full[19].n_neighbors == 8);
  CHECK (cell_full[21].n_neighbors == 8);
  CHECK (cell_full[22].n_neighbors == 8);
  CHECK (cell_full[23].n_neighbors == 8);

  CHECK (cell_full[11].n_neighbors == 8);

  CHECK (cell_full[11].neighbor[0] == 16);
  CHECK (cell_full[11].neighbor[1] == 17);
  CHECK (cell_full[11].neighbor[2] == 12);
  CHECK (cell_full[11].neighbor[3] ==  7);
  CHECK (cell_full[11].neighbor[4] ==  6);
  CHECK (cell_full[11].neighbor[5] ==  5);
  CHECK (cell_full[11].neighbor[6] == 10);
  CHECK (cell_full[11].neighbor[7] == 15);
  CHECK (cell_full[11].neighbor[8] ==  8);


  // double Z  = 0.0;
  // double dZ = 0.0;
  //
  // CHECK ( next_cell (ncells, cell, 11, 1, &Z, 11, &dZ) ==  17);



  {
    long origin = 1;
    long ar     = 0;

    double Z  = cell_full[origin].Z[ar];
    double dZ = 0.0;

    // for (long c = 0; c < ncells_full; c++)
    // {
    //   for (long r = 0; r < NRAYS; r++)
    //   {
    //     printf("endpoint (%ld,%ld) is %ld    %lE\n", c, r ,cell_full[c].endpoint[r], cell_full[c].Z[r]);
    //   }
    // }

    long current  = cell_full[origin].endpoint[ar];
    long previous = previous_cell (ncells_full, cell_full, origin, ar, &Z, current, &dZ);


    CHECK (current == 35);

    current  = previous;
    previous = previous_cell (ncells_full, cell_full, origin, ar, &Z, current, &dZ);

    CHECK (current == 21);

    current  = previous;
    previous = previous_cell (ncells_full, cell_full, origin, ar, &Z, current, &dZ);

    CHECK (current == 16);

    current  = previous;
    previous = previous_cell (ncells_full, cell_full, origin, ar, &Z, current, &dZ);

    CHECK (current == 11);

    current  = previous;
    previous = previous_cell (ncells_full, cell_full, origin, ar, &Z, current, &dZ);

    CHECK (current ==  6);

    current  = previous;
    previous = previous_cell (ncells_full, cell_full, origin, ar, &Z, current, &dZ);

    CHECK (current ==  1);
  }


  {
    long origin = 1;
    long r      = 7;

    double Z  = 0.0;
    double dZ = 0.0;

    long current = origin;
    long next    = next_cell (ncells_full, cell_full, origin, r, &Z, current, &dZ);

    CHECK (current == 1);

    current = next;
    next    = next_cell (ncells_full, cell_full, origin, r, &Z, current, &dZ);

    CHECK (current == 5);

    current = next;
    next    = next_cell (ncells_full, cell_full, origin, r, &Z, current, &dZ);

    CHECK (current == 46);

    current = next;
    next    = next_cell (ncells_full, cell_full, origin, r, &Z, current, &dZ);

    CHECK (current == 49);
  }

  delete [] cell;
  delete [] cell_full;

}




TEST_CASE ("Visually inspect 5x5 2D reduced grid: boundary cube")
{

  // SET UP TEST
  // ___________


  // Define grid input file

  std::string test_inputfile = "input/files/grid_2D_test_25.txt";


  /* Layout of the test grid:

     00 01 02 03 04
     05 06 07 08 09
     10 11 12 13 14
     15 16 17 18 19
     20 21 22 23 24

  */


  // Read input file

  long ncells_init = get_NCELLS_txt (test_inputfile);

  CELL *cell_init = new CELL[ncells_init];

  initialize_cells (ncells_init, cell_init);

  read_txt_input (test_inputfile, ncells_init, cell_init);


  // Find neighbors (needed in reduction)

  find_neighbors (ncells_init, cell_init);


  // Reduce grid

  double x_min =  0.0E+00;
  double x_max =  5.0E+00;
  double y_min =  0.0E+00;
  double y_max =  5.0E+00;
  double z_min = -1.0E+00;
  double z_max =  1.0E+00;

  double min_density_change = 0.0;

  long ncells_red = reduce (ncells_init, cell_init, min_density_change, x_min, x_max, y_min, y_max, z_min, z_max);


  // Define the reduced grid

  CELL *cell_red = new CELL[ncells_red];

  initialize_cells (ncells_red, cell_red);

  initialize_reduced_grid (ncells_red, cell_red, ncells_init, cell_init);


  // Define full grid

  long size_x = 9;
  long size_y = 9;
  long size_z = 0;


# if   (DIMENSIONS == 1)

    long n_extra = 2;

# elif (DIMENSIONS == 2)

    long n_extra = 2*(size_x + size_y);

# elif (DIMENSIONS == 3)

    long n_extra = 2*(size_x*size_z + size_y*size_z + size_x*size_y + 1);

# endif


  long ncells_full = ncells_red + n_extra;

  CELL *cell_full = new CELL[ncells_full];

  initialize_cells (ncells_full, cell_full);


  // Add boundary

  bound_cube (ncells_red, cell_red, cell_full, size_x, size_y, size_z);


  // Write grid

  write_grid ("", ncells_red, cell_red);

  write_grid ("full", ncells_full, cell_full);


  delete [] cell_init;
  delete [] cell_red;
  delete [] cell_full;

}
