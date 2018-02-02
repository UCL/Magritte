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
#include "../../../src/bound.hpp"
#include "../../../src/ray_tracing.hpp"
#include "../../../src/write_txt_tools.hpp"

#define EPS 1.0E-5


TEST_CASE ("Visually inspect grid: boundary cube")
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

  long ncells_init = get_NCELLS_txt (test_inputfile);

  CELL *cell_init = new CELL[ncells_init];

  initialize_cells (ncells_init, cell_init);

  read_txt_input (test_inputfile, ncells_init, cell_init);


  // Define full grid

  long size_x = 9;
  long size_y = 7;
  long size_z = 0;


  long n_extra = 2*(size_x + size_y);   // number of boundary cells

  long ncells_full = ncells_init+n_extra;

  CELL *cell_full = new CELL[ncells_full];

  initialize_cells (ncells_full, cell_full);


  // Add boundary

  bound_cube (ncells_init, cell_init, cell_full, size_x, size_y, size_z);


  // Write grid

  write_grid ("", ncells_init, cell_init);

  write_grid ("full", ncells_full, cell_full);

}




TEST_CASE ("Visually inspect grid: boundary circle")
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

  write_grid ("", ncells_init, cell_init);

  write_grid ("full", ncells_full, cell_full);

}




TEST_CASE ("Neighbor structure for 5x5, 2D grid")
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


  delete [] cell;

}
