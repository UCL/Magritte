// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <sstream>
#include <string>


#include "declarations.hpp"
#include "definitions.hpp"

#include "initializers.hpp"
#include "read_input.hpp"
#include "ray_tracing.hpp"
#include "reduce.hpp"
#include "bound.cpp"
#include "write_txt_tools.hpp"
#include "write_vtu_tools.hpp"

#include "../setup/setup_data_tools.hpp"


int main ()
{

  // Define and initialize cells

# if   (INPUT_FORMAT == '.txt')

    long ncells = get_NCELLS_txt (inputfile);

    CELL *cell = new CELL[ncells];

    initialize_cells (ncells, cell);

    read_txt_input (inputfile, ncells, cell);

# elif (INPUT_FORMAT == '.vtu')

    long ncells = get_NCELLS_vtu (inputfile);

    CELL *cell = new CELL[ncells];

    initialize_cells (ncells, cell);

    read_vtu_input (inputfile, ncells, cell);

# endif


  // Place boundary

  long nboundary_cells = 500;

  CELL *cell_bound = new CELL[ncells+nboundary_cells];

  bound_sphere (ncells, cell, cell_bound, nboundary_cells);


  // Find neighboring cells for each cell

  find_neighbors (ncells, cell);


  // Reduce grid

  long ncells_red1 = reduce (ncells, cell);
  CELL *cell_red1 = new CELL[ncells_red1];
  initialize_reduced_grid (ncells_red1, cell_red1, ncells, cell);


  long ncells_red2 = reduce (ncells_red1, cell_red1);
  CELL *cell_red2 = new CELL[ncells_red2];
  initialize_reduced_grid (ncells_red2, cell_red2, ncells_red1, cell_red1);


  long ncells_red3 = reduce (ncells_red2, cell_red2);
  CELL *cell_red3 = new CELL[ncells_red3];
  initialize_reduced_grid (ncells_red3, cell_red3, ncells_red2, cell_red2);


  long ncells_red4 = reduce (ncells_red3, cell_red3);
  CELL *cell_red4 = new CELL[ncells_red4];
  initialize_reduced_grid (ncells_red4, cell_red4, ncells_red3, cell_red3);


  long ncells_red5 = reduce (ncells_red4, cell_red4);
  CELL *cell_red5 = new CELL[ncells_red5];
  initialize_reduced_grid (ncells_red5, cell_red5, ncells_red4, cell_red4);




  FILE *file = fopen("src/grid_sizes.hpp", "w");

  fprintf (file, "#define NCELLS_RED1 %d\n", ncells_red1);
  fprintf (file, "#define NCELLS_RED2 %d\n", ncells_red2);
  fprintf (file, "#define NCELLS_RED3 %d\n", ncells_red3);
  fprintf (file, "#define NCELLS_RED4 %d\n", ncells_red4);
  fprintf (file, "#define NCELLS_RED5 %d\n", ncells_red5);

  fclose (file);


  // Define full grid
//
//   long size_x = 2;
//   long size_y = 0;
//   long size_z = 0;
//
//
// # if   (DIMENSIONS == 1)
//
//     long n_extra = 2;
//
// # elif (DIMENSIONS == 2)
//
//     long n_extra = 2*(size_x + size_y);
//
// # elif (DIMENSIONS == 3)
//
//     long n_extra = 2*(size_x*size_z + size_y*size_z + size_x*size_y + 1);
//
// # endif
//
//
//   long ncells_full = ncells_red + n_extra;
//
//   CELL *cell_full = new CELL[ncells_full];
//
//   initialize_cells (ncells_full, cell_full);
//
//
//   // Add boundary
//
//   bound_cube (ncells_red, cell_red, cell_full, size_x, size_y, size_z);
//



  // APPLY MAGRITTE ...



  // Interpolate reduced grid back to original grid

  // interpolate (ncells_red, cell_red, ncells, cell)
  //
  // delete [] cell_red;

  //
  // // write reduced grid as .txt file and .vtu file
  //
  // std::cout << "  Writing .txt grid...\n";
  //
  // std::ostringstream strs;
  // strs << threshold;
  // std::string thres = strs.str();
  //



  delete [] cell;


  return (0);

}
