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

  std::string grid_init = GRID_INIT;


# if   (INPUT_FORMAT == '.txt')

    long ncells = get_NCELLS_txt (grid_init);

# elif (INPUT_FORMAT == '.vtu')

    long ncells = get_NCELLS_vtu (grid_init);

# endif


  CELL *cell = new CELL[ncells];

  initialize_cells (ncells, cell);


  // Read input grid

# if   (INPUT_FORMAT == '.txt')

    read_txt_input (grid_init, ncells, cell);

# elif (INPUT_FORMAT == '.vtu')

    read_vtu_input (grid_init, ncells, cell);

# endif


  // Find neighboring cells for each cell

  find_neighbors (ncells, cell);


  // read_neighbors ("input/files/Aori/neighbors.txt", ncells, cell);
  // write_neighbors ("", NCELLS, cell);


  // Specify grid boundaries

  double x_min = X_MIN;
  double x_max = X_MAX;
  double y_min = Y_MIN;
  double y_max = Y_MAX;
  double z_min = Z_MIN;
  double z_max = Z_MAX;

  double threshold = THRESHOLD;   // keep cells if rel_density_change > threshold


  // Reduce grid

  long ncells_red ;//= reduce (ncells, cell, threshold, x_min, x_max, y_min, y_max, z_min, z_max);


  // Define the reduced grid

  CELL *cell_red = new CELL[ncells_red];

  initialize_cells (ncells_red, cell_red);

  initialize_reduced_grid (ncells_red, cell_red, ncells, cell);


  // Define full grid

  long size_x = 2;
  long size_y = 0;
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




  // APPLY MAGRITTE ...



  // Interpolate reduced grid back to original grid

  interpolate (ncells_red, cell_red, ncells, cell);

  delete [] cell_red;

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
