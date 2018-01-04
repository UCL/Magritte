// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "definitions.hpp"

#include "initializers.hpp"
#include "read_input.hpp"
#include "ray_tracing.hpp"
#include "reduce.hpp"
#include "write_txt_tools.hpp"
#include "write_vtu_tools.hpp"


int main ()
{

  // Define and initialize cells

  std::cout << "Defining and initializing cells...\n";

  CELL cell[NCELLS];

  initialize_cells (cell, NCELLS);


  // Read input grid

  std::cout << "Reading input grid...\n";

  read_vtu_input (inputfile, NCELLS, cell);


  // Find neighboring cells for each cell

  std::cout << "Finding neighbors...\n";

  find_neighbors (NCELLS, cell);


  // Crop grid

  std::cout << "Cropping input grid...\n";

  double x_min =  0.0E+00;
  double x_max =  8.0E+16;
  double y_min = -1.6E+17;
  double y_max =  6.0E+16;
  double z_min =  0.0E+00;
  double z_max =  0.0E+00;

  crop (NCELLS, cell, x_min, x_max, y_min, y_max, z_min, z_max);


  // Reduce grid

  std::cout << "Reducing input grid...\n";

  reduce (NCELLS, cell);


  // write reduced grid as .txt file

  std::cout << "Writing .txt grid...\n";

  write_grid ("", NCELLS, cell);


  // write inputfile with annotated cell id's

  std::cout << "Writing .vtu grid...\n";

  write_vtu_output (NCELLS, cell, inputfile);


  return (0);

}
