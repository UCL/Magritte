// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <sstream>
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
  // read_txt_input (inputfile, NCELLS, cell);


  // Find neighboring cells for each cell

  std::cout << "Finding neighbors...\n";

  read_neighbors ("input/files/Aori/neighbors.txt", NCELLS, cell);


  // find_neighbors (NCELLS, cell);

  // write_neighbors ("", NCELLS, cell);


  // Specify grid boundaries

  double x_min =  0.0E+00;
  double x_max =  8.0E+16;
  double y_min = -1.6E+17;
  double y_max =  6.0E+16;
  double z_min =  0.0E+00;
  double z_max =  0.0E+00;


  // Reduce grid

  std::cout << "Reducing grid...\n";

  double threshold = 1.0E9;   // keep cells if rel_density_change > threshold

  long ncells_red = reduce (NCELLS, cell, threshold, x_min, x_max, y_min, y_max, z_min, z_max);

  std::cout << "Reduced grid has " << ncells_red << " cells\n";


  // Define the reduced grid

  CELL *cell_red = new CELL[ncells_red];

  initialize_reduced_grid (ncells_red, cell_red, NCELLS, cell);




  // APPLY MAGRITTE ...




  // Interpolate reduced grid back to original grid

  interpolate (ncells_red, cell_red, NCELLS, cell);

  delete [] cell_red;


  // write reduced grid as .txt file and .vtu file

  std::cout << "  Writing .txt grid...\n";

  std::ostringstream strs;
  strs << threshold;
  std::string thres = strs.str();


  write_grid ("reduced_" + thres, NCELLS, cell);



  write_vtu_output (NCELLS, cell, inputfile);


  return (0);

}
