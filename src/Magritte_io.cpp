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

  // initialize_cells (cell, NCELLS);


  // Read input grid

  std::cout << "Reading input grid...\n";

  read_vtu_input (inputfile, NCELLS, cell);
  // read_txt_input (inputfile, NCELLS, cell);
  //
  // Find neighboring cells for each cell
  //
  std::cout << "Finding neighbors...\n";
  //
  read_neighbors ("input/files/Aori/neighbors.txt", NCELLS, cell);
  //
  //
  // find_neighbors (NCELLS, cell);
  //
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

// long ncells_red = NCELLS;

  std::cout << "Reduced grid has " << ncells_red << " cells\n";


  // Define the reduced grid

  CELL *cell_red = new CELL[ncells_red];

  for (long n = 0; n < NCELLS; n++)
  {
    if (!cell[n].removed)
    {
      cell_red[cell[n].id].density = cell[n].density;
    }
  }


  // Interpolate reduced grid back to original grid

  // interpolate (ncells_red, cell_red, NCELLS, cell);


  // write reduced grid as .txt file and .vtu file

  std::cout << "  Writing .txt grid...\n";

  // std::ostringstream strs;
  // strs << threshold;
  // std::string thres = strs.str();


  // // write_grid ("reduced_" + thres, NCELLS, cell);

  // long color = 0;
  //
  // for (long p = 0; p < NCELLS; p++)
  // {
  //   if (!cell[p].removed)
  //   {
  //     cell[p].density = color;
  //     cell[p].removed = true;
  //
  //     for (int n = 0; n < cell[p].n_neighbors; n++)
  //     {
  //       long nr = cell[p].neighbor[n];   // nr of neighbor in grid
  //
  //       if (!cell[nr].removed)
  //       {
  //         cell[nr].density = color;
  //         cell[nr].removed = true;
  //       }
  //     }
  //
  //     color++;
  //   }
  // }


  write_vtu_output (NCELLS, cell, inputfile);

  // std::cout << cell[0].n_neighbors << "\n";

  // delete [] cell_red;

  return (0);

}
