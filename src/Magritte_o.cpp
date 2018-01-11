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

  read_txt_input (inputfile, NCELLS, cell);


  // Append cell info to vtu input

  std::cout << "  Writing .vtu grid...\n";

  std::string append_file = "input/files/Aori/grid_reduced_0.1.vtu";

  // append_vtu_output (NCELLS, cell, append_file);


  return (0);

}
