// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
#include <fstream>
using namespace std;

#include "model.hpp"


///  Constructor for Model
///    @param[in] num_of_cells: number of cells
///////////////////////////////////////////////

Model ::
  Model (const long num_of_cells)
  : ncells      (num_of_cells),
    temperature (num_of_cells)
{


}   // END OF CONSTRUCTOR
