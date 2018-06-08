// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <vector>
using namespace std;

#include "temperature.hpp"


TEMPERATURE :: TEMPERATURE (long num_of_cells)
{
	
  ncells = num_of_cells;

	gas.resize (ncells);
	
	dust.resize (ncells);

	gas_prev.resize (ncells);


}   // END OF CONSTRUCTOR
