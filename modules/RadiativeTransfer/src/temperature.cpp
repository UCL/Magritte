// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <vector>
#include <string>
#include <fstream>
using namespace std;

#include "temperature.hpp"


TEMPERATURE :: TEMPERATURE (const long num_of_cells)
{
	
  ncells = num_of_cells;

	gas.resize (ncells);
	
	dust.resize (ncells);

	gas_prev.resize (ncells);


}   // END OF CONSTRUCTOR



int TEMPERATURE :: read (const string temperature_file)
{

  ifstream infile (temperature_file);

  for (long p = 0; p < ncells; p++)
	{
		infile >> gas[p];
	}


	return (0);

}
