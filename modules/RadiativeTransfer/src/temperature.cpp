// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
#include <fstream>
using namespace std;

#include "temperature.hpp"
#include "types.hpp"


///  Constructor for TEMPERATURE
///    @param[in] num_of_cells: number of cells
///////////////////////////////////////////////

TEMPERATURE :: TEMPERATURE (const long num_of_cells)
	: ncells (num_of_cells)
{

	gas.resize (ncells);
	dust.resize (ncells);
	gas_prev.resize (ncells);


}   // END OF CONSTRUCTOR




///  read: read initial temperatures
///    @param[in] temperature_file: name of file containing temperature data
////////////////////////////////////////////////////////////////////////////

int TEMPERATURE ::
    read (const string temperature_file)
{

  ifstream infile (temperature_file);

  for (long p = 0; p < ncells; p++)
	{
		infile >> gas[p];
	}


	return (0);

}
