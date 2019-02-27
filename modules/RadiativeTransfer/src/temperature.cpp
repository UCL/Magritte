// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
#include <fstream>
using namespace std;

#include "temperature.hpp"
#include "constants.hpp"
#include "types.hpp"


///  Constructor for TEMPERATURE
///    @param[in] num_of_cells: number of cells
///////////////////////////////////////////////

TEMPERATURE ::
TEMPERATURE (const long num_of_cells)
  : ncells (num_of_cells)
{

       gas.resize (ncells);
      dust.resize (ncells);
  gas_prev.resize (ncells);

    vturb2.resize (ncells);


}   // END OF CONSTRUCTOR




///  read: read initial temperatures
///    @param[in] temperature_file: name of file containing temperature data
////////////////////////////////////////////////////////////////////////////

int TEMPERATURE ::
    read                               (
        const string temperature_file,
        const string vturbulence_file  )
{

  // Read gas temperature file

  ifstream file_temp (temperature_file);

  for (long p = 0; p < ncells; p++)
  {
    file_temp >> gas[p];
  }

  file_temp.close();


 // Read gas turbulence file

  ifstream file_turb (vturbulence_file);

  for (long p = 0; p < ncells; p++)
  {
    file_turb >> vturb2[p];

    vturb2[p] /= CC;          // devide by speed of light
    vturb2[p] *= vturb2[p];   // square
  }

  file_turb.close();


  return (0);

}
