// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "temperature.hpp"
#include "Tools/logger.hpp"


const string Temperature::prefix = "Thermodynamics/Temperature/";


///  read: read in data structure
///    @param[in] io: io object
///    @param[in] parameters: model parameters object
/////////////////////////////////////////////////////

void Temperature :: read (const Io &io, Parameters &parameters)
{
  cout << "Reading temperature..." << endl;


  // Get number of cells from length of temperature/gas file
  io.read_length (prefix+"gas", ncells);


  parameters.set_ncells (ncells);


  // Read gas temperature file
  gas.resize (ncells);
  io.read_list (prefix+"gas", gas);


  //    dust.resize (ncells);
  //gas_prev.resize (ncells);
}




///  write: write out data structure
///    @param[in] io: io object
/////////////////////////////////

void Temperature :: write (const Io &io) const
{
  cout << "Writing temperature..." << endl;


  // Read gas temperature file
  io.write_list (prefix+"gas", gas);
}
