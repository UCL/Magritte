// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "temperature.hpp"


const string Temperature::prefix = "Thermodynamics/Temperature/";


///  read: read in data structure
///    @param[in] io: io object
///    @paran[in] parameters: model parameters object
/////////////////////////////////////////////////////

int Temperature ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  // Get number of cells from length of temperature/gas file
  io.read_length (prefix+"gas", ncells);


  parameters.set_ncells (ncells);


  // Read gas temperature file
  gas.resize (ncells);
  io.read_list (prefix+"gas", gas);


  //    dust.resize (ncells);
  //gas_prev.resize (ncells);


  return (0);

}




///  write: write out data structure
///    @param[in] io: io object
/////////////////////////////////

int Temperature ::
    write (
        const Io &io) const
{

  // Read gas temperature file
  io.write_list (prefix+"gas", gas);


  return (0);

}
