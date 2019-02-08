// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "temperature.hpp"
#include "constants.hpp"
#include "types.hpp"


///  read: read in data structure
///    @param[in] io: io object
/////////////////////////////////

int Temperature ::
    read (
        const Io &io)
{

  // Get number of cells from length of temperature/gas file
  io.read_length ("temperature/gas", ncells);

  // Read gas temperature file
  gas.resize (ncells);
  io.read_list ("temperature/gas", gas);


  // Read gas turbulence file
  vturb2.resize (ncells);
  io.read_list ("temperature/vturbulence", vturb2);

  // Convert to square of turbulent velocity w.r.t. c
  for (long p = 0; p < ncells; p++)
  {
    vturb2[p] /= CC;          // devide by speed of light
    vturb2[p] *= vturb2[p];   // square
  }


      dust.resize (ncells);
  gas_prev.resize (ncells);


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
  io.write_list ("temperature/gas", gas);


  // Read gas turbulence file
  io.write_list ("temperature/vturbulence", vturb2);


  return (0);

}
