// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "turbulence.hpp"
#include "Tools/constants.hpp"


const string Turbulence::prefix = "Thermodynamics/Turbulence/";


///  read: read in data structure
///    @param[in] io: io object
///    @paran[in] parameters: model parameters object
/////////////////////////////////////////////////////

int Turbulence ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  // Get number of cells from length of temperature/gas file
  io.read_length (prefix+"vturbulence", ncells);


  parameters.set_ncells (ncells);


  // Read gas turbulence file
  vturb2.resize (ncells);
  io.read_list (prefix+"vturbulence", vturb2);

  // Convert to square of turbulent velocity w.r.t. c
  for (long p = 0; p < ncells; p++)
  {
    vturb2[p] /= CC;          // devide by speed of light
    vturb2[p] *= vturb2[p];   // square
  }


  return (0);

}




///  write: write out data structure
///    @param[in] io: io object
/////////////////////////////////

int Turbulence ::
    write (
        const Io &io) const
{

  // Read gas turbulence file
  io.write_list (prefix+"vturbulence", vturb2);


  return (0);

}
