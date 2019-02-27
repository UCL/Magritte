// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "thermodynamics.hpp"


///  read: read in data structure
///    @param[in] io: io object
///    @paran[in] parameters: model parameters object
/////////////////////////////////////////////////////

int Thermodynamics ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  temperature.read (io, parameters);


  turbulence.read  (io, parameters);


  return (0);

}




///  write: write out data structure
///    @param[in] io: io object
/////////////////////////////////

int Thermodynamics ::
    write (
        const Io &io) const
{

  temperature.write (io);


  turbulence.write  (io);


  return (0);

}
