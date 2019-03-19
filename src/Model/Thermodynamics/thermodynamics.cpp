// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "thermodynamics.hpp"
#include "Tools/logger.hpp"


///  read: read in data structure
///    @param[in] io: io object
///    @paran[in] parameters: model parameters object
/////////////////////////////////////////////////////

int Thermodynamics ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  write_to_log ("Reading thermodynamics");


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

  write_to_log ("Writing thermodynamics");


  temperature.write (io);


  turbulence.write  (io);


  return (0);

}
