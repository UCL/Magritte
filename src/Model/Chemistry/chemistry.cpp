// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "chemistry.hpp"
#include "Tools/logger.hpp"


///  read: read model data
///    @param[in] io: io data object
///    @paran[in] parameters: model parameters object
/////////////////////////////////////////////////////

int Chemistry ::
   read (
      const Io         &io,
            Parameters &parameters)
{

  write_to_log ("Reading chemistry");


  species.read (io, parameters);


  return (0);

}




///  write: write out model data
///    @param[in] io: io data object
////////////////////////////////////

int Chemistry ::
   write (
      const Io &io) const
{

  write_to_log ("Writing chemistry");


  species.write (io);


 return (0);

}