// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "geometry.hpp"
#include "Tools/logger.hpp"


///  read: read the input into the data structure
///    @paran[in] io: io object
///    @paran[in] parameters: model parameters object
/////////////////////////////////////////////////////

int Geometry ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  write_to_log ("Reading geometry");


  cells.read    (io, parameters);

  rays.read     (io, parameters);

  boundary.read (io, parameters);


  return (0);

}




///  write: write the dat astructure
///  @paran[in] io: io object
////////////////////////////////////////////////

int Geometry ::
    write (
        const Io &io) const
{

  write_to_log ("Writing geometry");


  cells.write    (io);

  rays.write     (io);

  boundary.write (io);


  return (0);

}
