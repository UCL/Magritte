// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "model.hpp"
#include "Io/io.hpp"


///  read: read model data
///    @param[in] io: io data object
////////////////////////////////////

int Model ::
   read (
      const Io &io)
{

  geometry.read       (io, parameters);

  thermodynamics.read (io, parameters);

  chemistry.read      (io, parameters);

  lines.read          (io, parameters);

  radiation.read      (io, parameters);


  return (0);

}




///  write: write out model data
///    @param[in] io: io data object
////////////////////////////////////

int Model ::
   write (
      const Io &io) const
{

  geometry.write       (io);

  thermodynamics.write (io);

  chemistry.write      (io);

  lines.write          (io);

  radiation.write      (io);


 return (0);

}
