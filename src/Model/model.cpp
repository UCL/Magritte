// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "model.hpp"
#include "Io/io.hpp"
#include "Tools/logger.hpp"


///  read: read model data
///    @param[in] io: io data object
////////////////////////////////////

int Model ::
   read (
      const Io &io)
{

  write_to_log ("Reading Model");


  geometry.read       (io, parameters);

  thermodynamics.read (io, parameters);

  chemistry.read      (io, parameters);

  lines.read          (io, parameters);

  radiation.read      (io, parameters);


  write_to_log ("-----------------");
  write_to_log ("Model parameters:");
  write_to_log ("-----------------");
  write_to_log ("ncells     = ", parameters.ncells     ());
  write_to_log ("nrays      = ", parameters.nrays      ());
  write_to_log ("nrays_red  = ", parameters.nrays_red  ());
  write_to_log ("nboundary  = ", parameters.nboundary  ());
  write_to_log ("nfreqs     = ", parameters.nfreqs     ());
  write_to_log ("nfreqs_red = ", parameters.nfreqs_red ());
  write_to_log ("nspecs     = ", parameters.nspecs     ());
  write_to_log ("nlspecs    = ", parameters.nlspecs    ());
  write_to_log ("nlines     = ", parameters.nlines     ());
  write_to_log ("nquads     = ", parameters.nquads     ());
  write_to_log ("-----------------");


  return (0);

}




///  write: write out model data
///    @param[in] io: io data object
////////////////////////////////////

int Model ::
   write (
      const Io &io) const
{

  write_to_log ("Writing Model");


  geometry.write       (io);

  thermodynamics.write (io);

  chemistry.write      (io);

  lines.write          (io);

  radiation.write      (io);


 return (0);

}
