// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "geometry.hpp"
#include "Tools/logger.hpp"


///  read: read the input into the data structure
///    @param[in] io: io object
///    @param[in] parameters: model parameters object
/////////////////////////////////////////////////////

int Geometry ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  cout << "Reading geometry" << endl;


  cameras.read  (io, parameters);

  cells.read    (io, parameters);

  rays.read     (io, parameters);

  boundary.read (io, parameters);


  nrays = parameters.nrays();


  return (0);

}




///  write: write the dat astructure
///  @param[in] io: io object
////////////////////////////////////////////////

int Geometry ::
    write (
        const Io &io) const
{

  cout << "Writing geometry" << endl;


  cameras.write  (io);

  cells.write    (io);

  rays.write     (io);

  boundary.write (io);


  return (0);

}
