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

void Geometry :: read (const Io &io, Parameters &parameters)
{
    cout << "Reading geometry..." << endl;

    cells   .read (io, parameters);
    rays    .read (io, parameters);
    boundary.read (io, parameters);

    nrays = parameters.nrays();
}




///  write: write the dat a structure
///    @param[in] io: io object
////////////////////////////////////////////////

void Geometry :: write (const Io &io)
{
    cout << "Writing geometry..." << endl;

    cells.   write (io);
    rays.    write (io);
    boundary.write (io);
}
