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

    spherical_symmetry   = parameters.spherical_symmetry  ();
    adaptive_ray_tracing = parameters.adaptive_ray_tracing();

    cells   .read(io, parameters);
    boundary.read(io, parameters);

    if (parameters.adaptive_ray_tracing())
    {
        cout << "Setting adaptive rays..." << endl;

        size_t sample_size = 10000;
        if (sample_size > parameters.ncells()) sample_size = parameters.ncells()/2;

        set_adaptive_rays(parameters.order_min(), parameters.order_max(), sample_size);

        parameters.set_nrays(rays.nrays);
    }
    else
    {
        rays.read(io, parameters);
    }

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
