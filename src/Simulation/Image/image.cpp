// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "image.hpp"
#include "Tools/logger.hpp"
#include "Tools/Parallel/wrap_omp.hpp"


const string Image::prefix = "Simulation/Image/";


///  Constructor for Image
//////////////////////////

Image :: Image (const long ray_nr, const Parameters &parameters)
  : ray_nr     (ray_nr)
  , ncells     (parameters.ncells())
  , nfreqs     (parameters.nfreqs())
  , nfreqs_red (parameters.nfreqs_red())
{
    // Size and initialize
    ImX.resize (ncells);
    ImY.resize (ncells);

    I_p.resize (ncells);
    I_m.resize (ncells);

    for (size_t c = 0; c < ncells; c++)
    {
        I_p[c].resize (nfreqs_red);
        I_m[c].resize (nfreqs_red);
    }

}   // END OF CONSTRUCTOR




///  print: write out the images
///    @param[in] io: io object
////////////////////////////////

void Image :: write (const Io &io) const
{
    cout << "Writing image..."    << endl;
    cout << "nfreqs = " << nfreqs << endl;
    cout << "ncells = " << ncells << endl;

    const string str_ray_nr = std::to_string (ray_nr);

    io.write_list  (prefix+"ImX_"+str_ray_nr, ImX);
    io.write_list  (prefix+"ImY_"+str_ray_nr, ImY);

    // Create one intensity variable and reuse for I_m and I_p to save memory.
    Double2 intensity_m (ncells, Double1 (nfreqs));
    Double2 intensity_p (ncells, Double1 (nfreqs));

    OMP_PARALLEL_FOR (p, ncells)
    {
        for (size_t f = 0; f < nfreqs; f++)
        {
            intensity_m[p][f] = get (I_m[p], f);
            intensity_p[p][f] = get (I_p[p], f);
        }
    }

    io.write_array (prefix+"I_m_"+str_ray_nr, intensity_m);
    io.write_array (prefix+"I_p_"+str_ray_nr, intensity_p);

//    OMP_PARALLEL_FOR (p, ncells)
//    {
//        for (size_t f = 0; f < nfreqs; f++)
//        {
//            intensity[p][f] = get (I_p[p], f);
//        }
//    }

}




///  Setter for the coordinates on the image axes
///    @param[in] geometry : geometry object of the model
/////////////////////////////////////////////////////////

void Image :: set_coordinates (const Geometry &geometry)
{
    const double rx = geometry.rays.ray(0, ray_nr).x();
    const double ry = geometry.rays.ray(0, ray_nr).y();
    const double rz = geometry.rays.ray(0, ray_nr).z();

    const double         denominator = sqrt (rx*rx + ry*ry);
    const double inverse_denominator = 1.0 / denominator;

    const double ix =  ry * inverse_denominator;
    const double iy = -rx * inverse_denominator;

    const double jx =  rx * rz * inverse_denominator;
    const double jy =  ry * rz * inverse_denominator;
    const double jz = -denominator;

    if (denominator != 0.0)
    {
        OMP_PARALLEL_FOR (p, ncells)
        {
            ImX[p] =   ix * geometry.cells.position[p].x()
                     + iy * geometry.cells.position[p].y();

            ImY[p] =   jx * geometry.cells.position[p].x()
                     + jy * geometry.cells.position[p].y()
                     + jz * geometry.cells.position[p].z();
        }
    }

    else
    {
        OMP_PARALLEL_FOR (p, ncells)
        {
            ImX[p] = geometry.cells.position[p].x();

            ImY[p] = geometry.cells.position[p].y();
        }
    }
}
