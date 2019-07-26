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

Image ::
Image (
    const long        ray_nr,
    const Parameters &parameters)
  : ray_nr   (ray_nr)
  , ncells   (parameters.ncells())
  , ncameras (parameters.ncameras())
  , nfreqs   (parameters.nfreqs())
{

  // Size and initialize Ip_out and Im_out

  ImX.resize (ncells);
  ImY.resize (ncells);

  I_p.resize (ncells);
  I_m.resize (ncells);

  for (long c = 0; c < ncells; c++)
  //for (long c = 0; c < ncameras; c++)
  {
    I_p[c].resize (nfreqs);
    I_m[c].resize (nfreqs);
  }


}   // END OF CONSTRUCTOR




///  print: write out the images
///    @param[in] io: io object
////////////////////////////////

int Image ::
    write (
        const Io &io) const
{

  cout << "Writing image" << endl;

  const string str_ray_nr = std::to_string (ray_nr);

  io.write_list  (prefix+"ImX_"+str_ray_nr, ImX);
  io.write_list  (prefix+"ImY_"+str_ray_nr, ImY);

  io.write_array (prefix+"I_m_"+str_ray_nr, I_m);
  io.write_array (prefix+"I_p_"+str_ray_nr, I_p);


  return (0);

}



///  Setter for the coordinates on the image axes
///    @param[in] geometry : geometry object of the model
/////////////////////////////////////////////////////////


int Image ::
    set_coordinates (
        const Geometry &geometry)
{

  OMP_PARALLEL_FOR (p, ncells)
  //for (long c = 0; c < ncameras; c++)
  {

    //const long p = geometry.cameras.camera2cell_nr[c];


    const double rx = geometry.rays.x[p][ray_nr];
    const double ry = geometry.rays.y[p][ray_nr];
    const double rz = geometry.rays.z[p][ray_nr];

    const double         denominator = sqrt (rx*rx + ry*ry);
    const double inverse_denominator = 1.0 / denominator;

    const double ix =  ry * inverse_denominator;
    const double iy = -rx * inverse_denominator;

    const double jx =  rx * rz * inverse_denominator;
    const double jy =  ry * rz * inverse_denominator;
    const double jz = -denominator;


    ImX[p] =   ix * geometry.cells.x[p]
             + iy * geometry.cells.y[p];

    ImY[p] =   jx * geometry.cells.x[p]
             + jy * geometry.cells.y[p]
             + jz * geometry.cells.z[p];
  }


  return (0);

}
