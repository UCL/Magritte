// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
using namespace std;

#include "rays.hpp"


///  Constructor for Rays
///    @param[in] io: io object
/////////////////////////////////////

Rays ::
    Rays (
        const Io &io)
  : nrays (io.get_length ("rays"))
{

  allocate ();

  read (io);

  setup ();


}   // END OF CONSTRUCTOR




///  allocate: resize all data structures
/////////////////////////////////////////

int Rays ::
    allocate ()
{

  x.resize (nrays);
  y.resize (nrays);
  z.resize (nrays);

  Ix.resize (nrays);
  Iy.resize (nrays);

  Jx.resize (nrays);
  Jy.resize (nrays);
  Jz.resize (nrays);

  antipod.resize (nrays);


  return (0);

}




///  read: read the input into the data structure
///  @paran[in] io: io object
/////////////////////////////////////////////////

int Rays ::
    read (
        const Io &io)
{

  io.read_3_vector ("rays", x, y, z);


  return (0);

}




///  setup: setup data structure
////////////////////////////////

int Rays ::
    setup ()
{

  setup_antipodal_rays ();

  setup_image_axis ();


  return (0);

}




///  setup_antipodal_rays: identify which rays are each others antipodes
////////////////////////////////////////////////////////////////////////

int Rays ::
    setup_antipodal_rays ()
{

  // (!) HEALPix vectors are not perfectly antipodal, so a tolerance is given

  const double tolerance = 1.0E-9;

  for (long r1 = 0; r1 < nrays; r1++)
  {
    for (long r2 = 0; r2 < nrays; r2++)
    {
      if (    (fabs(x[r1]+x[r2]) < tolerance)
           && (fabs(y[r1]+y[r2]) < tolerance)
           && (fabs(z[r1]+z[r2]) < tolerance) )
      {
        antipod[r1] = r2;
      }
    }
  }

  return (0);

}




///  setup_image_axis: define the axis for the (2D) image frame
///////////////////////////////////////////////////////////////

int Rays ::
    setup_image_axis ()
{

  for (long r = 0; r < nrays; r++)
  {
    double inverse_denominator = 1.0 / sqrt(x[r]*x[r] + y[r]*y[r]);

    Ix[r] =  y[r] * inverse_denominator;
    Iy[r] = -x[r] * inverse_denominator;

    Jx[r] =  x[r] * z[r] * inverse_denominator;
    Jy[r] =  y[r] * z[r] * inverse_denominator;
    Jz[r] =              - inverse_denominator;
  }


  return (0);

}




///  write: write out the data structure
///  @paran[in] io: io object
/////////////////////////////////////////////////

int Rays ::
    write (
        const Io &io) const
{

  io.write_3_vector ("rays", x, y, z);


  return (0);

}


// int Rays ::
//     setup_mirror_rays ()
// {
//
//   // Find mirror rays about xz-plane
//
//   for (long r1 = 0; r1 < Nrays; r1++)
//   {
//     for (long r2 = 0; r2 < Nrays; r2++)
//     {
//       if (    (fabs(x[r1]-x[r2]) < tolerance)
//            && (fabs(y[r1]+y[r2]) < tolerance)
//            && (fabs(z[r1]-z[r2]) < tolerance) )
//       {
//         mirror_xz[r1] = r2;
//       }
//     }
//   }
//
//   return (0);
//
// }
