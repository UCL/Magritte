// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________

#include <math.h>

#include "declarations.hpp"
#include "HEALPix/chealpix.h"


/// Constructor for RAYS
////////////////////////

template <int Dimension, long Nrays>
RAYS <Dimension, Nrays> ::
RAYS()
{

  // Assert that Nrays is consistent with Dimension

  static_assert ( (Dimension == 1) || (Dimension == 2) || (Dimension == 3),
                  "Dimension should be 1, 2 or 3.");
  static_assert ( (Dimension != 1) || (Nrays == 2),
                  "In 1D there can only be 2 rays!");


  // Create (unit) HEALPix vectors

# if (Dimension == 1)

    x[0] = +1.0;
    y[0] =  0.0;
    z[0] =  0.0;

    x[1] = -1.0;
    y[1] =  0.0;
    z[1] =  0.0;

# endif


# if (Dimension == 2)

    for (long ray = 0; ray < Nrays; ray++)
    {
      double theta = (2.0*PI*ray) / Nrays;

      x[ray] = cos(theta);
      y[ray] = sin(theta);
      z[ray] = 0.0;
    }

# endif


# if (Dimension == 3)

    long nsides = (long) sqrt(Nrays/12);

    for (long ipix = 0; ipix < Nrays; ipix++)
    {
      double vector[3];   // unit vector in direction of HEALPix ray

      pix2vec_nest (nsides, ipix, vector);

      x[ipix] = vector[0];
      y[ipix] = vector[1];
      z[ipix] = vector[2];
    }

# endif


  // Find antipodal pairs
  // (!) HEALPix vectors are not perfectly antipodal, so a tolerance is given

  const double tolerance = 1.0E-9;

  for (long r1 = 0; r1 < Nrays; r1++)
  {
    for (long r2 = 0; r2 < Nrays; r2++)
    {
      if (    (fabs(x[r1]+x[r2]) < tolerance)
           && (fabs(y[r1]+y[r2]) < tolerance)
           && (fabs(z[r1]+z[r2]) < tolerance) )
      {
        antipod[r1] = r2;
      }
    }
  }


  // Find mirror rays about xz-plane

  for (long r1 = 0; r1 < Nrays; r1++)
  {
    for (long r2 = 0; r2 < Nrays; r2++)
    {
      if (    (fabs(x[r1]-x[r2]) < tolerance)
           && (fabs(y[r1]+y[r2]) < tolerance)
           && (fabs(z[r1]-z[r2]) < tolerance) )
      {
        mirror_xz[r1] = r2;
      }
    }
  }


}   // END OF CONSTRUCTOR
