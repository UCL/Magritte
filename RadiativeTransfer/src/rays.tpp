// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________

#include <math.h>
#include <iostream>

#include "declarations.hpp"
#include "HEALPix/chealpix.h"


/// Constructor for RAYS
////////////////////////

template <int Dimension, long Nrays>
RAYS <Dimension, Nrays> ::
RAYS ()
{

  // Assert that Nrays is consistent with Dimension

  static_assert ( (Dimension == 1) || (Dimension == 2) || (Dimension == 3),
                  "Dimension should be 1, 2 or 3.");
  static_assert ( (Dimension != 1) || (Nrays == 2),
                  "In 1D there can only be 2 rays!");
  static_assert ( (Dimension != 3) || (sqrt(Nrays/12.0) - long(sqrt(Nrays/12.0)) == 0),
                  "Nrays should be of the form 12*n*n for an integer n.");


  // Create (unit) HEALPix vectors

  if (Dimension == 1)
  {
    x[0] = +1.0;
    y[0] =  0.0;
    z[0] =  0.0;

    x[1] = -1.0;
    y[1] =  0.0;
    z[1] =  0.0;
  }


  if (Dimension == 2)
  {
    for (long r = 0; r < Nrays; r++)
    {
      double theta = (2.0*PI*r) / Nrays;

      x[r] = cos(theta);
      y[r] = sin(theta);
      z[r] = 0.0;
    }
  }


  if (Dimension == 3)
  {
    long nsides = (long) sqrt(Nrays/12);

    for (long r = 0; r < Nrays; r++)
    {
      double vector[3];   // unit vector in direction of HEALPix ray

      pix2vec_nest (nsides, r, vector);

      x[r] = vector[0];
      y[r] = vector[1];
      z[r] = vector[2];
    }
  }


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
