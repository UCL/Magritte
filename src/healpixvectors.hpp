// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <math.h>

#include "declarations.hpp"
#include "HEALPix/chealpix.h"

#define TOL


struct HEALPIXVECTORS
{
  // Direction
  double x[NRAYS];
  double y[NRAYS];
  double z[NRAYS];

  long antipod[NRAYS];
  long mirror_xz[NRAYS];


  HEALPIXVECTORS()
  {

    // Create (unit) HEALPix vectors

#   if   (DIMENSIONS == 1)

      if (NRAYS != 2)
      {
        printf ("\nERROR: In 1D there can only be rays in 2 rays !\n\n");
      }

      x[0] = +1.0;
      y[0] =  0.0;
      z[0] =  0.0;

      x[1] = -1.0;
      y[1] =  0.0;
      z[1] =  0.0;

#   elif (DIMENSIONS == 2)

      if (NRAYS <= 2)
      {
        printf ("\nERROR: In 2D there must be more than 2 rays !\n\n");
      }


      for (long ray = 0; ray < NRAYS; ray++)
      {
        double theta = (2.0*PI*ray) / NRAYS;

        x[ray] = cos(theta);
        y[ray] = sin(theta);
        z[ray] = 0.0;
      }

#   elif (DIMENSIONS == 3)

      long nsides = (long) sqrt(NRAYS/12);

      for (long ipix = 0; ipix < NRAYS; ipix++)
      {
        double vector[3];   // unit vector in direction of HEALPix ray

        pix2vec_nest (nsides, ipix, vector);

        x[ipix] = vector[0];
        y[ipix] = vector[1];
        z[ipix] = vector[2];
      }

#   endif


    // Find antipodal pairs
    // (!) HEALPix vectors are not perfectly antipodal, so a tolerance is given

    const double tolerance = 1.0E-9;

    for (long r1 = 0; r1 < NRAYS; r1++)
    {
      for (long r2 = 0; r2 < NRAYS; r2++)
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

    for (long r1 = 0; r1 < NRAYS; r1++)
    {
      for (long r2 = 0; r2 < NRAYS; r2++)
      {
        if (    (fabs(x[r1]-x[r2]) < tolerance)
             && (fabs(y[r1]+y[r2]) < tolerance)
             && (fabs(z[r1]-z[r2]) < tolerance) )
        {
          mirror_xz[r1] = r2;
        }
      }
    }


  } // end of constructor

};
