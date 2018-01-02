// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <math.h>

#include "setup_definitions.hpp"
#include "setup_healpixvectors.hpp"
#include "HEALPix/chealpix.hpp"


// setup_healpixvector: store HEALPix vectors and find antipodal pairs
// -------------------------------------------------------------------

int setup_healpixvectors (long nrays, double *healpixvector, long *antipod)
{


  /* Create the (unit) HEALPix vectors in 3D */


# if   (DIMENSIONS == 1)

    if (nrays != 2)
    {
      printf ("\nERROR: In 1D there can only be rays in 2 rays !\n\n");
    }

    /* Only 2 rays along the x-axis */

    healpixvector[VINDEX(0,0)] = +1.0;
    healpixvector[VINDEX(0,1)] =  0.0;
    healpixvector[VINDEX(0,2)] =  0.0;

    healpixvector[VINDEX(1,0)] = -1.0;
    healpixvector[VINDEX(1,1)] =  0.0;
    healpixvector[VINDEX(1,2)] =  0.0;

# elif (DIMENSIONS == 2)

    if (nrays <= 2)
    {
      printf ("\nERROR: In 2D there must be more than 2 rays !\n\n");
    }


    for (long ray = 0; ray < nrays; ray++)
    {
      double theta = (2.0*PI*ray) / nrays;

      healpixvector[VINDEX(ray,0)] = cos(theta);
      healpixvector[VINDEX(ray,1)] = sin(theta);
      healpixvector[VINDEX(ray,2)] = 0.0;
    }

# elif (DIMENSIONS == 3)

    for (long ipix = 0; ipix < nrays; ipix++)
    {
      double vector[3];   // unit vector in direction of HEALPix ray

      long nsides = (long) sqrt(nrays/12);

      pix2vec_nest (nsides, ipix, vector);

      healpixvector[VINDEX(ipix,0)] = vector[0];
      healpixvector[VINDEX(ipix,1)] = vector[1];
      healpixvector[VINDEX(ipix,2)] = vector[2];
    }

# endif


  // Find the antipodal pairs
  // HEALPix vectors are not perfectly antipodal, TOL is given tolerance

  for (long r1 = 0; r1 < nrays; r1++)
  {
    for (long r2 = 0; r2 < nrays; r2++)
    {
      if (    (fabs(healpixvector[VINDEX(r1,0)] + healpixvector[VINDEX(r2,0)]) < TOL)
           && (fabs(healpixvector[VINDEX(r1,1)] + healpixvector[VINDEX(r2,1)]) < TOL)
           && (fabs(healpixvector[VINDEX(r1,2)] + healpixvector[VINDEX(r2,2)]) < TOL) )
      {
        antipod[r1] = r2;
      }

    }
  }


  return(0);
}
