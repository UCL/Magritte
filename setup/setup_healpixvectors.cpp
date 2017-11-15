/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* setup_healpixvectors: Create the unit HEALPix vectors and find their antipodals               */
/*                                                                                               */
/* (based on the evaluation_points routine in 3D-PDR and a piece of main 3DPDR)                  */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <math.h>

#include "setup_definitions.hpp"
#include "setup_healpixvectors.hpp"
#include "HEALPix/chealpix.hpp"



/* setup_healpixvector: store the HEALPix vectors and find the antipodal pairs                   */
/*-----------------------------------------------------------------------------------------------*/

int setup_healpixvectors(long nrays, double *unit_healpixvector, long *antipod)
{


  /* Create the (unit) HEALPix vectors  */

  for (long r1=0; r1<nrays; r1++){

    long ipix = r1;

    double vector[3];                      /* unit vector in the direction of the HEALPix vector */

    pix2vec_nest(NSIDES, ipix, vector);

    unit_healpixvector[VINDEX(ipix,0)] = vector[0];
    unit_healpixvector[VINDEX(ipix,1)] = vector[1];
    unit_healpixvector[VINDEX(ipix,2)] = vector[2];
  }



  /* Find the antipodal pairs */
  /* HEALPix vectors are not perfectly antipodal, TOL gives the allowed tolerance */

  for (long r2=0; r2<nrays; r2++){

    for (long r3=0; r3<nrays; r3++){

      if (    (fabs(unit_healpixvector[VINDEX(r2,0)] + unit_healpixvector[VINDEX(r3,0)]) < TOL)
           && (fabs(unit_healpixvector[VINDEX(r2,1)] + unit_healpixvector[VINDEX(r3,1)]) < TOL)
           && (fabs(unit_healpixvector[VINDEX(r2,2)] + unit_healpixvector[VINDEX(r3,2)]) < TOL) ){

        antipod[r2] = r3;
      }

    }
  }


  return(0);
}

/*-----------------------------------------------------------------------------------------------*/
