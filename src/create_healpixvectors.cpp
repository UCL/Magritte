/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* create_healpixvectors: Create the unit HEALPix vectors and find their antipodals              */
/*                                                                                               */
/* (based on the evaluation_points routine in 3D-PDR and a piece of main 3DPDR)                  */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <math.h>

#include "declarations.hpp"
#include "create_healpixvectors.hpp"
#include "HEALPix/chealpix.hpp"



/* create_healpixvector: store the HEALPix vectors and find the antipodal pairs                  */
/*-----------------------------------------------------------------------------------------------*/

void create_healpixvectors(double *unit_healpixvector, long *antipod)
{

  double vector[3];                        /* unit vector in the direction of the HEALPix vector */

  long r1, r2, r3, ipix;                                                          /* ray indices */


  /* Create the (unit) HEALPix vectors  */

  for (r1=0; r1<NRAYS; r1++){

    ipix = r1;
    pix2vec_nest(NSIDES, ipix, vector);

    unit_healpixvector[VINDEX(ipix,0)] = vector[0];
    unit_healpixvector[VINDEX(ipix,1)] = vector[1];
    unit_healpixvector[VINDEX(ipix,2)] = vector[2];
  }



  /* Find the antipodal pairs */
  /* HEALPix vectors are not perfectly antipodal, TOL gives the allowed tolerance */

  for (r2=0; r2<NRAYS; r2++){

    for (r3=0; r3<NRAYS; r3++){

      if ( fabs(unit_healpixvector[VINDEX(r2,0)]+unit_healpixvector[VINDEX(r3,0)]) < TOL
           && fabs(unit_healpixvector[VINDEX(r2,1)]+unit_healpixvector[VINDEX(r3,1)]) < TOL
           && fabs(unit_healpixvector[VINDEX(r2,3)]+unit_healpixvector[VINDEX(r3,3)]) < TOL ){

        antipod[r2] = r3;
      }
    }
  }


}

/*-----------------------------------------------------------------------------------------------*/
