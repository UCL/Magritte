/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* culumn_density_calculator: Calculates the column density along each ray at each grid point    */
/*                                                                                               */
/* (based on 3DPDR in 3D-PDR)                                                                    */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "calc_column_density.hpp"
#include "ray_tracing.hpp"



/* calc_column_density: calculates column density for each species, ray and grid point           */
/*-----------------------------------------------------------------------------------------------*/


#ifdef ON_THE_FLY

int calc_column_density( GRIDPOINT *gridpoint, double *column_density, int spec )

#else

int calc_column_density( GRIDPOINT *gridpoint, EVALPOINT *evalpoint, long *key, long *raytot,
                         long *cum_raytot, double *column_density, int spec )

#endif


{


  /* For all grid points n */


# ifdef ON_THE_FLY

# pragma omp parallel                                                                             \
  shared( gridpoint, column_density, spec )                                                       \
  default( none )

# else

# pragma omp parallel                                                                             \
  shared( gridpoint, evalpoint, key, raytot, cum_raytot, column_density, spec )                   \
  default( none )

# endif


  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NGRID)/num_threads;
  long stop  = ((thread_num+1)*NGRID)/num_threads;  /* Note that the brackets are important here */


  for (long n=start; n<stop; n++){


#   ifdef ON_THE_FLY

    long key[NGRID];                  /* stores the nrs. of the grid points on the rays in order */

    long raytot[NRAYS];                    /* cumulative nr. of evaluation points along each ray */

    long cum_raytot[NRAYS];                /* cumulative nr. of evaluation points along each ray */


    EVALPOINT evalpoint[NGRID];

    get_local_evalpoint(gridpoint, evalpoint, key, raytot, cum_raytot, n);

#   endif


    for (long r=0; r<NRAYS; r++){

      column_density[RINDEX(n,r)] = column_density_at_point( gridpoint, evalpoint, key, raytot,
                                                             cum_raytot, n, spec, r);
    }

  } /* end of n loop over grid points */
  } /* end of OpenMP parallel region */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/




#ifdef ON_THE_FLY

/* calc_column_densities: calculates column densities for the species needed in chemistry        */
/*-----------------------------------------------------------------------------------------------*/

int calc_column_densities( GRIDPOINT *gridpoint, double *column_H2, double *column_HD,
                           double *column_C, double *column_CO )
{


  /* For all grid points n and rays r */

# pragma omp parallel                                                                             \
  shared( gridpoint, column_H2, column_HD, column_C, column_CO, H2_nr, HD_nr, C_nr, CO_nr )       \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NGRID)/num_threads;
  long stop  = ((thread_num+1)*NGRID)/num_threads;  /* Note that the brackets are important here */


  for (long n=start; n<stop; n++){

    long key[NGRID];                  /* stores the nrs. of the grid points on the rays in order */

    long raytot[NRAYS];                    /* cumulative nr. of evaluation points along each ray */

    long cum_raytot[NRAYS];                /* cumulative nr. of evaluation points along each ray */


    EVALPOINT evalpoint[NGRID];

    get_local_evalpoint(gridpoint, evalpoint, key, raytot, cum_raytot, n);


    for (long r=0; r<NRAYS; r++){

      column_H2[RINDEX(n,r)] = column_density_at_point( gridpoint, evalpoint, key, raytot,
                                                        cum_raytot, n, H2_nr, r );
      column_HD[RINDEX(n,r)] = column_density_at_point( gridpoint, evalpoint, key, raytot,
                                                        cum_raytot, n, HD_nr, r );
      column_C[RINDEX(n,r)]  = column_density_at_point( gridpoint, evalpoint, key, raytot,
                                                        cum_raytot, n, C_nr,  r );
      column_CO[RINDEX(n,r)] = column_density_at_point( gridpoint, evalpoint, key, raytot,
                                                        cum_raytot, n, CO_nr, r );
    }

  } /* end of n loop over grid points */
  } /* end of OpenMP parallel region */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/

#endif




/* column_density: calculates the column density for one species along one ray                   */
/*-----------------------------------------------------------------------------------------------*/

double column_density_at_point( GRIDPOINT *gridpoint, EVALPOINT *evalpoint, long *key,
                                long *raytot, long *cum_raytot, long gridp, int spec, long ray )
{


  double column_density_res = 0.0;                                   /* resulting column density */


# ifdef ON_THE_FLY

  long etot = raytot[ray];

# else

  long etot = raytot[RINDEX(gridp,ray)];

# endif


  if (etot > 0){


#   ifdef ON_THE_FLY

    long evnr       = LOCAL_GP_NR_OF_EVALP(ray,0);
    long gridp_evnr = evnr;

#   else

    long evnr       = GP_NR_OF_EVALP(gridp,ray,0);
    long gridp_evnr = GINDEX(gridp,evnr);

#   endif


    column_density_res = evalpoint[gridp_evnr].dZ * PC
                         *( gridpoint[gridp].density*species[spec].abn[gridp]
                            + gridpoint[evnr].density*species[spec].abn[evnr] ) / 2.0;


    /* Numerical integration along the ray (line of sight) */

    for (long e=1; e<etot; e++){


#     ifdef ON_THE_FLY

      long evnr       = LOCAL_GP_NR_OF_EVALP(ray,e);
      long evnrp      = LOCAL_GP_NR_OF_EVALP(ray,e-1);
      long gridp_evnr = evnr;

      // if (gridp == 95) {printf("%ld,   %ld\n", e, evnr);}

#     else

      long evnr       = GP_NR_OF_EVALP(gridp,ray,e);
      long evnrp      = GP_NR_OF_EVALP(gridp,ray,e-1);
      long gridp_evnr = GINDEX(gridp,evnr);

      // if (gridp == 95) {printf("%ld,   %ld\n", e, evnr);}

#     endif


      column_density_res = column_density_res
                           + evalpoint[gridp_evnr].dZ * PC
                             * ( gridpoint[evnrp].density*species[spec].abn[evnrp]
                                 + gridpoint[evnr].density*species[spec].abn[evnr] ) / 2.0;

    } /* end of e loop over evaluation points */

  }


  return column_density_res;

}

/*-----------------------------------------------------------------------------------------------*/
