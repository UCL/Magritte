/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* level_populations.c: Build the statistical equilibrium equations for the level populations    */
/*                                                                                               */
/* (based on ITER in the SMMOL code)                                                             */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "level_populations_otf.hpp"
#include "acceleration_Ng.hpp"
#include "lines.hpp"
#include "calc_C_coeff.hpp"
#include "initializers.hpp"
#include "radiative_transfer_otf.hpp"
#include "level_population_solver_otf.hpp"
#include "ray_tracing.hpp"
#include "write_output.hpp"



#ifdef ON_THE_FLY

/* level_populations: iteratively calculates the level populations                               */
/*-----------------------------------------------------------------------------------------------*/

int level_populations_otf( GRIDPOINT *gridpoint,
                           int *irad, int*jrad, double *frequency,
                           double *A_coeff, double *B_coeff,
                           double *pop,
                           double *C_data, double *coltemp, int *icol, int *jcol,
                           double *temperature_gas, double *temperature_dust,
                           double *weight, double *energy, double *mean_intensity,
                           double *Lambda_diagonal, double *mean_intensity_eff )
{


  double prev1_pop[NGRID*TOT_NLEV];                      /* level population n_i 1 iteration ago */
  double prev2_pop[NGRID*TOT_NLEV];                     /* level population n_i 2 iterations ago */
  double prev3_pop[NGRID*TOT_NLEV];                     /* level population n_i 3 iterations ago */


  /* For each line producing species */

  for (int lspec=0; lspec<NLSPEC; lspec++){

    bool not_converged = true;       /* true when popualations of a grid point are not converged */

    int niterations = 0;                                                 /* number of iterations */


    /* Iterate until the level populations converge                                              */

    while (not_converged){

      not_converged =  false;

      niterations++;


      printf( "(level_populations): Iteration %d for %s\n",
              niterations, species[lspec_nr[lspec]].sym.c_str() );


      long n_not_converged = 0;              /* number of grid points that are not yet converged */


      /* Perform an Ng acceleration step every 4th iteration */

      if ( ACCELERATION_POP_NG && (niterations%4 == 0) ){

        acceleration_Ng(lspec, prev3_pop, prev2_pop, prev1_pop, pop);
      }


      /* Store the populations of the previous 3 iterations */

      store_populations(lspec, prev3_pop, prev2_pop, prev1_pop, pop);


      /* Calculate the source and opacity for all transitions over the whole grid */

      double source[NGRID*TOT_NRAD];                                          /* source function */

      line_source( irad, jrad, A_coeff, B_coeff, pop, lspec, source );

      double opacity[NGRID*TOT_NRAD];                                                 /* opacity */

      line_opacity( irad, jrad, frequency, B_coeff, pop, lspec, opacity );


      /* For every grid point */

      for (long n=0; n<NGRID; n++){


        double R[TOT_NLEV2];                                           /* Transition matrix R_ij */


        /* Calculate the collisional terms                                                       */
        /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


        double C_coeff[TOT_NLEV2];                                  /* Einstein C_ij coefficient */

        calc_C_coeff( gridpoint, C_data, coltemp, icol, jcol, temperature_gas,
                      weight, energy, C_coeff, n, lspec );


        /* Fill the first part of the transition matrix R */

        for (int i=0; i<nlev[lspec]; i++){

          for (int j=0; j<nlev[lspec]; j++){

            long b_ij = LSPECLEVLEV(lspec,i,j);                  /* R, A_coeff and C_coeff index */


            R[b_ij] = A_coeff[b_ij] + C_coeff[b_ij];
          }
        }


        /* Calculate and add the B_ij<J_ij> term */
        /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


        /* For all transitions */

        for (int kr=0; kr<nrad[lspec]; kr++){


          int i     = irad[LSPECRAD(lspec,kr)];        /* i index corresponding to transition kr */
          int j     = jrad[LSPECRAD(lspec,kr)];        /* j index corresponding to transition kr */

          long b_ij = LSPECLEVLEV(lspec,i,j);            /* A_coeff, B_coeff and frequency index */
          long b_ji = LSPECLEVLEV(lspec,j,i);            /* A_coeff, B_coeff and frequency index */

          double A_ij = A_coeff[b_ij];
          double B_ij = B_coeff[b_ij];
          double B_ji = B_coeff[b_ji];

          long m_ij = LSPECGRIDRAD(lspec,n,kr);

          mean_intensity[m_ij] = 0.0;


          long key[NGRID];            /* stores the nrs. of the grid points on the rays in order */

          long raytot[NRAYS];              /* cumulative nr. of evaluation points along each ray */

          long cum_raytot[NRAYS];          /* cumulative nr. of evaluation points along each ray */


          EVALPOINT evalpoint[NGRID];

          get_local_evalpoint(gridpoint, evalpoint, key, raytot, cum_raytot, n);


          /* Calculate the mean intensity */

          radiative_transfer_otf( gridpoint, evalpoint, key, raytot, cum_raytot,
                                  mean_intensity, Lambda_diagonal, mean_intensity_eff,
                                  source, opacity, frequency, temperature_gas, temperature_dust,
                                  irad, jrad, n, lspec, kr );


          /* Fill the i>j part */

          R[b_ij] = R[b_ij] - A_ij*Lambda_diagonal[m_ij] + B_ij*mean_intensity_eff[m_ij];


          /* Add the j>i part */

          R[b_ji] = R[b_ji] + B_ji*mean_intensity_eff[m_ij];

        } /* end of kr loop over transitions */


        /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/




        /* Solve the equilibrium equation at each point */
        /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


        /* Solve the radiative balance equation for the level populations */

        level_population_solver_otf( gridpoint, n, lspec, R, pop );


        /* Check for convergence */

        for (int i=0; i<nlev[lspec]; i++){


          long p_i = LSPECGRIDLEV(lspec,n,i);                              /* pop and dpop index */


          if ( ( pop[p_i] > 1.0E-10 * species[ lspec_nr[lspec] ].abn[n] )
               && !( pop[p_i]==0.0 && prev1_pop[p_i]==0.0 ) ){


            double dpoprel = 2.0 * fabs(pop[p_i] - prev1_pop[p_i]) / (pop[p_i] + prev1_pop[p_i]);


            /* If the population of any of the levels is not converged */

            if (dpoprel > POP_PREC){

              not_converged = true;

              n_not_converged++;
            }

          }


        } /* end of i loop over levels */


      } /* end of n loop over gridpoints */


      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


      /* Limit the number of iterations */

      if (niterations > MAX_NITERATIONS || n_not_converged < NGRID/10){

        not_converged = false;
      }


      printf("(level_populations): Not yet converged for %ld of %d\n", n_not_converged, NGRID);


    } /* end of while loop of iterations */


    /* Print the stats for the calculations on lspec */

    printf( "(level_populations): population levels for %s converged after %d iterations\n"
            "                     with precision %.1lE\n",
            species[lspec_nr[lspec]].sym.c_str(), niterations, POP_PREC );


  } /* end of lspec loop over line producing species */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/

#endif
