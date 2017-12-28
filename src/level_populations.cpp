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

#include "level_populations.hpp"
#include "acceleration_Ng.hpp"
#include "calc_C_coeff.hpp"
#include "initializers.hpp"
#include "radiative_transfer.hpp"
#include "level_population_solver.hpp"
#include "ray_tracing.hpp"
#include "write_output.hpp"



#if !( ON_THE_FLY )

/* level_populations: iteratively calculates the level populations                               */
/*-----------------------------------------------------------------------------------------------*/

int level_populations( CELL *cell, EVALPOINT *evalpoint, long *key, long *raytot,
                       long *cum_raytot, int *irad, int*jrad, double *frequency, double *A_coeff,
                       double *B_coeff, double *R, double *pop, double *C_data, double *coltemp,
                       int *icol, int *jcol, double *temperature_gas, double *temperature_dust,
                       double *weight, double *energy, double *mean_intensity,
                       double *Lambda_diagonal, double *mean_intensity_eff )
{


  double prev1_pop[NCELLS*TOT_NLEV];                      /* level population n_i 1 iteration ago */
  double prev2_pop[NCELLS*TOT_NLEV];                     /* level population n_i 2 iterations ago */
  double prev3_pop[NCELLS*TOT_NLEV];                     /* level population n_i 3 iterations ago */

  long nshortcuts = 0;                                  /* number of times the shortcut is taken */

  long nno_shortcuts = 0;                           /* number of times the shortcut is not taken */


  /* For each line producing species */

  for (int lspec=0; lspec<NLSPEC; lspec++){


    /* DEFINE R_temp AS THE TERMS THAT DO NOT DEPEND ON THE LEVEL POPULATIONS                    */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    double R_temp[NCELLS*TOT_NLEV2];                   /* temporary storage the transition matrix */


    /* For all grid points */

#   pragma omp parallel                                                                          \
    shared( cell, C_data, coltemp, icol, jcol, temperature_gas, weight, energy,             \
            lspec, nlev, R_temp, A_coeff, B_coeff, cum_nlev2 )                                   \
    default( none )
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;     /* Note the brackets are important here */


    for (long n=start; n<stop; n++){

      double C_coeff[TOT_NLEV2];                                    /* Einstein C_ij coefficient */


      /* Calculate collisional (C) coefficients for temperature_gas */

      calc_C_coeff( cell, C_data, coltemp, icol, jcol, temperature_gas,
                    weight, energy, C_coeff, n, lspec );


      for (int i=0; i<nlev[lspec]; i++){

        for (int j=0; j<nlev[lspec]; j++){

          long r_ij = LSPECGRIDLEVLEV(lspec,n,i,j);                               /* R_tmp index */
          long b_ij = LSPECLEVLEV(lspec,i,j);                       /* A_coeff and C_coeff index */


          R_temp[r_ij] = A_coeff[b_ij] + C_coeff[b_ij];

        }
      }


    } /* end of n loop over grid points */
    } /* end of OpenMP parallel region */


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/





    /* ITERATE UNTIL LEVEL POPULATIONS CONVERGE                                                  */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    bool not_converged = true;       /* true when popualations of a grid point are not converged */

    int niterations = 0;                                                 /* number of iterations */



    while (not_converged){

      not_converged =  false;

      niterations++;


      printf( "(level_populations): Iteration %d for %s\n",
              niterations, species[lspec_nr[lspec]].sym.c_str() );


      long n_not_converged = 0;              /* number of grid points that are not yet converged */



      if ( ACCELERATION_POP_NG && (niterations%4 == 0) ){

        acceleration_Ng(lspec, prev3_pop, prev2_pop, prev1_pop, pop);
      }



      /* Store the populations of the previous 3 iterations */

      store_populations(lspec, prev3_pop, prev2_pop, prev1_pop, pop);



      /* Set R equal to R_temp */

      initialize_double_array_with( R, R_temp, NCELLS*TOT_NLEV2 );


      /* Initialize the Source and opacity with zero's */

      double Source[NCELLS*TOT_NRAD];                                          /* source function */

      double opacity[NCELLS*TOT_NRAD];                                                 /* opacity */


      /* Calculate source function and opacity for all cells                                */
      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


#     pragma omp parallel                                                                         \
      shared( nrad, lspec, irad, jrad, Source, opacity, pop, frequency, A_coeff, B_coeff, nlev,   \
              cum_nrad, cum_nlev, cum_nlev2 )                                                     \
      default( none )
      {

      int num_threads = omp_get_num_threads();
      int thread_num  = omp_get_thread_num();

      long start = (thread_num*NCELLS)/num_threads;
      long stop  = ((thread_num+1)*NCELLS)/num_threads;   /* Note the brackets are important here */


      for (long n=start; n<stop; n++){

        for (int kr=0; kr<nrad[lspec]; kr++){


          int i     = irad[LSPECRAD(lspec,kr)];        /* i index corresponding to transition kr */
          int j     = jrad[LSPECRAD(lspec,kr)];        /* j index corresponding to transition kr */

          long s_ij = LSPECGRIDRAD(lspec,n,kr);                      /* Source and opacity index */

          long b_ij = LSPECLEVLEV(lspec,i,j);            /* A_coeff, B_coeff and frequency index */
          long b_ji = LSPECLEVLEV(lspec,j,i);            /* A_coeff, B_coeff and frequency index */

          long p_i  = LSPECGRIDLEV(lspec,n,i);                                      /* pop index */
          long p_j  = LSPECGRIDLEV(lspec,n,j);                                      /* pop index */


          double hv_4pi = HH * frequency[b_ij] / 4.0 / PI;


          if (pop[p_j] > POP_LOWER_LIMIT || pop[p_i] > POP_LOWER_LIMIT){


            Source[s_ij]  = (A_coeff[b_ij] * pop[p_i])
                            / (pop[p_j]*B_coeff[b_ji] - pop[p_i]*B_coeff[b_ij]);



            opacity[s_ij] =  hv_4pi * (pop[p_j]*B_coeff[b_ji] - pop[p_i]*B_coeff[b_ij]);

          }

          else {

            Source[s_ij]  = 0.0;
            opacity[s_ij] = 1.0E-99;
          }


          if (opacity[s_ij] < 1.0E-99){

            opacity[s_ij] = 1.0E-99;
          }


        } /* end of kr loop over transitions */

      } /* end of n loop over cells */
      } /* end of OpenMP parallel region */


      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/




      /* Calculate and add the B_ij<J_ij> term */
      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


      /* For all radiative transitions transitions */

      for (int kr=0; kr<nrad[lspec]; kr++){


        int i = irad[LSPECRAD(lspec,kr)];        /* i level index corresponding to transition kr */
        int j = jrad[LSPECRAD(lspec,kr)];        /* j level index corresponding to transition kr */


        double P_intensity[NCELLS*NRAYS];                 /* Feautrier's mean intensity for a ray */

        initialize_double_array(P_intensity, NCELLS*NRAYS);


        nshortcuts = 0;

        nno_shortcuts = 0;


        /* For all grid points */

#       pragma omp parallel                                                                       \
        shared( evalpoint, key, raytot, cum_raytot, i, j, nlev, kr, nrad, lspec, irad, jrad,      \
                Source, opacity, pop, frequency, A_coeff, B_coeff, temperature_gas,               \
                temperature_dust, mean_intensity, Lambda_diagonal, R, mean_intensity_eff,         \
                P_intensity, cell, nno_shortcuts, nshortcuts, cum_nrad, cum_nlev, cum_nlev2 )\
        default( none )
        {

        int num_threads = omp_get_num_threads();
        int thread_num  = omp_get_thread_num();

        long start = (thread_num*NCELLS)/num_threads;
        long stop  = ((thread_num+1)*NCELLS)/num_threads;      /* Note the brackets are important */


        for (long n=start; n<stop; n++){


          long r_ij = LSPECGRIDLEVLEV(lspec,n,i,j);
          long r_ji = LSPECGRIDLEVLEV(lspec,n,j,i);

          long b_ij = LSPECLEVLEV(lspec,i,j);
          long b_ji = LSPECLEVLEV(lspec,j,i);

          long m_ij = LSPECGRIDRAD(lspec,n,kr);

          mean_intensity[m_ij] = 0.0;


          /* Calculate the mean intensity */

          radiative_transfer( cell, evalpoint, key, raytot, cum_raytot,
                              P_intensity, mean_intensity, Lambda_diagonal, mean_intensity_eff,
                              Source, opacity, frequency, temperature_gas, temperature_dust,
                              irad, jrad, n, lspec, kr, &nshortcuts, &nno_shortcuts );


          /* Fill the i>j part */

          R[r_ij] = R[r_ij] - A_coeff[b_ij]*Lambda_diagonal[m_ij]
                            + B_coeff[b_ij]*mean_intensity_eff[m_ij];


          /* Add the j>i part */

          R[r_ji] = R[r_ji] + B_coeff[b_ji]*mean_intensity_eff[m_ij];


        } /* end of n loop over grid points */
        } /* end of OpenMP parallel region */

      } /* end of kr loop over radiative transitions */


      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/




      /* Solve the equilibrium equation at each point */
      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


#     pragma omp parallel                                                                         \
      shared( cell, lspec, R, pop, prev1_pop, nlev, cum_nlev,not_converged,                  \
              n_not_converged, lspec_nr, species )                                                \
      default( none )
      {

      int num_threads = omp_get_num_threads();
      int thread_num  = omp_get_thread_num();

      long start = (thread_num*NCELLS)/num_threads;
      long stop  = ((thread_num+1)*NCELLS)/num_threads;      /* Note the brackets are important */


      for (long n=start; n<stop; n++){


        /* Solve the radiative balance equation for the level populations */

        level_population_solver( cell, n, lspec, R, pop );


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

      } /* end of n loop over grid points */
      } /* end of OpenMP parallel region */


      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


      /* Limit the number of iterations */

      if (niterations > MAX_NITERATIONS || n_not_converged < NCELLS/10){

        not_converged = false;
      }


      printf("(level_populations): Not yet converged for %ld of %d\n", n_not_converged, NCELLS);


    } /* end of while loop of iterations */


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




    /* PRINT THE STATS OF THE CALCULATIONS FOR lspec */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    printf( "(level_populations): nshortcuts = %ld, nno_shortcuts = %ld \n",
            nshortcuts, nno_shortcuts);

    printf( "(level_populations): nshortcuts/(nshortcuts+nno_shortcuts) = %.5lf \n",
            (double) nshortcuts/(nshortcuts+nno_shortcuts) );

    printf( "(level_populations): population levels for %s converged after %d iterations\n"
            "                     with precision %.1lE\n",
            species[lspec_nr[lspec]].sym.c_str(), niterations, POP_PREC );

    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  } /* end of lspec loop over line producing species */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/

#endif
