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
#include <signal.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "level_populations.hpp"
#include "calc_C_coeff.hpp"
#include "initializers.hpp"
#include "radiative_transfer.hpp"
#include "level_population_solver.hpp"
#include "write_output.hpp"



/* level_populations: iteratively calculates the level populations                               */
/*-----------------------------------------------------------------------------------------------*/

int level_populations( GRIDPOINT *gridpoint, EVALPOINT *evalpoint, long *antipod,
                       int *irad, int*jrad, double *frequency, double v_turb,
                       double *A_coeff, double *B_coeff, double *C_coeff,
                       double *R, double *pop, double *dpop, double *C_data,
                       double *coltemp, int *icol, int *jcol,
                       double *temperature_gas, double *temperature_dust,
                       double *weight, double *energy, double *mean_intensity )
{


  long nshortcuts = 0;                                  /* number of times the shortcut is taken */

  long nno_shortcuts = 0;                           /* number of times the shortcut is not taken */



  /* For each line producing species */

  for (int lspec=0; lspec<NLSPEC; lspec++){


    /* DEFINE R_temp AS THE TERMS THAT DO NOT DEPEND ON THE LEVEL POPULATIONS                    */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    double R_temp[NGRID*TOT_NLEV2];                   /* temporary storage the transition matrix */


    /* For all grid points */

    for (long n=0; n<NGRID; n++){


      /* Calculate collisional (C) coefficients for temperature_gas */

      calc_C_coeff( gridpoint, C_data, coltemp, icol, jcol, temperature_gas,
                    weight, energy, C_coeff, n, lspec );


      for (int i=0; i<nlev[lspec]; i++){

        for (int j=0; j<nlev[lspec]; j++){

          long r_ij = LSPECGRIDLEVLEV(lspec,n,i,j);                               /* R_tmp index */
          long b_ij = LSPECLEVLEV(lspec,i,j);                       /* A_coeff and C_coeff index */


          R_temp[r_ij] = A_coeff[b_ij] + C_coeff[b_ij];

        }
      }


    } /* end of n loop over grid points */


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/





    /* ITERATE UNTIL LEVEL POPULATIONS CONVERGE                                                  */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    bool some_not_converged = true;  /* true when popualations of a grid point are not converged */

    bool not_converged[NGRID];     /* true when populations of this grid point are not converged */

    initialize_bool(true, NGRID, not_converged);

    int niterations = 0;                                                 /* number of iterations */



    while( some_not_converged ){

      some_not_converged =  false;

      niterations++;


      printf( "(level_populations): Iteration %d for %s\n",
              niterations, species[lspec_nr[lspec]].sym.c_str() );


      long n_not_converged = 0;              /* number of grid points that are not yet converged */


      /* Set R equal to R_temp */

      initialize_double_array_with( R, R_temp, NGRID*TOT_NLEV2 );


      /* Initialize the Source and opacity with zero's */

      double Source[NGRID*TOT_NRAD];                                          /* source function */

      initialize_double_array(Source, NGRID*TOT_NRAD);

      double opacity[NGRID*TOT_NRAD];                                                 /* opacity */

      initialize_double_array(opacity, NGRID*TOT_NRAD);


      /* Calculate source function and opacity for all gridpoints                                */
      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


      for (long n=0; n<NGRID; n++){

        // if (not_converged[n]){

          for (int kr=0; kr<nrad[lspec]; kr++){


            int i     = irad[LSPECRAD(lspec,kr)];      /* i index corresponding to transition kr */
            int j     = jrad[LSPECRAD(lspec,kr)];      /* j index corresponding to transition kr */

            long s_ij = LSPECGRIDRAD(lspec,n,kr);                    /* Source and opacity index */

            long b_ij = LSPECLEVLEV(lspec,i,j);          /* A_coeff, B_coeff and frequency index */
            long b_ji = LSPECLEVLEV(lspec,j,i);          /* A_coeff, B_coeff and frequency index */

            long p_i  = LSPECGRIDLEV(lspec,n,i);                                    /* pop index */
            long p_j  = LSPECGRIDLEV(lspec,n,j);                                    /* pop index */


            double hv_4pi = HH * frequency[b_ij] / 4.0 / PI;


            if (pop[p_j] > POP_LOWER_LIMIT || pop[p_i] > POP_LOWER_LIMIT){


              Source[s_ij]  = (A_coeff[b_ij] * pop[p_i])
                              / (pop[p_j]*B_coeff[b_ji] - pop[p_i]*B_coeff[b_ij]);



              opacity[s_ij] =  hv_4pi * (pop[p_j]*B_coeff[b_ji] - pop[p_i]*B_coeff[b_ij]);

            }


            if (opacity[s_ij] < 1.0E-99){
              opacity[s_ij] = 1.0E-99;
            }


          } /* end of kr loop over transitions */

        // } /* end of if not converged */

      } /* end of n loop over gridpoints */


      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/




      /* Calculate and add the B_ij<J_ij> term */
      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


      /* For all radiative transitions transitions */

      for (int kr=0; kr<nrad[lspec]; kr++){


        int i = irad[LSPECRAD(lspec,kr)];        /* i level index corresponding to transition kr */
        int j = jrad[LSPECRAD(lspec,kr)];        /* j level index corresponding to transition kr */


        double P_intensity[NGRID*NRAYS];                 /* Feautrier's mean intensity for a ray */

        initialize_double_array(P_intensity, NGRID*NRAYS);


        nshortcuts = 0;

        nno_shortcuts = 0;


        /* For all grid points */

        for (long n=0; n<NGRID; n++){

          // if (not_converged[n]){

            long r_ij = LSPECGRIDLEVLEV(lspec,n,i,j);
            long r_ji = LSPECGRIDLEVLEV(lspec,n,j,i);

            long b_ij = LSPECLEVLEV(lspec,i,j);
            long b_ji = LSPECLEVLEV(lspec,j,i);

            long m_ij = LSPECGRIDRAD(lspec,n,kr);

            mean_intensity[m_ij] = 0.0;


            /* Calculate the mean intensity */

            radiative_transfer( gridpoint, evalpoint, antipod, P_intensity, mean_intensity,
                                Source, opacity, frequency, temperature_gas, temperature_dust,
                                irad, jrad, n, lspec, kr, v_turb, &nshortcuts, &nno_shortcuts );


            /* Fill the i>j part */

            R[r_ij] = R[r_ij] + B_coeff[b_ij]*mean_intensity[m_ij];


            /* Add the j>i part */

            R[r_ji] = R[r_ji] + B_coeff[b_ji]*mean_intensity[m_ij];


          // } /* end of if not converged */

        } /* end of n loop over grid points */

      } /* end of kr loop over radiative transitions */


      // write_R("1", 0, R);
      // write_R("2", 1, R);
      // write_R("50", 49, R);
      // write_R("75", 74, R);

      // raise(SIGABRT);


      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/




      /* Solve the equilibrium equation at each point */
      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


      for (long n=0; n<NGRID; n++){


        /* Save the previous populations */

        double previous_pop[nlev[lspec]];

        for (int i=0; i<nlev[lspec]; i++){

          previous_pop[i] = pop[LSPECGRIDLEV(lspec,n,i)];
        }


        /* Solve the radiative balance equation for the level populations */

        level_population_solver( gridpoint, n, lspec, R, pop );


        /* Check for convergence */

        if (not_converged[n]){

          not_converged[n] = false;


          for (int i=0; i<nlev[lspec]; i++){


            long p_i = LSPECGRIDLEV(lspec,n,i);                            /* pop and dpop index */


            if ( ( pop[p_i] > 1.0E-10 * species[ lspec_nr[lspec] ].abn[n] )
                 && !( pop[p_i]==0.0 && previous_pop[i]==0.0 ) ){


              double dpoprel = 2.0 * fabs(pop[p_i] - previous_pop[i]) / (pop[p_i] + previous_pop[i]);


              /* If the population of any of the levels is not converged */

              if (dpoprel > POP_PREC){

                not_converged[n]   = true;

                some_not_converged = true;

                n_not_converged++;

              }

            }


          } /* end of i loop over levels */

        } /* end of if not converged */

      } /* end of n loop over grid points */


      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


      /* Limit the number of iterations */

      if (niterations > MAX_NITERATIONS || n_not_converged < NGRID/10){

        some_not_converged = false;
      }


      printf("(level_populations): Not yet converged for %ld of %d\n", n_not_converged, NGRID);


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
