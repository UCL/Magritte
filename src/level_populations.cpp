/* Frederik De Ceuster - University College London                                               */
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
#include <stdlib.h>
#include <math.h>

#include "declarations.hpp"
#include "level_populations.hpp"
#include "calc_C_coeff.hpp"
#include "initializers.hpp"
#include "radiative_transfer.hpp"
#include "level_population_solver.hpp"



/* level_populations: iteratively calculates the level populations                               */
/*-----------------------------------------------------------------------------------------------*/

void level_populations( long *antipod, GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                        int *irad, int*jrad, double *frequency,
                        double *A_coeff, double *B_coeff, double *C_coeff,
                        double *R, double *pop, double *dpop, double *C_data,
                        double *coltemp, int *icol, int *jcol, double *temperature_gas,
                        double *weight, double *energy )
{


  long nshortcuts = 0;                                  /* number of times the shortcut is taken */

  long nno_shortcuts = 0;                           /* number of times the shortcut is not taken */


  double Source[NGRID*TOT_NRAD];                                              /* source function */

  initialize_double_array(Source, NGRID*TOT_NRAD);

  double opacity[NGRID*TOT_NRAD];                                                     /* opacity */

  initialize_double_array(opacity, NGRID*TOT_NRAD);

  double mean_intensity[NGRID*TOT_NRAD];                             /* mean intensity for a ray */

  initialize_double_array(mean_intensity, NGRID*TOT_NRAD);



  /* For each line producing species */

  for (int lspec=0; lspec<NLSPEC; lspec++){


    /* DEFINE R_temp AS THE TERMS THAT DO NOT DEPEND ON THE LEVEL POPULATIONS                    */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    double R_temp[NGRID*TOT_NLEV2];                   /* temporary storage the transition matrix */


    /* For all grid points */

    for (long n=0; n<NGRID; n++){


      /* Calculate collisional (C) coefficients for temperature_gas */

      calc_C_coeff( C_data, coltemp, icol, jcol, temperature_gas,
                    weight, energy, C_coeff, n, lspec );


      for (int i=0; i<nlev[lspec]; i++){

        for (int j=0; j<nlev[lspec]; j++){

          R_temp[LSPECGRIDLEVLEV(lspec,n,i,j)] = A_coeff[LSPECLEVLEV(lspec,i,j)]
                                                 + C_coeff[LSPECLEVLEV(lspec,i,j)] ;
        }
      }

    } /* end of n loop over grid points */


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




    /* ITERATE UNTIL LEVEL POPULATIONS CONVERGE                                                  */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    bool populations_not_converged = true;       /* true when level population are not converged */

    int niterations = 0;                                                 /* number of iterations */



    while( populations_not_converged ){

      populations_not_converged = false;

      niterations++;


      /* Initialize R with the terms (stored in R_temp) which do not depend on level populations */

      for (long n=0; n<NGRID; n++){

        for (int i=0; i<nlev[lspec]; i++){

          for (int j=0; j<nlev[lspec]; j++){

            R[LSPECGRIDLEVLEV(lspec,n,i,j)] = R_temp[LSPECGRIDLEVLEV(lspec,n,i,j)];
          }
        }
      }


      /* Calculate source function and opacity for all gridpoints                                */
      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


      for (int n1=0; n1<NGRID; n1++){

        for (int kr=0; kr<nrad[lspec]; kr++){

          int i = irad[LSPECRAD(lspec,kr)];            /* i index corresponding to transition kr */
          int j = jrad[LSPECRAD(lspec,kr)];            /* j index corresponding to transition kr */

          double hv_4pi = HH * frequency[LSPECLEVLEV(lspec,i,j)] / 4.0 / PI;


          if (pop[LSPECGRIDLEV(lspec,n1,j)] > 1.0E-30 || pop[LSPECGRIDLEV(lspec,n1,i)] > 1.0E-30){

            Source[LSPECGRIDRAD(lspec,n1,kr)]
                  = ( A_coeff[LSPECLEVLEV(lspec,i,j)] * pop[LSPECLEVLEV(lspec,n1,i)] )
                    /( pop[LSPECGRIDLEV(lspec,n1,j)]*B_coeff[LSPECLEVLEV(lspec,j,i)]
                       - pop[LSPECGRIDLEV(lspec,n1,i)]*B_coeff[LSPECLEVLEV(lspec,i,j)] ) ;

            opacity[LSPECGRIDRAD(lspec,n1,kr)]
                   =  hv_4pi * ( pop[LSPECGRIDLEV(lspec,n1,j)]*B_coeff[LSPECLEVLEV(lspec,j,i)]
                                 - pop[LSPECGRIDLEV(lspec,n1,i)]*B_coeff[LSPECLEVLEV(lspec,i,j)] );
          }

          Source[LSPECGRIDRAD(lspec,n1,kr)] = Source[LSPECGRIDRAD(lspec,n1,kr)] + 0.0E-20;

          opacity[LSPECGRIDRAD(lspec,n1,kr)] = opacity[LSPECGRIDRAD(lspec,n1,kr)] + 1.0E-1;
        }

      } /* end of n1 loop over gridpoints */


      // Source[LSPECGRIDRAD(0,0,0)] = 1.0E-5;
      // Source[LSPECGRIDRAD(0,0,1)] = 1.0E-5;
      // Source[LSPECGRIDRAD(0,0,2)] = 1.0E-5;


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

        for (int n2=0; n2<NGRID; n2++){

          mean_intensity[LSPECGRIDRAD(lspec,n2,kr)] = 0.0;


          /* Calculate the mean intensity */

          radiative_transfer( antipod, evalpoint, P_intensity, mean_intensity, Source, opacity,
                              irad, jrad, n2, lspec, kr, &nshortcuts, &nno_shortcuts );


          /* Fill the i>j part (since we loop over the transitions i -> j) */

          R[LSPECGRIDLEVLEV(lspec,n2,i,j)] = R[LSPECGRIDLEVLEV(lspec,n2,i,j)]
                                             + B_coeff[LSPECLEVLEV(lspec,i,j)]
                                               *mean_intensity[LSPECGRIDRAD(lspec,n2,kr)];


          /* Add the j>i part */

          R[LSPECGRIDLEVLEV(lspec,n2,j,i)] = R[LSPECGRIDLEVLEV(lspec,n2,j,i)]
                                             + B_coeff[LSPECLEVLEV(lspec,j,i)]
                                               *mean_intensity[LSPECGRIDRAD(lspec,n2,kr)];

          // printf("Mean intensity is %lE \n", mean_intensity[LSPECGRIDRAD(lspec,n2,kr)]);


        } /* end of n2 loop over grid points */

      } /* end of kr loop over radiative transitions */


      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/




      /* Solve the equilibrium equation at each point */
      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


      double dpoprel;                      /* relative change in the level population (dpop/pop) */



      for (int n3=0; n3<NGRID; n3++){


        /* Solve the radiative balance equation for the level populations */

        level_population_solver( gridpoint, R, pop, dpop, n3, lspec );


        /* Check for convergence */

        for (int i=0; i<nlev[lspec]; i++){


          /* Avoid too small numbers */

          if (pop[LSPECGRIDLEV(lspec,n3,i)] < 1.0E-18){

            pop[LSPECGRIDLEV(lspec,n3,i)] = 0.0;
          }


          if ( pop[LSPECGRIDLEV(lspec,n3,i)] != 0.0 ){

            dpoprel = dpop[LSPECGRIDLEV(lspec,n3,i)]
                      / (pop[LSPECGRIDLEV(lspec,n3,i)]+dpop[LSPECGRIDLEV(lspec,n3,i)]);

            // printf("(level_populations): dpop/pop is %.2lE for grid point %ld \n", dpoprel, n3);


            /* If the population of any of the levels is not converged */

            if (dpoprel > POP_PREC){

              populations_not_converged = true;

              // printf("(level_populations): dpop/pop is %.2lE for grid point %ld \n", dpoprel, n3);

            }
          }
        }


      } /* end of n3 loop over grid points */


      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


      /* Limit the number of iterations */

      if (niterations >= MAX_NITERATIONS){

        populations_not_converged = false;
      }

    } /* end of while loop of iterations */


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    /* Write the mean intensity at each point for each transition to a file (only for testing) */

    // FILE *meanintensity = fopen("output/mean_intensity.txt", "w");
    //
    // if (meanintensity == NULL){
    //
    //   printf("Error opening file!\n");
    //   exit(1);
    // }
    //
    //
    // for (kr=0; kr<nrad[lspec]; kr++){
    //
    //   for (n=0; n<NGRID; n++){
    //
    //     fprintf( meanintensity, "%lE\t", mean_intensity[LSPECGRIDRAD(lspec,n,kr)] );
    //   }
    //
    //   fprintf( meanintensity, "\n" );
    // }
    //
    // fclose(meanintensity);



    /* PRINT THE STATS OF THE CALCULATIONS FOR lspec */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    printf( "(level_populations): nshortcuts = %ld, nno_shortcuts = %ld \n",
            nshortcuts, nno_shortcuts);

    printf( "(level_populations): nshortcuts/(nshortcuts+nno_shortcuts) = %.5lf \n",
            (double) nshortcuts/(nshortcuts+nno_shortcuts) );

    printf( "(level_populations): population levels for %d converged after %d iterations\n"
            "                     with precision %.1lE\n", lspec, niterations, POP_PREC );

    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  } /* end of lspec loop over line producing species */

}

/*-----------------------------------------------------------------------------------------------*/
