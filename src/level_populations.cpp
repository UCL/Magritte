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
#include "radiative_transfer.hpp"
#include "level_population_solver.hpp"



/* level_populations: iteratively calculates the level populations                               */
/*-----------------------------------------------------------------------------------------------*/

void level_populations( long *antipod, GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                        int *irad, int*jrad, double *frequency, double *A_coeff,
                        double *B_coeff, double *C_coeff, double *P_intensity,
                        double *R, double *pop, double *dpop, double *C_data,
                        double *coltemp, int *icol, int *jcol, double *temperature,
                        double *weight, double *energy, int lspec, bool sobolev )
{

  int n;                                                                     /* grid point index */

  int i, j;                                                                     /* level indices */

  int kr;                                                                    /* transition index */

  long r1, r2;                                                                      /* ray index */
  long ar1;                                                      /* index of antipodal ray to r1 */
  long n1, n2, n3;                                                         /* grid point indices */
  long e1, e2;                                                         /* evaluation point index */
  long etot1, etot2;                            /* total number of evaluation points along a ray */
  long ndep;           /* number of depth points along a pair of antipodal rays (=etot1+etot2-2) */
  long ndepav=0;                       /* average number of depth points used in exact_feautrier */
  long nav=0;                                     /* number of times ecact_feautrier is executed */

  long nshortcuts;                                      /* number of times the shortcut is taken */
  long nno_shortcuts;                               /* number of times the shortcut is not taken */

  int niterations=0;                                                     /* number of iterations */

  bool not_converged=true;         /* is true when the level population have not converged (yet) */

  double dpoprel;                          /* relative change in the level population (dpop/pop) */

  double Source[NGRID*TOT_NRAD];                                              /* source function */

  double opacity[NGRID*TOT_NRAD];                                                     /* opacity */

  double R_temp[NGRID*TOT_NLEV2];                     /* temporary storage the transition matrix */

  double mean_intensity[NGRID*TOT_NRAD];                             /* mean intensity for a ray */


  double hv_4pi;                                                      /* photon energy over 4 pi */



  /* First term in transition matrix R for each grid point */

  for (n=0; n<NGRID; n++){


    /* Calculate collisional (C) coefficients for current temperature */

    void calc_C_coeff( double *C_data, double *coltemp, int *icol, int *jcol, double *temperature,
                       double *weight, double *energy, double *C_coeff, long n, int lspec );

    calc_C_coeff( C_data, coltemp, icol, jcol, temperature, weight, energy, C_coeff, n, lspec );


    /* Initialize transition matrix R_ij with terms that do not depend on level populations */

    for (i=0; i<nlev[lspec]; i++){

       for (j=0; j<nlev[lspec]; j++){

         R_temp[LSPECGRIDLEVLEV(lspec,n,i,j)] = A_coeff[LSPECLEVLEV(lspec,i,j)]
                                                + C_coeff[LSPECLEVLEV(lspec,i,j)] ;
       }
    }
  }



  /* Iterate until the level populations converge */

  while( not_converged ){

    not_converged = false;

    niterations = niterations + 1;


    /* Initialize R with the stored values in R_temp */

    for (n=0; n<NGRID; n++){

      for (i=0; i<nlev[lspec]; i++){

        for (j=0; j<nlev[lspec]; j++){

          R[LSPECGRIDLEVLEV(lspec,n,i,j)] = R_temp[LSPECGRIDLEVLEV(lspec,n,i,j)];
        }
      }
    }


    /* Calculate source function and opacity for all gridpoints */

    for (n1=0; n1<NGRID; n1++){

      for (kr=0; kr<nrad[lspec]; kr++){

        i = irad[LSPECRAD(lspec,kr)];                   /* i index corresponding to transition kr */
        j = jrad[LSPECRAD(lspec,kr)];                   /* j index corresponding to transition kr */

        hv_4pi = HH * frequency[LSPECLEVLEV(lspec,i,j)] / 4.0 / PI;

        Source[LSPECGRIDRAD(lspec,n1,kr)] = 0.0;
        opacity[LSPECGRIDRAD(lspec,n1,kr)] = 0.0;


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


Source[LSPECGRIDRAD(0,20,0)] = 1.0E-5;
Source[LSPECGRIDRAD(0,20,1)] = 1.0E-5;
Source[LSPECGRIDRAD(0,20,2)] = 1.0E-5;







    /* Calculate and add the B_ij<J_ij> term */


    /* For all radiative transitions transitions */

    for (kr=0; kr<nrad[lspec]; kr++){

      i = irad[LSPECRAD(lspec,kr)];               /* i level index corresponding to transition kr */
      j = jrad[LSPECRAD(lspec,kr)];               /* j level index corresponding to transition kr */


      /* Initialize intensities */

      for (n=0; n<NGRID; n++){

        for (r2=0; r2<NRAYS; r2++){

          P_intensity[RINDEX(n,r2)] = 0.0;
        }
      }

      nshortcuts = 0;
      nno_shortcuts = 0;


      /* For all grid points */

      for (n2=0; n2<NGRID; n2++){

        mean_intensity[LSPECGRIDRAD(lspec,n2,kr)] = 0.0;


        /* Calculate the mean intensity */

        void radiative_transfer( long *antipod, EVALPOINT *evalpoint, double *P_intensity,
                                 double *mean_intensity, double *Source, double *opacity,
                                 int *irad, int*jrad, long n2, int lspec, int kr,
                                 long *nshortcuts, long *nno_shortcuts, bool sobolev );

        radiative_transfer( antipod, evalpoint, P_intensity, mean_intensity, Source, opacity,
                            irad, jrad, n2, lspec, kr, &nshortcuts, &nno_shortcuts, sobolev );


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



    /* Solve the equilibrium equation at each point */

    for (n3=0; n3<NGRID; n3++){


      /* Solve the radiative balance equation for the level populations */

      void level_population_solver( GRIDPOINT *gridpoint, double *R, double *pop, double *dpop,
                                    long n3, int lspec);

      level_population_solver( gridpoint, R, pop, dpop, n3, lspec );


      /* Check for convergence */

      for (i=0; i<nlev[lspec]; i++){


        /* Avoid too small numbers */

        if (pop[LSPECGRIDLEV(lspec,n3,i)] < 1.0E-18){

          pop[LSPECGRIDLEV(lspec,n3,i)] = 0.0;
        }


        if ( pop[LSPECGRIDLEV(lspec,n3,i)] != 0.0 ){

          dpoprel = dpop[LSPECGRIDLEV(lspec,n3,i)] / (pop[LSPECGRIDLEV(lspec,n3,i)]+dpop[LSPECGRIDLEV(lspec,n3,i)]);

          // printf("(level_populations): dpop/pop is %.2lE for grid point %ld \n", dpoprel, n3);

          /* If the population of any of the levels is not converged */

          if (dpoprel > POP_PREC){

            not_converged = true;

             // printf("(level_populations): dpop/pop is %.2lE for grid point %ld \n", dpoprel, n3);

          }
        }
      }


    } /* end of n3 loop over grid points */


    /* Limit the number of iterations */

    if (niterations >= MAX_NITERATIONS){
      not_converged = false;
    }

  } /* end of while loop of iterations */




  /* Write the mean intensity at each point for each transition to a file (only for testing) */

  FILE *meanintensity = fopen("output/mean_intensity.txt", "w");

  if (meanintensity == NULL){

      printf("Error opening file!\n");
      exit(1);
    }


  for (kr=0; kr<nrad[lspec]; kr++){

    for (n=0; n<NGRID; n++){

      fprintf( meanintensity, "%lE\t", mean_intensity[LSPECGRIDRAD(lspec,n,kr)] );
    }

    fprintf( meanintensity, "\n" );
  }

  fclose(meanintensity);



  printf( "(level_populations): nshortcuts = %ld, nno_shortcuts = %ld \n",
            nshortcuts, nno_shortcuts);

  printf( "(level_populations): nshortcuts/(nshortcuts+nno_shortcuts) = %.5lf \n",
            (double) nshortcuts/(nshortcuts+nno_shortcuts) );


  printf( "(level_populations): population levels for %d converged after %d iterations\n"
          "                     with precision %.1lE\n", lspec, niterations, POP_PREC );

}

/*-----------------------------------------------------------------------------------------------*/
