/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* radiative_transfer.c: Calculate mean intensity by solving transfer equation along each ray    */
/*                                                                                               */
/* (based on ITER in the SMMOL code)                                                             */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "exact_feautrier.cpp"



/* radiative_transfer: calculate mean intensity at grid point "gridp", by solving the transfer   */
/*                     equation along all pairs of a rays and their antipodals                   */
/*-----------------------------------------------------------------------------------------------*/

void radiative_transfer( long *antipod, EVALPOINT *evalpoint, double *P_intensity,
                         double *mean_intensity, double *Source, double *opacity,
                         int *irad, int*jrad, long gridp, int lspec, int kr, 
                         long *nshortcuts, long *nno_shortcuts, bool sobolev )
{

  int i, j;                                       /* indices for the population level n_i or n_j */
 
  long n;                                                                    /* grid point index */
  long r, r1;                                                                     /* ray indices */
  long ar1;                                                      /* index of antipodal ray to r1 */
  long e1, e2;                                                         /* evaluation point index */
  long etot1, etot2;                            /* total number of evaluation points along a ray */
  long ndep;           /* number of depth points along a pair of antipodal rays (=etot1+etot2-2) */
  long ndepav=0;                       /* average number of depth points used in exact_feautrier */
  long nav=0;                                     /* number of times exact_feautrier is executed */

  double boundary_condition=0.0;                        /* Intensity at the boundary of the grid */

  double dtau_test;                     /* store dtau here untill we checked it is not too small */

  long temp_sc;
  long temp_nsc;

  i = irad[LSPECRAD(lspec,kr)];                   /* i level index corresponding to transition kr */
  j = jrad[LSPECRAD(lspec,kr)];                   /* j level index corresponding to transition kr */

  

  /* Calculate mean_intensity <J_ij> */

  for (r1=0; r1<NRAYS/2; r1++){

    temp_sc = *nshortcuts;
    temp_nsc = *nno_shortcuts;


    /* Get the antipodal ray for r1 */

    ar1 = antipod[r1];


    /* Check if intensity is already calculated in an equivalent ray */

    if (P_intensity[RINDEX(gridp,r1)] > 0.0){
        
      *nshortcuts = temp_sc + 1;

      // printf("Shortcut taken! \n");
      mean_intensity[LSPECGRIDRAD(lspec,gridp,kr)] = mean_intensity[LSPECGRIDRAD(lspec,gridp,kr)]
                                                     + P_intensity[RINDEX(gridp,r1)];
    }

    else if (P_intensity[RINDEX(gridp,ar1)] > 0.0){

      *nshortcuts = temp_sc + 1;

      // printf("Shortcut taken! \n");
      mean_intensity[LSPECGRIDRAD(lspec,gridp,kr)] = mean_intensity[LSPECGRIDRAD(lspec,gridp,kr)]
                                                     + P_intensity[RINDEX(gridp,ar1)];
    }

    else {





      /*   SOLVE TRANSFER EQUATION ALONG THE RAY
      /*_________________________________________________________________________________________*/


      /* Fill the source function and the optical depth increment along ray r1 */

      etot1 = raytot[RINDEX(gridp, ar1)];
      etot2 = raytot[RINDEX(gridp, r1)];


      if (etot1>0 && etot2>0){

        *nno_shortcuts = temp_nsc + 1;

        // printf("No shortcut for grid point %ld and ray %ld \n", gridp, r1);

        // printf("etot1 = %ld\n", etot1 );
        // printf("etot2 = %ld\n", etot2 );


        ndep = etot1 + etot2;


        /* Allocate memory for the source function and optical depth */

        double *S;                                             /* source function along this ray */
        S = (double*) malloc( ndep*sizeof(double) );

        double *dtau;                                            /* optical depth along this ray */
        dtau = (double*) malloc( ndep*sizeof(double) );


        /* For the antipodal ray to ray r1 */

        for (e1=1; e1<etot1; e1++){

          long ge1   = GP_NR_OF_EVALP(gridp, ar1, etot1-e1);
          long ge1m1 = GP_NR_OF_EVALP(gridp, ar1, etot1-e1-1);

          S[e1-1]    = ( Source[LSPECGRIDRAD(lspec,ge1,kr)]
                         + Source[LSPECGRIDRAD(lspec,ge1m1,kr)] ) / 2.0;

          dtau[e1-1] = evalpoint[GINDEX(gridp, ge1)].dZ
                       *( opacity[LSPECGRIDRAD(lspec,ge1,kr)]
                          + opacity[LSPECGRIDRAD(lspec,ge1m1,kr)] ) / 2.0;
        }


        /* Adding the grid point itself (the origin for both rays) */

        S[etot1-1]    = ( Source[LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, ar1, 0),kr)]
                          + Source[LSPECGRIDRAD(lspec,gridp,kr)] ) / 2.0;

        dtau[etot1-1] = evalpoint[GINDEX(gridp, GP_NR_OF_EVALP(gridp, ar1, 0))].dZ
                        *( opacity[LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, ar1, 0),kr)]
                           + opacity[LSPECGRIDRAD(lspec,gridp,kr)] ) / 2.0;

        S[etot1]      = ( Source[LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, r1, 0),kr)]
                               + Source[LSPECGRIDRAD(lspec,gridp,kr)] ) / 2.0;

        dtau[etot1]   = evalpoint[GINDEX(gridp, GP_NR_OF_EVALP(gridp, r1, 0))].dZ
                             *( opacity[LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, r1, 0),kr)]
                                + opacity[LSPECGRIDRAD(lspec,gridp, kr)] ) / 2.0;


        /* For ray r1 itself */

        for (e2=1; e2<etot2; e2++){

          long ge2   = GP_NR_OF_EVALP(gridp, r1, e2);
          long ge2m1 = GP_NR_OF_EVALP(gridp, r1, e2-1);

          S[etot1+e2]    = ( Source[LSPECGRIDRAD(lspec,ge2,kr)]
                             + Source[LSPECGRIDRAD(lspec,ge2m1,kr)] ) / 2.0;

          dtau[etot1+e2] = evalpoint[GINDEX(gridp, ge2)].dZ
                           *( opacity[LSPECGRIDRAD(lspec,ge2,kr)]
                              + opacity[LSPECGRIDRAD(lspec,ge2m1,kr)] ) / 2.0; 
        }

        

        /*-------------------------------------------------------------------------------------*/


        /*   Sobolev approximation   */
        /*   +++++++++++++++++++++   */

        /* (to compare with 3D-PDR) */

        /* NOTE: Make sure ray_separation2=0.0 when sobolev=true !!! */

        if (sobolev == true){

          for (n=0; n<ndep; n++){


            /* Source function is only non-zero at the point under consideration */

            if ( !(n==etot1-1) || (n==etot1-2) ){

              S[n] = 0.0;
            }

            printf("%lE\n", S[n]);

          }
        }


        /*-------------------------------------------------------------------------------------*/



        double exact_feautrier( long ndep, double *S, double *dtau, long etot1, long etot2,
                                EVALPOINT *evalpoint, double *P_intensity, long gridp,
                                long r1, long ar1 );

        mean_intensity[LSPECGRIDRAD(lspec,gridp,kr)]
          = mean_intensity[LSPECGRIDRAD(lspec,gridp,kr)]
            + exact_feautrier( ndep, S, dtau, etot1, etot2, evalpoint,
                               P_intensity, gridp, r1, ar1 );


        // printf("(radiative_transfer): number of depth points %ld\n", ndep);

        // printf( "(radiative_transfer): Feautrier P for ray %ld is %lE \n",
        //         r1, mean_intensity[LSPECGRIDRAD(lspec,gridp,kr)] );        


        ndepav = ndepav + ndep;
        nav = nav + 1;


        /* Free the allocated memory for temporary variables */

        free(S);
        free(dtau);

      } /* end of if etot1>1 && etot2>1 */


      /* Impose boundary conditions */

      else{

        mean_intensity[LSPECGRIDRAD(lspec,gridp,kr)]
          = mean_intensity[LSPECGRIDRAD(lspec,gridp,kr)] + boundary_condition;

      }

/*
      printf("(radiative_transfer): etot1 and etot2 are %ld and %ld \n", etot1, etot2);
*/

      /*_________________________________________________________________________________________*/





    } /* end of else (no shortcut) */

  } /* end of r1 loop over rays */


  mean_intensity[LSPECGRIDRAD(lspec,gridp,kr)]
    = mean_intensity[LSPECGRIDRAD(lspec,gridp,kr)] + 1.0E-13 ; /// 4.0 / PI;


  // printf( "(radiative_transfer): mean intensity at %ld is %lE \n",
  //         gridp, mean_intensity[LSPECGRIDRAD(lspec,gridp,kr)] );


  // printf("(radiative_transfer): average ndep %.2lf \n", (double) ndepav/nav);

}

/*-----------------------------------------------------------------------------------------------*/
