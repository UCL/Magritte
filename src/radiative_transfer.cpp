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

#include "declarations.hpp"
#include "radiative_transfer.hpp"
#include "exact_feautrier.hpp"



/* radiative_transfer: calculate mean intensity at grid point "gridp", by solving the transfer   */
/*                     equation along all pairs of a rays and their antipodals                   */
/*-----------------------------------------------------------------------------------------------*/

void radiative_transfer( long *antipod, EVALPOINT *evalpoint, double *P_intensity,
                         double *mean_intensity, double *Source, double *opacity,
                         int *irad, int*jrad, long gridp, int lspec, int kr,
                         long *nshortcuts, long *nno_shortcuts )
{


  long ndepav = 0;                     /* average number of depth points used in exact_feautrier */
  long nav = 0;                                   /* number of times exact_feautrier is executed */

  long m_ij = LSPECGRIDRAD(lspec,gridp,kr);               /* mean_intensity, S and opacity index */

  int i = irad[LSPECRAD(lspec,kr)];              /* i level index corresponding to transition kr */
  int j = jrad[LSPECRAD(lspec,kr)];              /* j level index corresponding to transition kr */



  /* For half of the rays (only half is needed since we also consider the antipodals) */

  for (long r=0; r<NRAYS/2; r++){

    long temp_sc  = *nshortcuts;
    long temp_nsc = *nno_shortcuts;


    /* Get the antipodal ray for r */

    long ar = antipod[r];                                         /* index of antipodal ray to r */


    /* Check if intensity is already calculated in an equivalent ray */

    if (P_intensity[RINDEX(gridp,r)] > 0.0){

      *nshortcuts = temp_sc + 1;


      mean_intensity[m_ij] = mean_intensity[m_ij] + P_intensity[RINDEX(gridp,r)];

    }

    else if (P_intensity[RINDEX(gridp,ar)] > 0.0){

      *nshortcuts = temp_sc + 1;


      mean_intensity[m_ij] = mean_intensity[m_ij] + P_intensity[RINDEX(gridp,ar)];

    }

    else {


      /*   SOLVE TRANSFER EQUATION ALONG THE RAY
      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


      /* Fill the source function and the optical depth increment along ray r */

      long etot1 = raytot[RINDEX(gridp, ar)];  /* total number of evaluation points along ray ar */

      long etot2 = raytot[RINDEX(gridp, r)];    /* total number of evaluation points along ray r */


      if (etot1>0 && etot2>0){

        *nno_shortcuts = temp_nsc + 1;

        long ndep = etot1 + etot2;         /* nr. of depth points along a pair of antipodal rays */


        /* Allocate memory for the source function and optical depth */

        double *S;                                             /* source function along this ray */
        S = (double*) malloc( ndep*sizeof(double) );

        double *dtau;                                            /* optical depth along this ray */
        dtau = (double*) malloc( ndep*sizeof(double) );


        /* For the antipodal ray to ray r */

        for (long e1=1; e1<etot1; e1++){


          long e_n  = GINDEX(gridp, GP_NR_OF_EVALP(gridp, ar, etot1-e1));

          long s_n  = LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, ar, etot1-e1),kr);
          long s_np = LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, ar, etot1-e1-1),kr);


          S[e1-1]    = (Source[s_n] + Source[s_np]) / 2.0;

          dtau[e1-1] = evalpoint[e_n].dZ * (opacity[s_n] + opacity[s_np]) / 2.0;

        }


        /* Adding the grid point itself (the origin for both rays) */

        long e_a0 = GINDEX(gridp, GP_NR_OF_EVALP(gridp, ar, 0));
        long e_0  = GINDEX(gridp, GP_NR_OF_EVALP(gridp, r, 0));

        long s_an = LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, ar, 0),kr);
        long s_n  = LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, r, 0),kr);


        S[etot1-1]    = (Source[s_an] + Source[m_ij]) / 2.0;

        dtau[etot1-1] = evalpoint[e_a0].dZ * (opacity[s_an] + opacity[m_ij]) / 2.0;

        S[etot1]      = (Source[s_n] + Source[m_ij]) / 2.0;

        dtau[etot1]   = evalpoint[e_0].dZ * (opacity[s_n] + opacity[m_ij]) / 2.0;


        /* For ray r itself */

        for (long e2=1; e2<etot2; e2++){


          long e_n  = GINDEX(gridp, GP_NR_OF_EVALP(gridp, r, e2));

          long s_n  = LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, r, e2),kr);
          long s_np = LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, r, e2-1),kr);


          S[etot1+e2]    = (Source[s_n] + Source[s_np]) / 2.0;

          dtau[etot1+e2] = evalpoint[e_n].dZ * (opacity[s_n] + opacity[s_np]) / 2.0;

        }


        /*-------------------------------------------------------------------------------------*/


        /*   Sobolev approximation   */
        /*   +++++++++++++++++++++   */

        /* (to compare with 3D-PDR) */


        if (SOBOLEV == true){


          /* NOTE: Make sure RAY_SEPARATION2=0.0 when SOBOLEV=true !!! */

          if (RAY_SEPARATION2 != 0.0){

            printf("\n\n !!! ERROR in ray tracing !!! \n\n");
            printf("   [ERROR]:   SOBOLEV = true   while   RAY_SEPARATION2 != 0.0 \n\n");
          }

          for (int n=0; n<ndep; n++){


            /* Source function is only non-zero at the point under consideration */

            if ( !(n==etot1-1) || (n==etot1-2) ){

              S[n] = 0.0;
            }

          }
        }


        /*-------------------------------------------------------------------------------------*/


        /* Solve the transfer equation wit hthe exact Feautrier solver */

        exact_feautrier( ndep, S, dtau, etot1, etot2, evalpoint, P_intensity, gridp, r, ar );


        mean_intensity[m_ij] = mean_intensity[m_ij] + P_intensity[RINDEX(gridp,r)];


        // printf("(radiative_transfer): number of depth points %ld\n", ndep);

        // printf( "P contribution to mean intensity %lE \n", mean_intensity[m_ij] );


        ndepav = ndepav + ndep;
        nav = nav + 1;


        /* Free the allocated memory for temporary variables */

        free(S);
        free(dtau);

      } /* end of if etot1>1 && etot2>1 */


      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


    } /* end of else (no shortcut) */

  } /* end of r loop over half of the rays */


  mean_intensity[m_ij] = mean_intensity[m_ij]; // / 4.0 / PI;


  // printf( "(radiative_transfer): mean intensity at gridp %ld for trans %d is %lE \n",
  //         gridp, kr, mean_intensity[m_ij] );


  // printf("(radiative_transfer): average ndep %.2lf \n", (double) ndepav/nav);

}

/*-----------------------------------------------------------------------------------------------*/
