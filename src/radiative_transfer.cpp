/* Frederik De Ceuster - University College London & KU Leuven                                   */
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
#include <iostream>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "radiative_transfer.hpp"
#include "feautrier.hpp"



/* radiative_transfer: calculate mean intensity at grid point "gridp", by solving the transfer   */
/*                     equation along all pairs of a rays and their antipodals                   */
/*-----------------------------------------------------------------------------------------------*/

void radiative_transfer( CELL *cell, EVALPOINT *evalpoint,
                         long *key, long *raytot, long *cum_raytot,
                         double *P_intensity, double *mean_intensity,
                         double *Lambda_diagonal, double *mean_intensity_eff,
                         double *Source, double *opacity, double *frequency,
                         double *temperature_gas, double *temperature_dust,
                         int *irad, int*jrad, long gridp, int lspec, int kr,
                         long *nshortcuts, long *nno_shortcuts )
{


  long ndepav = 0;                           /* average number of depth points used in feautrier */
  long nav = 0;                                         /* number of times feautrier is executed */

  long m_ij = LSPECGRIDRAD(lspec,gridp,kr);               /* mean_intensity, S and opacity index */

  int i = irad[LSPECRAD(lspec,kr)];              /* i level index corresponding to transition kr */
  int j = jrad[LSPECRAD(lspec,kr)];              /* j level index corresponding to transition kr */

  long b_ij = LSPECLEVLEV(lspec,i,j);                                         /* frequency index */

  double escape_probability = 0.0;          /* escape probability from the Sobolev approximation */


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


      /*   DO THE RADIATIVE TRANSFER
      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


      /* Fill the source function and the optical depth increment along ray r */


#     if ( ON_THE_FLY )

      long etot1 = raytot[ar];                 /* total number of evaluation points along ray ar */
      long etot2 = raytot[r];                   /* total number of evaluation points along ray r */

#     else

      long etot1 = raytot[RINDEX(gridp, ar)];  /* total number of evaluation points along ray ar */
      long etot2 = raytot[RINDEX(gridp, r)];    /* total number of evaluation points along ray r */

#     endif


      if ( etot1>0 || etot2>0 ){

        *nno_shortcuts = temp_nsc + 1;

        long ndep = etot1 + etot2;         /* nr. of depth points along a pair of antipodal rays */


        /* Allocate memory for the source function and optical depth */

        double *S;                                             /* source function along this ray */
        S = (double*) malloc( ndep*sizeof(double) );

        double *dtau;                                            /* optical depth along this ray */
        dtau = (double*) malloc( ndep*sizeof(double) );


        /* For the antipodal ray to ray r */

        if (etot1 > 1){
        for (long e1=1; e1<etot1; e1++){


#         if ( ON_THE_FLY )

          long e_n  = LOCAL_GP_NR_OF_EVALP(ar, etot1-e1);

          long s_n  = LSPECGRIDRAD(lspec,LOCAL_GP_NR_OF_EVALP(ar, etot1-e1),kr);
          long s_np = LSPECGRIDRAD(lspec,LOCAL_GP_NR_OF_EVALP(ar, etot1-e1-1),kr);

#         else

          long e_n  = GINDEX(gridp, GP_NR_OF_EVALP(gridp, ar, etot1-e1));

          long s_n  = LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, ar, etot1-e1),kr);
          long s_np = LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, ar, etot1-e1-1),kr);

#         endif


          S[e1-1]    = (Source[s_n] + Source[s_np]) / 2.0;

          dtau[e1-1] = evalpoint[e_n].dZ * PC * (opacity[s_n] + opacity[s_np]) / 2.0;

        }
        }


        /* Adding the piece that contains the origin for ar */

        if (etot1 > 0){


#         if ( ON_THE_FLY )

          long e_a0 = LOCAL_GP_NR_OF_EVALP(ar, 0);

          long s_an = LSPECGRIDRAD(lspec,LOCAL_GP_NR_OF_EVALP(ar, 0),kr);

#         else

          long e_a0 = GINDEX(gridp, GP_NR_OF_EVALP(gridp, ar, 0));

          long s_an = LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, ar, 0),kr);

#         endif


          S[etot1-1]    = (Source[s_an] + Source[m_ij]) / 2.0;

          dtau[etot1-1] = evalpoint[e_a0].dZ * PC * (opacity[s_an] + opacity[m_ij]) / 2.0;

        }


        /* Adding the piece that contains the origin for r */

        if (etot2 > 0){


#         if ( ON_THE_FLY )

          long e_0  = LOCAL_GP_NR_OF_EVALP(r, 0);

          long s_n  = LSPECGRIDRAD(lspec,LOCAL_GP_NR_OF_EVALP(r, 0),kr);

#         else

          long e_0  = GINDEX(gridp, GP_NR_OF_EVALP(gridp, r, 0));

          long s_n  = LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, r, 0),kr);

#         endif


          S[etot1]      = (Source[s_n] + Source[m_ij]) / 2.0;

          dtau[etot1]   = evalpoint[e_0].dZ * PC * (opacity[s_n] + opacity[m_ij]) / 2.0;
        }



        /* For ray r itself */

        if (etot2 > 1){
        for (long e2=1; e2<etot2; e2++){


#         if ( ON_THE_FLY )

          long e_n  = LOCAL_GP_NR_OF_EVALP(r, e2);

          long s_n  = LSPECGRIDRAD(lspec,LOCAL_GP_NR_OF_EVALP(r, e2),kr);
          long s_np = LSPECGRIDRAD(lspec,LOCAL_GP_NR_OF_EVALP(r, e2-1),kr);

#         else

          long e_n  = GINDEX(gridp, GP_NR_OF_EVALP(gridp, r, e2));

          long s_n  = LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, r, e2),kr);
          long s_np = LSPECGRIDRAD(lspec,GP_NR_OF_EVALP(gridp, r, e2-1),kr);

#         endif


          S[etot1+e2]    = (Source[s_n] + Source[s_np]) / 2.0;

          dtau[etot1+e2] = evalpoint[e_n].dZ * PC * (opacity[s_n] + opacity[s_np]) / 2.0;

        }
        }


        if (SOBOLEV == true){

          /* Sobolev approximation */

          double optical_depth1 = 0.0;
          double optical_depth2 = 0.0;

          double speed_width = sqrt( 8.0*KB*temperature_gas[gridp]/PI/MP + pow(V_TURB,2) );


          for (long e1=0; e1<etot1; e1++){

            optical_depth1 = optical_depth1 + dtau[e1];
          }

          optical_depth1 = CC / frequency[b_ij] / speed_width * optical_depth1;

          if (optical_depth1 < -5.0){

            escape_probability = escape_probability + (1 - exp(5.0)) / (-5.0);
          }

          else if( fabs(optical_depth1) < 1.0E-8){

            escape_probability = escape_probability + 1.0;
          }

          else{

            escape_probability = escape_probability + (1 - exp(-optical_depth1)) / optical_depth1;
          }



          // for (long e2=0; e2<etot2; e2++){
          //
          //   optical_depth2 = optical_depth2 + dtau[etot1+e2];
          // }
          //
          //
          // if (optical_depth2 < -5.0){
          //
          //   escape_probability = escape_probability + (1 - exp(5.0)) / (-5.0);
          // }
          //
          // else if( fabs(optical_depth2) < 1.0E-8){
          //
          //   escape_probability = escape_probability + 1.0;
          // }
          //
          // else{
          //
          //   escape_probability = escape_probability + (1 - exp(-optical_depth2)) / optical_depth2;
          // }


        } /* end of if SOBOLEV */

        // else{
        //
        //   /* Solve the transfer equation with the exact Feautrier solver */
        //
        //   feautrier( evalpoint, gridp, r, ar, S, dtau, P_intensity, Lambda_diagonal );
        //
        //   mean_intensity[m_ij] = mean_intensity[m_ij] + P_intensity[RINDEX(gridp,r)];
        //
        // }



        // printf("(radiative_transfer): number of depth points %ld\n", ndep);

        // printf( "P contribution to mean intensity %lE \n", mean_intensity[m_ij] );


        ndepav = ndepav + ndep;
        nav = nav + 1;

        // if (gridp == 1){
        //   for (int i=0; i<ndep; i++){
        //
        //     std::cout << "opacity and source at << " << i << " are : " << dtau[i] << " " << S[i] << "\n";
        //   }
        // }




        /* Free the allocated memory for temporary variables */

        free(S);
        free(dtau);

      } /* end of if etot1>1 || etot2>1 */


      /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


    } /* end of else (no shortcut) */

  } /* end of r loop over half of the rays */


# if (DIMENSIONS == 1)

  mean_intensity[m_ij] = mean_intensity[m_ij] / NRAYS;

  escape_probability = escape_probability / NRAYS;

# elif (DIMENSIONS == 2)

  mean_intensity[m_ij] = mean_intensity[m_ij] / NRAYS;

  escape_probability = escape_probability / NRAYS;

# elif (DIMENSIONS == 3)

  mean_intensity[m_ij] = mean_intensity[m_ij] / NRAYS;

  escape_probability = escape_probability / NRAYS;

# endif


  /* Add the continuum radiation (due to dust and CMB) */

  double factor          = 2.0*HH*pow(frequency[b_ij],3)/pow(CC,2);

  double rho_grain       = 2.0;

  double ngrain          = 2.0E-12*cell[gridp].density*METALLICITY*100.0/GAS_TO_DUST;

  double emissivity_dust = rho_grain*ngrain*0.01*1.3*frequency[b_ij]/3.0E11;

  double Planck_dust     = 1.0 / ( exp(HH*frequency[b_ij]/KB/temperature_dust[gridp])-1.0 );

  double Planck_CMB      = 1.0 / ( exp(HH*frequency[b_ij]/KB/T_CMB)-1.0 );


  /* NOTE: Continuum radiation is assumed to be local */

  double continuum_mean_intensity = factor * (Planck_CMB + emissivity_dust*Planck_dust);


  if(SOBOLEV == true){

    mean_intensity[m_ij] = (1.0 - escape_probability) * Source[m_ij]
                           + escape_probability * continuum_mean_intensity;


    Lambda_diagonal[m_ij]    = (1.0 - escape_probability);

    mean_intensity_eff[m_ij] = escape_probability * continuum_mean_intensity;

  }
  // else {
  //
  //   mean_intensity[m_ij] = mean_intensity[m_ij] + continuum_mean_intensity;
  // }

  // if (gridp == 1){
  //   printf("%d, %d   %lE  THIS IS  %lE \n", i, j, escape_probability, continuum_mean_intensity);
  // }
  // printf( "(radiative_transfer): mean intensity at gridp %ld for trans %d is %lE \n",
  //         gridp, kr, mean_intensity[m_ij] );


  // printf("(radiative_transfer): average ndep %.2lf \n", (double) ndepav/nav);

}

/*-----------------------------------------------------------------------------------------------*/
