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

#include "radiative_transfer_otf.hpp"
#include "feautrier.hpp"



#ifdef ON_THE_FLY

/* radiative_transfer: calculate mean intensity at grid point "gridp", by solving the transfer   */
/*                     equation along all pairs of a rays and their antipodals                   */
/*-----------------------------------------------------------------------------------------------*/

int radiative_transfer_otf( GRIDPOINT *gridpoint, EVALPOINT *evalpoint, long *key, long *raytot,
                            long *cum_raytot, double *mean_intensity, double *Lambda_diagonal,
                            double *mean_intensity_eff, double *source, double *opacity,
                            double *frequency, double *temperature_gas, double *temperature_dust,
                            int *irad, int*jrad, long gridp, int lspec, int kr )
{


  long m_ij = LSPECGRIDRAD(lspec,gridp,kr);               /* mean_intensity, S and opacity index */

  int i = irad[LSPECRAD(lspec,kr)];              /* i level index corresponding to transition kr */
  int j = jrad[LSPECRAD(lspec,kr)];              /* j level index corresponding to transition kr */

  long b_ij = LSPECLEVLEV(lspec,i,j);                                         /* frequency index */

  double escape_probability = 0.0;          /* escape probability from the Sobolev approximation */


  /* For half of the rays (only half is needed since we also consider the antipodals) */

  for (long r=0; r<NRAYS/2; r++){


    /* Get the antipodal ray for r */

    long ar = antipod[r];                                         /* index of antipodal ray to r */


    /*   DO THE RADIATIVE TRANSFER
    /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


    /* Fill the source function and the optical depth increment along ray r */

    long etot1 = raytot[ar];                 /* total number of evaluation points along ray ar */
    long etot2 = raytot[r];                   /* total number of evaluation points along ray r */


    if ( etot1>0 || etot2>0 ){

      long ndep = etot1 + etot2;         /* nr. of depth points along a pair of antipodal rays */

      double *S;                                               /* source function along this ray */
      S = (double*) malloc( ndep*sizeof(double) );

      double *dtau;                                              /* optical depth along this ray */
      dtau = (double*) malloc( ndep*sizeof(double) );


      /* For the antipodal ray to ray r */

      if (etot1 > 1){
      for (long e1=1; e1<etot1; e1++){

        long e_n   = LOCAL_GP_NR_OF_EVALP(ar, etot1-e1);
        long e_np  = LOCAL_GP_NR_OF_EVALP(ar, etot1-e1-1);

        long s_n  = LSPECGRIDRAD(lspec,e_n,kr);
        long s_np = LSPECGRIDRAD(lspec,e_np,kr);


        S[e1-1]    = (source[s_n] + source[s_np]) / 2.0;

        dtau[e1-1] = evalpoint[e_n].dZ * PC * (opacity[s_n] + opacity[s_np]) / 2.0;
      }
      }


      /* Adding the piece that contains the origin for ar */

      if (etot1 > 0){

        long e_a0 = LOCAL_GP_NR_OF_EVALP(ar, 0);

        long s_an = LSPECGRIDRAD(lspec,e_a0,kr);


        S[etot1-1]    = (source[s_an] + source[m_ij]) / 2.0;

        dtau[etot1-1] = evalpoint[e_a0].dZ * PC * (opacity[s_an] + opacity[m_ij]) / 2.0;
      }


      /* Adding the piece that contains the origin for r */

      if (etot2 > 0){

        long e_0  = LOCAL_GP_NR_OF_EVALP(r, 0);

        long s_n  = LSPECGRIDRAD(lspec,e_0,kr);


        S[etot1]      = (source[s_n] + source[m_ij]) / 2.0;

        dtau[etot1]   = evalpoint[e_0].dZ * PC * (opacity[s_n] + opacity[m_ij]) / 2.0;
      }



      /* For ray r itself */

      if (etot2 > 1){
      for (long e2=1; e2<etot2; e2++){

        long e_n  = LOCAL_GP_NR_OF_EVALP(r, e2);
        long e_np = LOCAL_GP_NR_OF_EVALP(r, e2-1);

        long s_n  = LSPECGRIDRAD(lspec,e_n,kr);
        long s_np = LSPECGRIDRAD(lspec,e_np,kr);


        S[etot1+e2]    = (source[s_n] + source[s_np]) / 2.0;

        dtau[etot1+e2] = evalpoint[e_n].dZ * PC * (opacity[s_n] + opacity[s_np]) / 2.0;
      }
      }


      /* Solve the transfer equation with the exact Feautrier solver */

      // feautrier( evalpoint, gridp, r, ar, S, dtau, P_intensity, Lambda_diagonal );
      //
      // mean_intensity[m_ij] = mean_intensity[m_ij] + P_intensity[RINDEX(gridp,r)];


      /* Free the allocated memory for temporary variables */

      free(S);
      free(dtau);

    } /* end of if etot1>1 || etot2>1 */


    /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


  } /* end of r loop over half of the rays */


  mean_intensity[m_ij] = mean_intensity[m_ij]; // / NRAYS;

  escape_probability = escape_probability; // / NRAYS;


  /* Add the continuum radiation (due to dust and CMB) */

  double factor          = 2.0*HH*pow(frequency[b_ij],3)/pow(CC,2);

  double rho_grain       = 2.0;

  double ngrain          = 2.0E-12*gridpoint[gridp].density*METALLICITY*100.0/GAS_TO_DUST;

  double emissivity_dust = rho_grain*ngrain*0.01*1.3*frequency[b_ij]/3.0E11;

  double Planck_dust     = 1.0 / ( exp(HH*frequency[b_ij]/KB/temperature_dust[gridp])-1.0 );

  double Planck_CMB      = 1.0 / ( exp(HH*frequency[b_ij]/KB/T_CMB)-1.0 );


  /* NOTE: Continuum radiation is assumed to be local */

  double continuum_mean_intensity = factor * (Planck_CMB + emissivity_dust*Planck_dust);


  mean_intensity[m_ij] = mean_intensity[m_ij] + continuum_mean_intensity;


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/

#endif



//
// /* radiative_transfer: calculate mean intensity at grid point "gridp", by solving the transfer   */
// /*                     equation along all pairs of a rays and their antipodals                   */
// /*-----------------------------------------------------------------------------------------------*/
//
// double u_intensity( GRIDPOINT *gridpoint, EVALPOINT *evalpoint, long *key, long *raytot,
//                     long *cum_raytot, double *source, double *opacity, int *irad, int*jrad,
//                     long gridp, int lspec, int kr, double )
// {
//
//
//   long m_ij = LSPECGRIDRAD(lspec,gridp,kr);               /* mean_intensity, S and opacity index */
//
//   int i = irad[LSPECRAD(lspec,kr)];              /* i level index corresponding to transition kr */
//   int j = jrad[LSPECRAD(lspec,kr)];              /* j level index corresponding to transition kr */
//
//   long b_ij = LSPECLEVLEV(lspec,i,j);                                         /* frequency index */
//
//   double escape_probability = 0.0;          /* escape probability from the Sobolev approximation */
//
//
//
//   /* Get the antipodal ray for r */
//
//   long ar = antipod[r];                                           /* index of antipodal ray to r */
//
//
//   /* Fill the source function and the optical depth increment along ray r */
//
//   long etot1 = raytot[ar];                     /* total number of evaluation points along ray ar */
//   long etot2 = raytot[r];                       /* total number of evaluation points along ray r */
//
//
//   if ( etot1>0 || etot2>0 ){
//
//     long ndep = etot1 + etot2;             /* nr. of depth points along a pair of antipodal rays */
//
//     double *S;                                                 /* source function along this ray */
//     S = (double*) malloc( ndep*sizeof(double) );
//
//     double *dtau;                                                /* optical depth along this ray */
//     dtau = (double*) malloc( ndep*sizeof(double) );
//
//
//     /* For the antipodal ray to ray r */
//
//     if (etot1 > 1){
//     for (long e1=1; e1<etot1; e1++){
//
//       long e_n   = LOCAL_GP_NR_OF_EVALP(ar, etot1-e1);
//       long e_np  = LOCAL_GP_NR_OF_EVALP(ar, etot1-e1-1);
//
//       long s_n  = LSPECGRIDRAD(lspec,e_n,kr);
//       long s_np = LSPECGRIDRAD(lspec,e_np,kr);
//
//
//       S[e1-1]    = (source[s_n] + source[s_np]) / 2.0;
//
//       dtau[e1-1] = evalpoint[e_n].dZ * PC * (opacity[s_n] + opacity[s_np]) / 2.0;
//     }
//     }
//
//
//     /* Adding the piece that contains the origin for ar */
//
//     if (etot1 > 0){
//
//       long e_a0 = LOCAL_GP_NR_OF_EVALP(ar, 0);
//
//       long s_an = LSPECGRIDRAD(lspec,e_a0,kr);
//
//
//       S[etot1-1]    = (source[s_an] + source[m_ij]) / 2.0;
//
//       dtau[etot1-1] = evalpoint[e_a0].dZ * PC * (opacity[s_an] + opacity[m_ij]) / 2.0;
//     }
//
//
//     /* Adding the piece that contains the origin for r */
//
//     if (etot2 > 0){
//
//       long e_0  = LOCAL_GP_NR_OF_EVALP(r, 0);
//
//       long s_n  = LSPECGRIDRAD(lspec,e_0,kr);
//
//
//       S[etot1]      = (source[s_n] + source[m_ij]) / 2.0;
//
//       dtau[etot1]   = evalpoint[e_0].dZ * PC * (opacity[s_n] + opacity[m_ij]) / 2.0;
//     }
//
//
//
//     /* For ray r itself */
//
//     if (etot2 > 1){
//     for (long e2=1; e2<etot2; e2++){
//
//       long e_n  = LOCAL_GP_NR_OF_EVALP(r, e2);
//       long e_np = LOCAL_GP_NR_OF_EVALP(r, e2-1);
//
//       long s_n  = LSPECGRIDRAD(lspec,e_n,kr);
//       long s_np = LSPECGRIDRAD(lspec,e_np,kr);
//
//
//       S[etot1+e2]    = (source[s_n] + source[s_np]) / 2.0;
//
//       dtau[etot1+e2] = evalpoint[e_n].dZ * PC * (opacity[s_n] + opacity[s_np]) / 2.0;
//     }
//     }
//
//
//     if ( SOBOLEV ){
//
//
//       /* Sobolev approximation (large-velocity-gradient) */
//
//
//
//     } /* end of if SOBOLEV */
//
//     else{
//
//
//       /* Solve the transfer equation with the Feautrier solver */
//
//       feautrier( evalpoint, gridp, r, ar, S, dtau, P_intensity, Lambda_diagonal );
//
//       mean_intensity[m_ij] = mean_intensity[m_ij] + P_intensity[RINDEX(gridp,r)];
//
//     }
//
//
//     /* Free the allocated memory for temporary variables */
//
//     free(S);
//     free(dtau);
//
//   } /* end of if etot1>1 || etot2>1 */
//
//
// }
//
// /*-----------------------------------------------------------------------------------------------*/
