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
#include "lines.hpp"
#include "feautrier.hpp"



#if (ON_THE_FLY)

/* radiative_transfer: calculate mean intensity at grid point "gridp", by solving the transfer   */
/*                     equation along all pairs of a rays and their antipodals                   */
/*-----------------------------------------------------------------------------------------------*/

int radiative_transfer_otf( CELL *cell, EVALPOINT *evalpoint, long *key, long *raytot,
                            long *cum_raytot, double *mean_intensity, double *Lambda_diagonal,
                            double *mean_intensity_eff, double *source, double *opacity,
                            double *frequency, double *temperature_gas, double *temperature_dust,
                            int *irad, int*jrad, long gridp, int lspec, int kr )
{


  long m_ij = LSPECGRIDRAD(lspec,gridp,kr);               /* mean_intensity, S and opacity index */

  int i = irad[LSPECRAD(lspec,kr)];              /* i level index corresponding to transition kr */
  int j = jrad[LSPECRAD(lspec,kr)];              /* j level index corresponding to transition kr */

  long b_ij = LSPECLEVLEV(lspec,i,j);                                         /* frequency index */


  /* For half of the rays (only half is needed since we also consider the antipodals) */

  for (long ray=0; ray<NRAYS/2; ray++){


    /* For all frequencies (Gauss-Hermite quadrature) */

    for (int ny=0; ny<NFREQ; ny++){

      double u_local;
      double v_local;
      double L_local;

      double line_frequency = frequency[b_ij];

      double frequency_shift = line_frequency * evalpoint[gridp].vol / CC;

      double frequency_width = line_frequency / CC
                               * sqrt(2.0*KB*temperature_gas[gridp]/MP + V_TURB*V_TURB);


      double freq = H_4_roots[ny]*frequency_width + frequency_shift;


      intensities( cell, evalpoint, key, raytot, cum_raytot, source, opacity, frequency, freq,
                   temperature_gas, irad, jrad, gridp, ray, lspec, kr, &u_local, &v_local, &L_local );


      mean_intensity[m_ij]  = mean_intensity[m_ij]  + H_4_weights[ny]/frequency_width*u_local;

      Lambda_diagonal[m_ij] = Lambda_diagonal[m_ij] + H_4_weights[ny]/frequency_width*L_local;

    } /* end of ny loop over frequencies */

  } /* end of r loop over half of the rays */


  mean_intensity[m_ij] = mean_intensity[m_ij]; // / NRAYS;


  /* Add the continuum radiation (due to dust and CMB) */

  double factor          = 2.0*HH*pow(frequency[b_ij],3)/pow(CC,2);

  double rho_grain       = 2.0;

  double ngrain          = 2.0E-12*cell[gridp].density*METALLICITY*100.0/GAS_TO_DUST;

  double emissivity_dust = rho_grain*ngrain*0.01*1.3*frequency[b_ij]/3.0E11;

  double Planck_dust     = 1.0 / (exp(HH*frequency[b_ij]/KB/temperature_dust[gridp]) - 1.0);

  double Planck_CMB      = 1.0 / (exp(HH*frequency[b_ij]/KB/T_CMB) - 1.0);


  /* NOTE: Continuum radiation is assumed to be local */

  double continuum_mean_intensity = factor * (Planck_CMB + emissivity_dust*Planck_dust);


  mean_intensity[m_ij] = mean_intensity[m_ij]; // + continuum_mean_intensity;


  if ( ACCELERATION_APPROX_LAMBDA ){

    mean_intensity_eff[m_ij] = mean_intensity[m_ij] - Lambda_diagonal[m_ij]*source[m_ij];
  }

  else {

    Lambda_diagonal[m_ij] = 0.0;

    mean_intensity_eff[m_ij] = mean_intensity[m_ij];
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* intensity: calculate the intensity along a certain ray through a certain point                */
/*-----------------------------------------------------------------------------------------------*/

int intensities( CELL *cell, EVALPOINT *evalpoint, long *key, long *raytot,
                 long *cum_raytot, double *source, double *opacity, double *frequency, double freq,
                 double *temperature_gas,  int *irad, int*jrad, long gridp, long r, int lspec,
                 int kr, double *u_local, double *v_local, double *L_local )
{


  /* Get the antipodal ray for r */

  long ar = antipod[r];                                           /* index of antipodal ray to r */

  long m_ij = LSPECGRIDRAD(lspec,gridp,kr);               /* mean_intensity, S and opacity index */

  int i = irad[LSPECRAD(lspec,kr)];              /* i level index corresponding to transition kr */
  int j = jrad[LSPECRAD(lspec,kr)];              /* j level index corresponding to transition kr */

  long b_ij = LSPECLEVLEV(lspec,i,j);                                         /* frequency index */




  /* Fill the source function and the optical depth increment on the subgrid                     */
  /*_____________________________________________________________________________________________*/


  long etot1 = raytot[ar];                     /* total number of evaluation points along ray ar */
  long etot2 = raytot[r];                       /* total number of evaluation points along ray r */


  if ( etot1>0 && etot2>0 ){

    long ndep = etot1 + etot2;             /* nr. of depth points along a pair of antipodal rays */


    double *S;                                                 /* source function along this ray */
    S = (double*) malloc( ndep*sizeof(double) );

    double *dtau;                                                /* optical depth along this ray */
    dtau = (double*) malloc( ndep*sizeof(double) );

    double *u;
    u = (double*) malloc( ndep*sizeof(double) );

    double *L_diag_approx;
    L_diag_approx = (double*) malloc( ndep*sizeof(double) );


    /* For the antipodal ray to ray r */

    if (etot1 > 1){
    for (long e1=1; e1<etot1; e1++){

      long e_n   = LOCAL_GP_NR_OF_EVALP(ar, etot1-e1);
      long e_np  = LOCAL_GP_NR_OF_EVALP(ar, etot1-e1-1);

      double phi_n  = line_profile(evalpoint, temperature_gas, freq, frequency[b_ij], e_n);
      double phi_np = line_profile(evalpoint, temperature_gas, freq, frequency[b_ij], e_np);

      long s_n  = LSPECGRIDRAD(lspec,e_n,kr);
      long s_np = LSPECGRIDRAD(lspec,e_np,kr);


      S[e1-1]    = (source[s_n] + source[s_np]) / 2.0;

      dtau[e1-1] = evalpoint[e_n].dZ * PC * (opacity[s_n]*phi_n + opacity[s_np]*phi_np) / 2.0;
    }
    }


    /* Adding the piece that contains the origin for ar */

    {
      long e_a0 = LOCAL_GP_NR_OF_EVALP(ar, 0);

      double phi_a0 = line_profile(evalpoint, temperature_gas, freq, frequency[b_ij], e_a0);
      double phi    = line_profile(evalpoint, temperature_gas, freq, frequency[b_ij], gridp);

      long s_a0 = LSPECGRIDRAD(lspec,e_a0,kr);


      S[etot1-1]    = (source[s_a0] + source[m_ij]) / 2.0;

      dtau[etot1-1] = evalpoint[e_a0].dZ * PC * (opacity[s_a0]*phi_a0 + opacity[m_ij]*phi) / 2.0;
    }


    /* Adding the piece that contains the origin for r */

    {
      long e_0  = LOCAL_GP_NR_OF_EVALP(r, 0);

      double phi_0 = line_profile(evalpoint, temperature_gas, freq, frequency[b_ij], e_0);
      double phi   = line_profile(evalpoint, temperature_gas, freq, frequency[b_ij], gridp);

      long s_0  = LSPECGRIDRAD(lspec,e_0,kr);


      S[etot1]    = (source[s_0] + source[m_ij]) / 2.0;

      dtau[etot1] = evalpoint[e_0].dZ * PC * (opacity[s_0]*phi_0 + opacity[m_ij]*phi) / 2.0;
    }


    /* For ray r itself */

    if (etot2 > 1){
    for (long e2=1; e2<etot2; e2++){

      long e_n  = LOCAL_GP_NR_OF_EVALP(r, e2);
      long e_np = LOCAL_GP_NR_OF_EVALP(r, e2-1);

      double phi_n  = line_profile(evalpoint, temperature_gas, freq, frequency[b_ij], e_n);
      double phi_np = line_profile(evalpoint, temperature_gas, freq, frequency[b_ij], e_np);

      long s_n  = LSPECGRIDRAD(lspec,e_n,kr);
      long s_np = LSPECGRIDRAD(lspec,e_np,kr);


      S[etot1+e2]    = (source[s_n] + source[s_np]) / 2.0;

      dtau[etot1+e2] = evalpoint[e_n].dZ * PC * (opacity[s_n]*phi_n + opacity[s_np]*phi_np) / 2.0;
    }
    }


    /*___________________________________________________________________________________________*/




    /* Avoid too small optical depth increments                                                  */
    /*___________________________________________________________________________________________*/

    for (long dep=0; dep<ndep; dep++){

      if (dtau[dep]<1.0E-99){

        dtau[dep] = 1.0E-99;
      }

    }

    /*___________________________________________________________________________________________*/




    /* Solve the transfer equation with the exact Feautrier solver (on the subgrid)              */
    /*___________________________________________________________________________________________*/


    feautrier(evalpoint, key, raytot, cum_raytot, gridp, r, S, dtau, u, L_diag_approx);


    /*___________________________________________________________________________________________*/




    /* Map the results back from the subgrid to the grid                                         */
    /*___________________________________________________________________________________________*/


    *u_local = (u[etot1] + u[etot1-1]) / 2.0;

    *v_local = 2.0 * (u[etot1] - u[etot1-1]) / (dtau[etot1] + dtau[etot1-1]) ;

    *L_local = (L_diag_approx[etot1] + L_diag_approx[etot1-1]) / 2.0;


    /*___________________________________________________________________________________________*/


    /* Free the allocated memory for temporary variables */

    free(S);
    free(dtau);
    free(u);
    free(L_diag_approx);

  } /* end of if etot1>1 && etot2>1 */


  return(0);

}

#endif
