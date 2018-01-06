// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#if (!CELL_BASED)

#include "sobolev.hpp"



// sobolev: calculate mean intensity using LVG approximation and escape probabilities
// ----------------------------------------------------------------------------------

int sobolev( long ncells, CELL *cell, EVALPOINT *evalpoint, long *key, long *raytot, long *cum_raytot,
             double *mean_intensity, double *Lambda_diagonal, double *mean_intensity_eff, double *source,
             double *opacity, double *frequency, int *irad, int*jrad, long gridp, int lspec, int kr )
{


  long m_ij = LSPECGRIDRAD(lspec,gridp,kr);               /* mean_intensity, S and opacity index */

  int i = irad[LSPECRAD(lspec,kr)];              /* i level index corresponding to transition kr */
  int j = jrad[LSPECRAD(lspec,kr)];              /* j level index corresponding to transition kr */

  long b_ij = LSPECLEVLEV(lspec,i,j);                                         /* frequency index */

  double escape_probability = 0.0;          /* escape probability from the Sobolev approximation */


  /* For half of the rays (only half is needed since we also consider the antipodals) */

  for (long r=0; r<NRAYS/2; r++)
  {

    /* Get the antipodal ray for r */

    long ar = antipod[r];                                         /* index of antipodal ray to r */


    /*   DO THE RADIATIVE TRANSFER
    /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


    /* Fill the source function and the optical depth increment along ray r */

    long etot1 = raytot[ar];                   /* total number of evaluation points along ray ar */
    long etot2 = raytot[r];                     /* total number of evaluation points along ray r */


    if ( etot1>0 || etot2>0 )
    {
      long ndep = etot1 + etot2;           /* nr. of depth points along a pair of antipodal rays */

      double *dtau;                                              /* optical depth along this ray */
      dtau = (double*) malloc( ndep*sizeof(double) );


      /* For the antipodal ray to ray r */

      if (etot1 > 1)
      {
        for (long e1=1; e1<etot1; e1++)
        {
          long e_n   = LOCAL_GP_NR_OF_EVALP(ar, etot1-e1);
          long e_np  = LOCAL_GP_NR_OF_EVALP(ar, etot1-e1-1);

          long s_n  = LSPECGRIDRAD(lspec,e_n,kr);
          long s_np = LSPECGRIDRAD(lspec,e_np,kr);

          dtau[e1-1] = evalpoint[e_n].dZ * PC * (opacity[s_n] + opacity[s_np]) / 2.0;
        }
      }


      /* Adding the piece that contains the origin for ar */

      if (etot1 > 0)
      {
        long e_a0 = LOCAL_GP_NR_OF_EVALP(ar, 0);

        long s_an = LSPECGRIDRAD(lspec,e_a0,kr);

        dtau[etot1-1] = evalpoint[e_a0].dZ * PC * (opacity[s_an] + opacity[m_ij]) / 2.0;
      }


      /* Adding the piece that contains the origin for r */

      if (etot2 > 0)
      {
        long e_0  = LOCAL_GP_NR_OF_EVALP(r, 0);

        long s_n  = LSPECGRIDRAD(lspec,e_0,kr);

        dtau[etot1]   = evalpoint[e_0].dZ * PC * (opacity[s_n] + opacity[m_ij]) / 2.0;
      }



      /* For ray r itself */

      if (etot2 > 1)
      {
        for (long e2=1; e2<etot2; e2++)
        {
          long e_n  = LOCAL_GP_NR_OF_EVALP(r, e2);
          long e_np = LOCAL_GP_NR_OF_EVALP(r, e2-1);

          long s_n  = LSPECGRIDRAD(lspec,e_n,kr);
          long s_np = LSPECGRIDRAD(lspec,e_np,kr);

          dtau[etot1+e2] = evalpoint[e_n].dZ * PC * (opacity[s_n] + opacity[s_np]) / 2.0;
        }
      }


      double optical_depth1 = 0.0;
      double optical_depth2 = 0.0;

      double speed_width = sqrt(8.0*KB*cell[gridp].temperature.gas/PI/MP + pow(V_TURB, 2));


      for (long e1=0; e1<etot1; e1++)
      {
        optical_depth1 = optical_depth1 + dtau[e1];
      }

      optical_depth1 = CC / frequency[b_ij] / speed_width * optical_depth1;

      if (optical_depth1 < -5.0)
      {
        escape_probability = escape_probability + (1 - exp(5.0)) / (-5.0);
      }

      else if (fabs(optical_depth1) < 1.0E-8)
      {
        escape_probability = escape_probability + 1.0;
      }

      else
      {
        escape_probability = escape_probability + (1.0 - exp(-optical_depth1)) / optical_depth1;
      }

      optical_depth2 = CC / frequency[b_ij] / speed_width * optical_depth2;

      for (long e2=0; e2<etot2; e2++)
      {
        optical_depth2 = optical_depth2 + dtau[etot1+e2];
      }

      if (optical_depth2 < -5.0)
      {
        escape_probability = escape_probability + (1.0 - exp(5.0)) / (-5.0);
      }

      else if( fabs(optical_depth2) < 1.0E-8)
      {
        escape_probability = escape_probability + 1.0;
      }

      else
      {
        escape_probability = escape_probability + (1.0 - exp(-optical_depth2)) / optical_depth2;
      }


      /* Free the allocated memory for temporary variable */

      free(dtau);

    } /* end of if etot1>1 || etot2>1 */


    /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


  } /* end of r loop over half of the rays */


  escape_probability = escape_probability / NRAYS;

  // printf("esc prob %lE\n", escape_probability);


  // ADD CONTINUUM RADIATION (due to dust and CMB)
  // _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


  // Continuum radiation is assumed to be local

  double factor          = 2.0*HH*pow(frequency[b_ij],3)/pow(CC,2);

  double rho_grain       = 2.0;

  double ngrain          = 2.0E-12*cell[gridp].density*METALLICITY*100.0/GAS_TO_DUST;

  double emissivity_dust = rho_grain*ngrain*0.01*1.3*frequency[b_ij]/3.0E11;

  double Planck_dust     = 1.0 / (exp(HH*frequency[b_ij]/KB/cell[gridp].temperature.dust) - 1.0);

  double Planck_CMB      = 1.0 / (exp(HH*frequency[b_ij]/KB/T_CMB) - 1.0);


  double continuum_mean_intensity = factor * (Planck_CMB + emissivity_dust*Planck_dust);


  mean_intensity[m_ij] = (1.0 - escape_probability) * source[m_ij]
                         + escape_probability * continuum_mean_intensity;


  if (ACCELERATION_APPROX_LAMBDA)
  {
    Lambda_diagonal[m_ij]    = (1.0 - escape_probability);

    mean_intensity_eff[m_ij] = escape_probability * continuum_mean_intensity;
  }

  else
  {
    Lambda_diagonal[m_ij]    = 0.0;

    mean_intensity_eff[m_ij] = mean_intensity[m_ij];
  }


  return(0);
}


#endif // if not CELL_BASED
