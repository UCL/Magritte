/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* cooling: calculate the cooling                                                                */
/*                                                                                               */
/* (based on escape_probability in 3D-PDR)                                                       */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <string>
#include <iostream>
using namespace std;

#include "declarations.hpp"
#include "cooling.hpp"



/* cooling: calculate the total cooling                                                          */
/*-----------------------------------------------------------------------------------------------*/


double cooling( long gridp, int *irad, int *jrad, double *A_coeff, double *B_coeff,
                double *frequency, double *pop, double *mean_intensity )
{


  double cooling_total = 0.0;                                                   /* total cooling */
  double cooling_radiative = 0.0;                                           /* radiative cooling */


  /* For all line producing species */

  for (int lspec=0; lspec<NLSPEC; lspec++){


    /* For all transitions with (i>j) */

    for (int kr=0; kr<nrad[lspec]; kr++){


      int i     = irad[LSPECRAD(lspec,kr)];      /* i level index corresponding to transition kr */
      int j     = jrad[LSPECRAD(lspec,kr)];      /* j level index corresponding to transition kr */

      long b_ij = LSPECLEVLEV(lspec,i,j);                /* A_coeff, B_coeff and frequency index */
      long b_ji = LSPECLEVLEV(lspec,j,i);                /* A_coeff, B_coeff and frequency index */

      long p_i  = LSPECGRIDLEV(lspec,gridp,i);                          /* population at level i */
      long p_j  = LSPECGRIDLEV(lspec,gridp,j);                          /* population at level j */

      long m_ij = LSPECGRIDRAD(lspec,gridp,kr);                          /* mean intensity index */


      /* Calculate the source function */

      double Source = 0.0;

      if (pop[p_j] > POP_LOWER_LIMIT || pop[p_i] > POP_LOWER_LIMIT){

        Source = (A_coeff[b_ij] * pop[p_i])/(pop[p_j]*B_coeff[b_ji] - pop[p_i]*B_coeff[b_ij]);
      }


      /* Calculate the radiative cooling */

      if (Source != 0.0){

        cooling_radiative = cooling_radiative
                            + HH*frequency[b_ij] * pop[p_i] * (1.0 - mean_intensity[m_ij]/Source);
      }

      else {

        cooling_radiative = cooling_radiative + 0.0;
      }

    } /* end of kr loop over transitions */

  } /* end of lspec loop over line producing species */


  cooling_total = cooling_radiative;


  return cooling_total;

}

/*-----------------------------------------------------------------------------------------------*/





/* 3D-PDR piece of code */
/*----------------------*/

// double factor          = 2.0*HH*pow(nu_ij,3)/pow(CC,2);

// double rho_grain       = 2.0;

// double ngrain          = 2.0E-12 * gridpoint[gridp].density * metallicity * 100/gas_to_dust;

// double emissivity_dust = rho_grain * ngrain * 0.01 * 1.3 * nu_ij / 3.0E11;

// double Planck_dust     = 1.0 / ( exp(HH*nu_ij/KB/temperature_dust[gridp])-1.0 );

// double Planck_CMB      = 1.0 / ( exp(HH*nu_ij/KB/T_CMB)-1.0 );

// double Planck          = factor * (Planck_CMB + emissivity_dust*Planck_dust);
