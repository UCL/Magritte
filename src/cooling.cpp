/* Frederik De Ceuster - University College London & KU Leuven                                   */
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
                double *frequency, double *weight, double *pop, double *mean_intensity )
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

      double factor = 2.0 * HH * pow(frequency[b_ij], 3) / pow(CC, 2);

      double tpop   = pop[p_j]*weight[LSPECLEV(lspec,i)]/pop[p_i]/weight[LSPECLEV(lspec,j)] - 1.0;


      if ( fabs(tpop)<1.0E-50 ){

        Source = HH * frequency[b_ij] * pop[p_i] * A_coeff[b_ij] / 4.0 / PI;
      }

      else if ( pop[p_i] > 0.0 ) {

        Source = factor / tpop;
      }


      /* Calculate the radiative line cooling */

      if (Source != 0.0){

        cooling_radiative = cooling_radiative
                            + HH*frequency[b_ij] * A_coeff[b_ij] * pop[p_i]
                              * (1.0 - mean_intensity[m_ij]/Source);
      }


    } /* end of kr loop over transitions */

  } /* end of lspec loop over line producing species */


  cooling_total = cooling_radiative;


  return cooling_total;

}

/*-----------------------------------------------------------------------------------------------*/
