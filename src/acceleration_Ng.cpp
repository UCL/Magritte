/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* acceleration_Ng: Perform a Ng accelerated iteration for the level populations                 */
/*                                                                                               */
/* (based on accelerate in SMMOL)                                                                */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "acceleration_Ng.hpp"



/* acceleration_Ng: perform a Ng accelerated iteration for the level populations                 */
/*-----------------------------------------------------------------------------------------------*/

int acceleration_Ng( int lspec, double *prev3_pop, double *prev2_pop, double *prev1_pop,
                     double *pop )
{


  /* All variable names are based on the lecture notes on radiative transfer by C.P. Dullemond */

  double Q1[NGRID*nlev[lspec]];
  double Q2[NGRID*nlev[lspec]];
  double Q3[NGRID*nlev[lspec]];

  double Wt[NGRID*nlev[lspec]];                                  /* weights of the inner product */


  for (long gridp=0; gridp<NGRID; gridp++){

    for (int i=0; i<nlev[lspec]; i++){

      long p_i = LSPECGRIDLEV(lspec,gridp,i);
      long w_i = LINDEX(gridp,i);

      Q1[w_i] = pop[p_i] - 2.0*prev1_pop[p_i] + prev2_pop[p_i];
      Q2[w_i] = pop[p_i] - prev1_pop[p_i] - prev2_pop[p_i] + prev3_pop[p_i];
      Q3[w_i] = pop[p_i] - prev1_pop[p_i];

      if (pop[p_i] > 0.0){

        Wt[w_i] = 1.0 / fabs(pop[p_i]);
      }

      else {

        Wt[w_i] = 1.0;
      }

    } /* end of i loop over levels */

  } /* end of gridp loop over grid points */


  double A1 = 0.0;
  double A2 = 0.0;

  double B1 = 0.0;
  double B2 = 0.0;

  double C1 = 0.0;
  double C2 = 0.0;


  for (long gi=0; gi<NGRID*nlev[lspec]; gi++){

    A1      = A1 + Wt[gi]*Q1[gi]*Q1[gi];
    A2 = B1 = A2 + Wt[gi]*Q1[gi]*Q2[gi];
    B2      = B2 + Wt[gi]*Q2[gi]*Q2[gi];
    C1      = C1 + Wt[gi]*Q1[gi]*Q3[gi];
    C2      = C2 + Wt[gi]*Q2[gi]*Q3[gi];
  }


  double denominator = A1*B2 - A2*B1;

  if (denominator == 0.0){

    return(0);
  }

  else {

    double a = (C1*B2 - C2*B1) / denominator;
    double b = (C2*A1 - C1*A2) / denominator;


    for (long gridp=0; gridp<NGRID; gridp++){

      for (int i=0; i<nlev[lspec]; i++){

        long p_i = LSPECGRIDLEV(lspec,gridp,i);
        long w_i = LINDEX(gridp,i);

        double pop_tmp = pop[p_i];

        pop[p_i] = (1.0 - a - b)*pop[p_i] + a*prev1_pop[p_i] + b*prev2_pop[p_i];

        prev3_pop[p_i] = prev2_pop[p_i];
        prev2_pop[p_i] = prev1_pop[p_i];
        prev1_pop[p_i] = pop_tmp;

      } /* end of i loop over levels */

    } /* end of gridp loop over grid points */

  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* store_populations: update the previous populations                                            */
/*-----------------------------------------------------------------------------------------------*/

int store_populations( int lspec, double *prev3_pop, double *prev2_pop, double *prev1_pop,
                       double *pop )
{


  for (long gridp=0; gridp<NGRID; gridp++){

    for (int i=0; i<nlev[lspec]; i++){

      long p_i = LSPECGRIDLEV(lspec,gridp,i);

      prev3_pop[p_i] = prev2_pop[p_i];
      prev2_pop[p_i] = prev1_pop[p_i];
      prev1_pop[p_i] = pop[p_i];

    } /* end of i loop over levels */

  } /* end of gridp loop over grid points */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





// /* acceleration_temperature_Ng: perform a Ng accelerated iteration for the gas temperature       */
// /*-----------------------------------------------------------------------------------------------*/
//
// int acceleration_temperature_Ng( double *prev3_temperature_gas, double *prev2_temperature_gas,
//                                  double *prev1_temperature_gas, double *temperature_gas )
// {
//
//
//   /* All variable names are based on the lecture notes on radiative transfer by C.P. Dullemond */
//
//   double Q1[NGRID];
//   double Q2[NGRID];
//   double Q3[NGRID];
//
//   double Wt[NGRID];                                              /* weights of the inner product */
//
//
//   for (long gridp=0; gridp<NGRID; gridp++){
//
//     Q1[gridp] = temperature_gas[gridp] - 2.0*prev1_temperature_gas[gridp]
//                 + prev2_temperature_gas[gridp];
//
//     Q2[gridp] = temperature_gas[gridp] - prev1_temperature_gas[gridp]
//                 - prev2_temperature_gas[gridp] + prev3_temperature_gas[gridp];
//
//     Q3[gridp] = temperature_gas[gridp] - prev1_temperature_gas[gridp];
//
//
//     if (temperature_gas[gridp] > 0.0){
//
//       Wt[gridp] = 1.0 / fabs(temperature_gas[gridp]);
//     }
//
//     else {
//
//       Wt[gridp] = 1.0;
//     }
//
//
//
//   } /* end of gridp loop over grid points */
//
//
//   double A1 = 0.0;
//   double A2 = 0.0;
//
//   double B1 = 0.0;
//   double B2 = 0.0;
//
//   double C1 = 0.0;
//   double C2 = 0.0;
//
//
//   for (long gridp=0; gridp<NGRID; gridp++){
//
//     A1      = A1 + Wt[gridp]*Q1[gridp]*Q1[gridp];
//     A2 = B1 = A2 + Wt[gridp]*Q1[gridp]*Q2[gridp];
//     B2      = B2 + Wt[gridp]*Q2[gridp]*Q2[gridp];
//     C1      = C1 + Wt[gridp]*Q1[gridp]*Q3[gridp];
//     C2      = C2 + Wt[gridp]*Q2[gridp]*Q3[gridp];
//   }
//
//
//   double denominator = A1*B2 - A2*B1;
//
//   if (denominator == 0.0){
//
//     return(0);
//   }
//
//   else {
//
//     double a = (C1*B2 - C2*B1) / denominator;
//     double b = (C2*A1 - C1*A2) / denominator;
//
//
//     for (long gridp=0; gridp<NGRID; gridp++){
//
//       double temperature_gas_tmp = (1.0 - a - b)*temperature_gas[gridp]
//                                    + a*prev1_temperature_gas[gridp]
//                                    + b*prev2_temperature_gas[gridp];
//
//       prev3_temperature_gas[gridp] = prev2_temperature_gas[gridp];
//       prev2_temperature_gas[gridp] = prev1_temperature_gas[gridp];
//       prev1_temperature_gas[gridp] = temperature_gas[gridp];
//
//       if (temperature_gas_tmp < T_CMB){
//
//         temperature_gas[gridp] = T_CMB;
//       }
//
//       else if (temperature_gas_tmp > 30000.0){
//
//         temperature_gas[gridp] = 30000.0;
//       }
//
//       else {
//
//         temperature_gas[gridp] = temperature_gas_tmp;
//       }
//
//     } /* end of gridp loop over grid points */
//
//   }
//
//
//   return(0);
//
// }
//
// /*-----------------------------------------------------------------------------------------------*/
//
//
//
//
//
// /* store_temperatures: update the previous temperatures                                          */
// /*-----------------------------------------------------------------------------------------------*/
//
// int store_temperatures( double *prev3_temperature_gas, double *prev2_temperature_gas,
//                         double *prev1_temperature_gas, double *temperature_gas )
// {
//
//
//   for (long gridp=0; gridp<NGRID; gridp++){
//
//     prev3_temperature_gas[gridp] = prev2_temperature_gas[gridp];
//     prev2_temperature_gas[gridp] = prev1_temperature_gas[gridp];
//     prev1_temperature_gas[gridp] = temperature_gas[gridp];
//
//   }
//
//
//   return(0);
//
// }
//
// /*-----------------------------------------------------------------------------------------------*/
