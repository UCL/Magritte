/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* update_temperatures_gas.c: Update the gas temperatures after a thermal balance iteration      */
/*                                                                                               */
/* (based on 3DPDR in 3DPDR)                                                                     */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "update_temperature_gas.hpp"



#define EPSILON 3.0E-8                             /* lower bound on the precision we can obtain */
#define PREC    5.0E-3                                   /* precision we need on the temperature */



/* update_temperature_gas: update the gas temperature after a thermal balance iteration          */
/*-----------------------------------------------------------------------------------------------*/

int update_temperature_gas( double *thermal_ratio, double *temperature_gas,
                            double *prev_temperature_gas, double *temperature_a,
                            double *temperature_b, double *thermal_ratio_a,
                            double *thermal_ratio_b )
{


  /* For all grid points */

# pragma omp parallel                                                                             \
  shared( thermal_ratio, temperature_gas, prev_temperature_gas, temperature_a, temperature_b,     \
          thermal_ratio_a, thermal_ratio_b  )                                                     \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NGRID)/num_threads;
  long stop  = ((thread_num+1)*NGRID)/num_threads;       /* Note the brackets are important here */


  for (long gridp=start; gridp<stop; gridp++)
  {

    /* When there is net heating, the temperature was too low -> increase temperature */

    if (thermal_ratio[gridp] > 0.0)
    {

      /* Get a lower bound on the temperature for Brent's algorithm */

      // if (temperature_gas[gridp] < temperature_a[gridp]){

        temperature_a[gridp]   = 0.95*temperature_gas[gridp];

        thermal_ratio_a[gridp] = thermal_ratio[gridp];
      // }


      /* When we also increrased the tempoerature previous iteration */

      if (prev_temperature_gas[gridp] < temperature_gas[gridp])
      {
        prev_temperature_gas[gridp] = temperature_gas[gridp];

        temperature_gas[gridp]      = 1.2 * temperature_gas[gridp];
      }


      /* When we decreased the temperature previous iteration */

      else
      {
        double temp = temperature_gas[gridp];

        temperature_gas[gridp]      = ( temp + prev_temperature_gas[gridp] ) / 2.0;

        prev_temperature_gas[gridp] = temp;
      }


    } /* end of net heating */



    /* When there is net cooling, the temperature was too high -> decrease temperature */

    if (thermal_ratio[gridp] < 0.0)
    {

      /* Get an upper bound on the temperature for Brent's algorithm */

      // if (temperature_gas[gridp] > temperature_b[gridp]){

        temperature_b[gridp]   = 1.05*temperature_gas[gridp];

        thermal_ratio_b[gridp] = thermal_ratio[gridp];
      // }


      /* When we also decrerased the tempoerature previous iteration */

      if (prev_temperature_gas[gridp] > temperature_gas[gridp])
      {
        prev_temperature_gas[gridp] = temperature_gas[gridp];

        temperature_gas[gridp]      = 0.8 * temperature_gas[gridp];
      }


      /* When we increased the temperature previous iteration */

      else
      {
        double temp = temperature_gas[gridp];

        temperature_gas[gridp]      = ( temp + prev_temperature_gas[gridp] ) / 2.0;

        prev_temperature_gas[gridp] = temp;
      }

    } /* end of net cooling */



    /* Enforce the minimun and maximum temperature */

    if (temperature_gas[gridp] < TEMPERATURE_MIN)
    {
      temperature_gas[gridp] = TEMPERATURE_MIN;
    }

    else if (temperature_gas[gridp] > TEMPERATURE_MAX)
    {
      temperature_gas[gridp] = TEMPERATURE_MAX;
    }

  } /* end of gridp loop over grid points */
  } /* end of OpenMP parallel region */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* shuffle_temperatures: rename the variables for Brent's method                                 */
/*-----------------------------------------------------------------------------------------------*/

int shuffle_Brent( long gridp, double *temperature_a, double *temperature_b, double *temperature_c,
                   double *temperature_d, double *temperature_e, double *thermal_ratio_a,
                   double *thermal_ratio_b, double *thermal_ratio_c )
{


  /* Shuffle method used in the Van Wijngaarden-Dekker-Brent rootfinding algorithm */
  /* (see Numerical Recipes 9.4 for the original algorithm) */


  if ( (thermal_ratio_b[gridp] > 0.0 && thermal_ratio_c[gridp] > 0.0)
       || (thermal_ratio_b[gridp] < 0.0 && thermal_ratio_c[gridp] < 0.0) )
  {
    temperature_c[gridp]   = temperature_a[gridp];

    thermal_ratio_c[gridp] = thermal_ratio_a[gridp];

    temperature_d[gridp]   = temperature_b[gridp] - temperature_a[gridp];

    temperature_e[gridp]   = temperature_d[gridp];
  }


  if ( fabs(thermal_ratio_c[gridp]) < fabs(thermal_ratio_b[gridp]) )
  {
    temperature_a[gridp]   = temperature_b[gridp];
    temperature_b[gridp]   = temperature_c[gridp];
    temperature_c[gridp]   = temperature_a[gridp];

    thermal_ratio_a[gridp] = thermal_ratio_b[gridp];
    thermal_ratio_b[gridp] = thermal_ratio_c[gridp];
    thermal_ratio_c[gridp] = thermal_ratio_a[gridp];
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* update_temperature_gas: update the gas temperature using Brent's method                       */
/*-----------------------------------------------------------------------------------------------*/

int update_temperature_gas_Brent( long gridp, double *temperature_a, double *temperature_b,
                                  double *temperature_c, double *temperature_d,
                                  double *temperature_e, double *thermal_ratio_a,
                                  double *thermal_ratio_b, double *thermal_ratio_c )
{


  /* Update method based on the Van Wijngaarden-Dekker-Brent rootfinding algorithm */
  /* (see Numerical Recipes 9.4 for the original algorithm) */


  double tolerance = 2.0*EPSILON*fabs(temperature_b[gridp])+0.5*PREC;

  double xm = (temperature_c[gridp] - temperature_b[gridp]) / 2.0;


  if ( (fabs(temperature_e[gridp]) >= tolerance)
       && (fabs(thermal_ratio_a[gridp]) > fabs(thermal_ratio_b[gridp])) )
  {

    /* Attempt inverse quadratic interpolation */

    double s = thermal_ratio_b[gridp] / thermal_ratio_a[gridp];

    double p, q, r;

    if (temperature_a[gridp] == temperature_c[gridp])
    {
      p = 2.0 * xm * s;
      q = 1.0 - s;
    }

    else
    {
      q = thermal_ratio_a[gridp] / thermal_ratio_c[gridp];
      r = thermal_ratio_b[gridp] / thermal_ratio_c[gridp];
      p = s*( 2.0*xm*q*(q-r) - (temperature_b[gridp]-temperature_a[gridp])*(r-1.0) );
      q = (q-1.0)*(r-1.0)*(s-1.0);
    }

    if (p > 0.0){ q = -q; }

    p = fabs(p);


    double min1 = 3.0*xm*q - fabs(tolerance*q);
    double min2 = fabs(temperature_e[gridp]*q);


    if (2.0*p < (min1 < min2 ? min1 :  min2))                            /* Accept interpolation */
    {
      temperature_e[gridp] = temperature_d[gridp];
      temperature_d[gridp] = p / q;
    }

    else                                                  /* Interpolation failed, use bisection */
    {
      temperature_d[gridp] = xm;
      temperature_e[gridp] = temperature_d[gridp];
    }

  }

  else                                            /* Bounds decreasing too slowly, use bisection */
  {
    temperature_d[gridp] = xm;
    temperature_e[gridp] = temperature_d[gridp];
  }


  /* Move last best guess to temperature_a */

  temperature_a[gridp]   = temperature_b[gridp];

  thermal_ratio_a[gridp] = thermal_ratio_b[gridp];


  /* Evaluate new trial root */

  if ( fabs(temperature_d[gridp]) > tolerance )
  {
    temperature_b[gridp] = temperature_b[gridp] + temperature_d[gridp];
  }

  else
  {
    if (xm > 0.0){ temperature_b[gridp] = temperature_b[gridp] + fabs(tolerance); }

    else { temperature_b[gridp] = temperature_b[gridp] - fabs(tolerance); }
  }



  /* Enforce the minimun and maximum temperature */

  if(temperature_b[gridp] < TEMPERATURE_MIN)
  {
    temperature_b[gridp] = TEMPERATURE_MIN;
  }

  else if (temperature_b[gridp] > TEMPERATURE_MAX)
  {
    temperature_b[gridp] = TEMPERATURE_MAX;
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
