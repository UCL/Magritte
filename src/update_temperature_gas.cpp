// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <omp.h>


#include "declarations.hpp"
#include "update_temperature_gas.hpp"


#define EPSILON 3.0E-8   // lower bound on precision we can obtain
#define PREC    5.0E-3   // precision we need on temperature



// update_temperature_gas: update gas temperature after a thermal balance iteration
// --------------------------------------------------------------------------------

int update_temperature_gas (long ncells, CELL *cell, long gridp)
{

  // When there is net heating, temperature was too low -> increase temperature

  if (cell[gridp].thermal_ratio > 0.0)
  {

    // When we also increrased tempoerature previous iteration => up scaling

    if (cell[gridp].temperature.gas_prev < cell[gridp].temperature.gas)
    {
      cell[gridp].temperature.gas_prev = cell[gridp].temperature.gas;
      cell[gridp].temperature.gas      = 1.1 * cell[gridp].temperature.gas;
    }


    // When we decreased temperature previous iteration => binary chop

    else
    {
      double temp = cell[gridp].temperature.gas;

      cell[gridp].temperature.gas      = ( temp + cell[gridp].temperature.gas_prev ) / 2.0;
      cell[gridp].temperature.gas_prev = temp;
    }


  } // end of net heating



  // When there is net cooling, temperature was too high -> decrease temperature

  if (cell[gridp].thermal_ratio < 0.0)
  {

    // When we also decrerased tempoerature previous iteration => down scaling

    if (cell[gridp].temperature.gas_prev > cell[gridp].temperature.gas)
    {
      cell[gridp].temperature.gas_prev = cell[gridp].temperature.gas;
      cell[gridp].temperature.gas      = 0.9 * cell[gridp].temperature.gas;
    }


    // When we increased temperature previous iteration => binary chop

    else
    {
      double temp = cell[gridp].temperature.gas;

      cell[gridp].temperature.gas      = (temp + cell[gridp].temperature.gas_prev) / 2.0;
      cell[gridp].temperature.gas_prev = temp;
    }

  } // end of net cooling



  // Enforce the minimun and maximum temperature

  if      (cell[gridp].temperature.gas < TEMPERATURE_MIN)
  {
    cell[gridp].temperature.gas = TEMPERATURE_MIN;
  }

  else if (cell[gridp].temperature.gas > TEMPERATURE_MAX)
  {
    cell[gridp].temperature.gas = TEMPERATURE_MAX;
  }


  return (0);

}




// shuffle_temperatures: rename variables for Brent's method
// ---------------------------------------------------------

int shuffle_Brent (long gridp, double *temperature_a, double *temperature_b, double *temperature_c,
                   double *temperature_d, double *temperature_e, double *thermal_ratio_a,
                   double *thermal_ratio_b, double *thermal_ratio_c)
{

  // Shuffle method used in the Van Wijngaarden-Dekker-Brent rootfinding algorithm
  // (see Numerical Recipes 9.4 for the original algorithm)


  if (    (thermal_ratio_b[gridp] > 0.0 && thermal_ratio_c[gridp] > 0.0)
       || (thermal_ratio_b[gridp] < 0.0 && thermal_ratio_c[gridp] < 0.0) )
  {
    temperature_c[gridp]   = temperature_a[gridp];
    thermal_ratio_c[gridp] = thermal_ratio_a[gridp];
    temperature_d[gridp]   = temperature_b[gridp] - temperature_a[gridp];
    temperature_e[gridp]   = temperature_d[gridp];
  }


  if (fabs(thermal_ratio_c[gridp]) < fabs(thermal_ratio_b[gridp]))
  {
    temperature_a[gridp]   = temperature_b[gridp];
    temperature_b[gridp]   = temperature_c[gridp];
    temperature_c[gridp]   = temperature_a[gridp];

    thermal_ratio_a[gridp] = thermal_ratio_b[gridp];
    thermal_ratio_b[gridp] = thermal_ratio_c[gridp];
    thermal_ratio_c[gridp] = thermal_ratio_a[gridp];
  }


  return (0);

}




// update_temperature_gas: update gas temperature using Brent's method
// -------------------------------------------------------------------

int update_temperature_gas_Brent (long gridp, double *temperature_a, double *temperature_b,
                                  double *temperature_c, double *temperature_d,
                                  double *temperature_e, double *thermal_ratio_a,
                                  double *thermal_ratio_b, double *thermal_ratio_c)
{

  // Update method based on the Van Wijngaarden-Dekker-Brent rootfinding algorithm
  // (see Numerical Recipes 9.4 for the original algorithm)


  double tolerance = 2.0*EPSILON*fabs(temperature_b[gridp]) + 0.5*PREC;

  double xm = (temperature_c[gridp] - temperature_b[gridp]) / 2.0;


  if ( (fabs(temperature_e[gridp]) >= tolerance)
       && (fabs(thermal_ratio_a[gridp]) > fabs(thermal_ratio_b[gridp])) )
  {

    // Attempt inverse quadratic interpolation

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


    if (2.0*p < (min1 < min2 ? min1 :  min2))   // Accept interpolation
    {
      temperature_e[gridp] = temperature_d[gridp];
      temperature_d[gridp] = p / q;
    }

    else   // Interpolation failed, use bisection
    {
      temperature_d[gridp] = xm;
      temperature_e[gridp] = temperature_d[gridp];
    }

  }

  else   // Bounds decreasing too slowly, use bisection
  {
    temperature_d[gridp] = xm;
    temperature_e[gridp] = temperature_d[gridp];
  }


  // Move last best guess to temperature_a

  temperature_a[gridp]   = temperature_b[gridp];

  thermal_ratio_a[gridp] = thermal_ratio_b[gridp];


  // Evaluate new trial root

  if (fabs(temperature_d[gridp]) > tolerance)
  {
    temperature_b[gridp] = temperature_b[gridp] + temperature_d[gridp];
  }

  else
  {
    if (xm > 0.0)
    {
      temperature_b[gridp] = temperature_b[gridp] + fabs(tolerance);
    }

    else
    {
      temperature_b[gridp] = temperature_b[gridp] - fabs(tolerance);
    }
  }


  // Enforce minimun and maximum temperature

  if      (temperature_b[gridp] < TEMPERATURE_MIN)
  {
    temperature_b[gridp] = TEMPERATURE_MIN;
  }

  else if (temperature_b[gridp] > TEMPERATURE_MAX)
  {
    temperature_b[gridp] = TEMPERATURE_MAX;
  }


  return (0);

}
