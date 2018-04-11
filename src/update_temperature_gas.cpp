// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <math.h>
#include <omp.h>


#include "declarations.hpp"
#include "update_temperature_gas.hpp"


#define EPSILON 3.0E-8   // lower bound on precision we can obtain
#define PREC    5.0E-3   // precision we need on temperature



// update_temperature_gas: update gas temperature after a thermal balance iteration
// --------------------------------------------------------------------------------

int update_temperature_gas (long ncells, CELLS *cells, long o)
{

  // Secant method

  double T   = cells->temperature_gas[o];
  double T_p = cells->temperature_gas_prev[o];

  double f   = cells->thermal_ratio[o];
  double f_p = cells->thermal_ratio_prev[o];


  if ( /*(f == f_p) &&*/ f > 0.0 )
  {
    cells->temperature_gas[o] = 1.1 * cells->temperature_gas[o];
  }

  else if ( /*(f == f_p) &&*/ f < 0.0 )
  {
    cells->temperature_gas[o] = 0.9 * cells->temperature_gas[o];
  }

  else
  {
    cells->temperature_gas[o]      = (T*f_p - T_p*f) / (f_p - f);
    cells->temperature_gas_prev[o] = T;
  }


  // Enforce the minimun and maximum temperature

  if      (cells->temperature_gas[o] < TEMPERATURE_MIN)
  {
    cells->temperature_gas[o] = TEMPERATURE_MIN;
  }

  else if (cells->temperature_gas[o] > TEMPERATURE_MAX)
  {
    cells->temperature_gas[o] = TEMPERATURE_MAX;
  }


  return (0);

}


//
//
// // shuffle_temperatures: rename variables for Brent's method
// // ---------------------------------------------------------
//
// int shuffle_Brent (long o, double *temperature_a, double *temperature_b, double *temperature_c,
//                    double *temperature_d, double *temperature_e, double *thermal_ratio_a,
//                    double *thermal_ratio_b, double *thermal_ratio_c)
// {
//
//   // Shuffle method used in the Van Wijngaarden-Dekker-Brent rootfinding algorithm
//   // (see Numerical Recipes 9.4 for the original algorithm)
//
//
//   if (    (thermal_ratio_b[o] > 0.0 && thermal_ratio_c[o] > 0.0)
//        || (thermal_ratio_b[o] < 0.0 && thermal_ratio_c[o] < 0.0) )
//   {
//     temperature_c[o]   = temperature_a[o];
//     thermal_ratio_c[o] = thermal_ratio_a[o];
//     temperature_d[o]   = temperature_b[o] - temperature_a[o];
//     temperature_e[o]   = temperature_d[o];
//   }
//
//
//   if (fabs(thermal_ratio_c[o]) < fabs(thermal_ratio_b[o]))
//   {
//     temperature_a[o]   = temperature_b[o];
//     temperature_b[o]   = temperature_c[o];
//     temperature_c[o]   = temperature_a[o];
//
//     thermal_ratio_a[o] = thermal_ratio_b[o];
//     thermal_ratio_b[o] = thermal_ratio_c[o];
//     thermal_ratio_c[o] = thermal_ratio_a[o];
//   }
//
//
//   return (0);
//
// }
//
//
//
//
// // update_temperature_gas: update gas temperature using Brent's method
// // -------------------------------------------------------------------
//
// int update_temperature_gas_Brent (long o, double *temperature_a, double *temperature_b,
//                                   double *temperature_c, double *temperature_d,
//                                   double *temperature_e, double *thermal_ratio_a,
//                                   double *thermal_ratio_b, double *thermal_ratio_c)
// {
//
//   // Update method based on the Van Wijngaarden-Dekker-Brent rootfinding algorithm
//   // (see Numerical Recipes 9.4 for the original algorithm)
//
//
//   double tolerance = 2.0*EPSILON*fabs(temperature_b[o]) + 0.5*PREC;
//
//   double xm = (temperature_c[o] - temperature_b[o]) / 2.0;
//
//
//   if (    (fabs(temperature_e[o]) >= tolerance)
//        && (fabs(thermal_ratio_a[o]) > fabs(thermal_ratio_b[o])) )
//   {
//
//     // Attempt inverse quadratic interpolation
//
//     double s = thermal_ratio_b[o] / thermal_ratio_a[o];
//
//     double p, q, r;
//
//     if (temperature_a[o] == temperature_c[o])
//     {
//       p = 2.0 * xm * s;
//       q = 1.0 - s;
//     }
//
//     else
//     {
//       q = thermal_ratio_a[o] / thermal_ratio_c[o];
//       r = thermal_ratio_b[o] / thermal_ratio_c[o];
//       p = s*( 2.0*xm*q*(q-r) - (temperature_b[o]-temperature_a[o])*(r-1.0) );
//       q = (q-1.0)*(r-1.0)*(s-1.0);
//     }
//
//     if (p > 0.0){ q = -q; }
//
//     p = fabs(p);
//
//
//     double min1 = 3.0*xm*q - fabs(tolerance*q);
//     double min2 = fabs(temperature_e[o]*q);
//
//
//     if (2.0*p < (min1 < min2 ? min1 :  min2))   // Accept interpolation
//     {
//       temperature_e[o] = temperature_d[o];
//       temperature_d[o] = p / q;
//     }
//
//     else   // Interpolation failed, use bisection
//     {
//       temperature_d[o] = xm;
//       temperature_e[o] = temperature_d[o];
//     }
//
//   }
//
//   else   // Bounds decreasing too slowly, use bisection
//   {
//     temperature_d[o] = xm;
//     temperature_e[o] = temperature_d[o];
//   }
//
//
//   // Move last best guess to temperature_a
//
//   temperature_a[o]   = temperature_b[o];
//
//   thermal_ratio_a[o] = thermal_ratio_b[o];
//
//
//   // Evaluate new trial root
//
//   if (fabs(temperature_d[o]) > tolerance)
//   {
//     temperature_b[o] = temperature_b[o] + temperature_d[o];
//   }
//
//   else
//   {
//     if (xm > 0.0)
//     {
//       temperature_b[o] = temperature_b[o] + fabs(tolerance);
//     }
//
//     else
//     {
//       temperature_b[o] = temperature_b[o] - fabs(tolerance);
//     }
//   }
//
//
//   // Enforce minimun and maximum temperature
//
//   if      (temperature_b[o] < TEMPERATURE_MIN)
//   {
//     temperature_b[o] = TEMPERATURE_MIN;
//   }
//
//   else if (temperature_b[o] > TEMPERATURE_MAX)
//   {
//     temperature_b[o] = TEMPERATURE_MAX;
//   }
//
//
//   return (0);
//
// }
