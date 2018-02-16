// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __UPDATE_TEMPERATURE_GAS_HPP_INCLUDED__
#define __UPDATE_TEMPERATURE_GAS_HPP_INCLUDED__


// update_temperature_gas: update gas temperature after a thermal balance iteration
// --------------------------------------------------------------------------------

int update_temperature_gas (long ncells, CELL *cell, double *thermal_ratio, long gridp);


// shuffle_temperatures: rename variables for Brent's method
// ---------------------------------------------------------

int shuffle_Brent (long gridp, double *temperature_a, double *temperature_b, double *temperature_c,
                   double *temperature_d, double *temperature_e, double *thermal_ratio_a,
                   double *thermal_ratio_b, double *thermal_ratio_c);


// update_temperature_gas: update gas temperature using Brent's method
// ------------------------------ ------------------------------------

int update_temperature_gas_Brent (long gridp, double *temperature_a, double *temperature_b,
                                  double *temperature_c, double *temperature_d,
                                  double *temperature_e, double *thermal_ratio_a,
                                  double *thermal_ratio_b, double *thermal_ratio_c);


#endif // __UPDATE_TEMPERATURE_GAS_HPP_INCLUDED__
