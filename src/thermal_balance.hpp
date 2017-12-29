// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __THERMAL_BALANCE_HPP_INCLUDED__
#define __THERMAL_BALANCE_HPP_INCLUDED__


#include "declarations.hpp"


// thermal_balance: perform a thermal balance iteration to calculate thermal flux
// ------------------------------------------------------------------------------

int thermal_balance( CELL *cell,
                     double *column_H2, double *column_HD, double *column_C, double *column_CO,
                     double *UV_field, double *temperature_gas, double *temperature_dust,
                     double *rad_surface, double *AV, int *irad, int *jrad, double *energy,
                     double *weight, double *frequency, double *A_coeff, double *B_coeff,
                     double *C_data, double *coltemp, int *icol, int *jcol, double *pop,
                     double *mean_intensity, double *Lambda_diagonal, double *mean_intensity_eff,
                     double *thermal_ratio, double *initial_abn,
                     double *time_chemistry, double *time_level_pop );


#endif // __THERMAL_BALANCE_HPP_INCLUDED__
