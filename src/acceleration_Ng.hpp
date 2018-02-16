// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __ACCELERATION_NG_HPP_INCLUDED__
#define __ACCELERATION_NG_HPP_INCLUDED__


// acceleration_Ng: perform a Ng accelerated iteration for level populations
// -------------------------------------------------------------------------

int acceleration_Ng (long ncells, CELL *cell, int lspec,
                     double *prev3_pop, double *prev2_pop, double *prev1_pop);


// store_populations: update previous populations
// ----------------------------------------------

int store_populations (long ncells, CELL *cell, int lspec,
                       double *prev3_pop, double *prev2_pop, double *prev1_pop);


#endif // __ACCELERATION_NG_HPP_INCLUDED__
