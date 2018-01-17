// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __COOLING_HPP_INCLUDED__
#define __COOLING_HPP_INCLUDED__


// cooling: calculate total cooling
// --------------------------------

double cooling (long ncells, LINE_SPECIES *line_species, long gridp, int *irad, int *jrad,
                double *A_coeff, double *B_coeff, double *frequency, double *weight,
                double *pop, double *mean_intensity);


#endif // __COOLING_HPP_INCLUDED__
