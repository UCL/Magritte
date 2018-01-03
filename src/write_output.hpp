// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __WRITE_OUTPUT_HPP_INCLUDED__
#define __WRITE_OUTPUT_HPP_INCLUDED__


#include <string>

#include "declarations.hpp"



// write_txt_output: write output in txt format
// --------------------------------------------

int write_txt_output (long ncells, CELL *cell, double *pop, double *mean_intensity);


// write_performance_log: write performance results
// ------------------------------------------------

int write_performance_log (double time_total, double time_level_pop, double time_chemistry,
                           double time_ray_tracing, int niterations);


#endif // __WRITE_OUTPUT_HPP_INCLUDED__
