// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SETUP_HPP_INCLUDED__
#define __SETUP_HPP_INCLUDED__

#include <string>


// write_int_array: write an array of int to config file
// -----------------------------------------------------

int write_int_array (FILE *file, std::string NAME, int *array, long length);


// write_long_array: write an array of long to config file
// -------------------------------------------------------

int write_long_array (FILE *file, std::string NAME, long *array, long length);


// write_long_matrix: write a matrix of longs to config file
// ---------------------------------------------------------

int write_long_matrix (FILE *file, std::string NAME, long **array, long nrows, long ncols);


// write_double_array: write an array of int to config file
// --------------------------------------------------------

int write_double_array (FILE *file, std::string NAME, double *array, long length);


#endif // __SETUP_HPP_INCLUDED__
