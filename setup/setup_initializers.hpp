// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SETUP_INITIALIZERS_HPP_INCLUDED__
#define __SETUP_INITIALIZERS_HPP_INCLUDED__


// initialize_int_array: sets all entries of linearized array of ints equal to zero
// --------------------------------------------------------------------------------

int initialize_int_array (long length, int *array);


// initialize_long_array: sets all entries of linearized array of longs equal to zero
// ----------------------------------------------------------------------------------

int initialize_long_array (long length, long *array);


// initialize_double_array: sets all entries of linearized array of doubles equal to zero
// --------------------------------------------------------------------------------------

int initialize_double_array (long length, double *array);


// initialize_double_array_with: sets entries of first array of doubles equal to second
// ------------------------------------------------------------------------------------

int initialize_double_array_with (long length, double *array1, double *array2);


// initialize_char_array: sets all entries of linearized array of doubles equal to 'i'
// -----------------------------------------------------------------------------------

int initialize_char_array (long length, char *array);


// initialize_bool: initialize a boolean variable
// ----------------------------------------------

int initialize_bool (long length, bool value, bool *variable);


#endif // __SETUP_INITIALIZERS_HPP_INCLUDED__
