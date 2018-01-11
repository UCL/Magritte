// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __INITIALIZERS_HPP_INCLUDED__
#define __INITIALIZERS_HPP_INCLUDED__

#include "declarations.hpp"


// initialize_int_array: sets all entries of array of ints equal to zero
// ---------------------------------------------------------------------

int initialize_int_array (int *array, long length);


// initialize_long_array: sets all entries of array of longs equal to zero
// ------------------------------------------ ----------------------------

int initialize_long_array (long *array, long length);


// initialize_double_array: sets all entries of array of doubles equal to zero
// ---------------------------------------------------------------------------

int initialize_double_array (double *array, long length);


// initialize_double_array_with: sets entries of first array of doubles equal to second
// ------------------------------------------------------------------------------------

int initialize_double_array_with (double *array1, double *array2, long length);


// initialize_double_array_with_value: sets entries of array of doubles equal to value
// -----------------------------------------------------------------------------------

int initialize_double_array_with_value (double *array, double value, long length);


// initialize_char_array: sets all entries of linearized array of doubles equal to 'i'
// -----------------------------------------------------------------------------------

int initialize_char_array (char *array, long length);


// initialize_cells: initialize the cell array
// -------------------------------------------

int initialize_cells (CELL *cell, long ncells);


// initialize_cell_id: initialize the cell id's
// --------------------------------------------

int initialize_cell_id (CELL *cell, long ncells);


// initialize_temperature_gas: set gas temperature to a certain initial value
// --------------------------------------------------------------------------

int initialize_temperature_gas (long ncells, CELL *cell);


// initialize_previous_temperature_gas: set "previous" gas temperature 0.9*temperature_gas
// ---------------------------------------------------------------------------------------

int initialize_previous_temperature_gas (long ncells, CELL *cell);


// gueess_temperature_gas: make a guess for gas temperature based on UV field
// --------------------------------------------------------------------------

int guess_temperature_gas (long ncells, CELL *cell, double *UV_field);


// initialize_abundances: set abundanceces to initial values
// ---------------------------------------------------------

int initialize_abundances (long ncells, CELL *cell, SPECIES *species);


// initialize_bool: initialize a boolean variable
// ----------------------------------------------

int initialize_bool (bool value, bool *variable, long length);


#endif // __INITIALIZERS_HPP_INCLUDED__
