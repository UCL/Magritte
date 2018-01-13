// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>

#include "setup_definitions.hpp"
#include "setup_initializers.hpp"


// initialize_int_array: sets all entries of linearized array of ints equal to zero
// --------------------------------------------------------------------------------

int initialize_int_array (long length, int *array)
{

  for (long i = 0; i < length; i++)
  {
    array[i] = 0;
  }


  return (0);

}




// initialize_long_array: sets all entries of linearized array of longs equal to zero
// ----------------------------------------------------------------------------------

int initialize_long_array (long length, long *array)
{

  for (long i = 0; i < length; i++)
  {
    array[i] = 0;
  }


  return(0);

}




// initialize_double_array: sets all entries of linearized array of doubles equal to zero
// --------------------------------------------------------------------------------------

int initialize_double_array (long length, double *array)
{

  for (long i = 0; i < length; i++)
  {
    array[i] = 0.0;
  }


  return(0);

}




// initialize_double_array_with: sets entries of first array of doubles equal to second
// ------------------------------------------------------------------------------------

int initialize_double_array_with (long length, double *array1, double *array2)
{

  for (long i = 0; i < length; i++)
  {
    array1[i] = array2[i];
  }


  return(0);

}




// initialize_char_array: sets all entries of linearized array of doubles equal to 'i'
// -----------------------------------------------------------------------------------

int initialize_char_array (long length, char *array)
{

  for (long i = 0; i < length; i++)
  {
    array[i] = 0.0;
  }


  return(0);

}




// initialize_bool: initialize a boolean variable
// ----------------------------------------------

int initialize_bool (long length, bool value, bool *variable)
{

  for (long n = 0; n < length; n++)
  {
    variable[n] = value;
  }


  return(0);

}
