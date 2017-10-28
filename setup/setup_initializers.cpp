/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Initializers.cpp: Initialization functions for all (linearized) arrays                          */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <math.h>

#include "setup_initializers.hpp"



/* initialize_int_array: sets all entries of the linearized array of ints equal to zero          */
/*-----------------------------------------------------------------------------------------------*/

int initialize_int_array(int *array, long length)
{


  for (long i=0; i<length; i++){

    array[i] = 0;
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/



/* initialize_long_array: sets all entries of the linearized array of longs equal to zero        */
/*-----------------------------------------------------------------------------------------------*/

int initialize_long_array(long *array, long length)
{


  for (long i=0; i<length; i++){

    array[i] = 0;
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_double_array: sets all entries of the linearized array of doubles equal to zero    */
/*-----------------------------------------------------------------------------------------------*/

int initialize_double_array(double *array, long length)
{


  for (long i=0; i<length; i++){

    array[i] = 0.0;
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_double_array_with: sets entries of the first array of doubles equal to the second  */
/*-----------------------------------------------------------------------------------------------*/

int initialize_double_array_with(double *array1, double *array2, long length)
{


  for (long i=0; i<length; i++){

    array1[i] = array2[i];
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_char_array: sets all entries of the linearized array of doubles equal to 'i'       */
/*-----------------------------------------------------------------------------------------------*/

int initialize_char_array(char *array, long length)
{


  for (long i=0; i<length; i++){

    array[i] = 0.0;
  }


  return(0);

}


/*-----------------------------------------------------------------------------------------------*/





/* initialize_bool: initialize a boolean variable                                                */
/*-----------------------------------------------------------------------------------------------*/

int initialize_bool(bool value, long length, bool *variable)
{


  for (long n=0; n<length; n++){

    variable[n] = value;
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
