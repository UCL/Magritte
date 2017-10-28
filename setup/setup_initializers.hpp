/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for initializers.cpp                                                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __SETUP_INITIALIZERS_HPP_INCLUDED__
#define __SETUP_INITIALIZERS_HPP_INCLUDED__





/* initialize_int_array: sets all entries of the linearized array of ints equal to zero          */
/*-----------------------------------------------------------------------------------------------*/

int initialize_int_array(int *array, long length);

/*-----------------------------------------------------------------------------------------------*/



/* initialize_long_array: sets all entries of the linearized array of longs equal to zero        */
/*-----------------------------------------------------------------------------------------------*/

int initialize_long_array(long *array, long length);

/*-----------------------------------------------------------------------------------------------*/



/* initialize_double_array: sets all entries of the linearized array of doubles equal to zero    */
/*-----------------------------------------------------------------------------------------------*/

int initialize_double_array(double *array, long length);

/*-----------------------------------------------------------------------------------------------*/



/* initialize_double_array_with: sets entries of the first array of doubles equal to the second  */
/*-----------------------------------------------------------------------------------------------*/

int initialize_double_array_with(double *array1, double *array2, long length);

/*-----------------------------------------------------------------------------------------------*/



/* initialize_char_array: sets all entries of the linearized array of doubles equal to 'i'       */
/*-----------------------------------------------------------------------------------------------*/

int initialize_char_array(char *array, long length);

/*-----------------------------------------------------------------------------------------------*/



/* initialize_bool: initialize a boolean variable                                                */
/*-----------------------------------------------------------------------------------------------*/

int initialize_bool(bool value, long length, bool *variable);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __SETUP_INITIALIZERS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
