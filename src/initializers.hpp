/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for initializers.cpp                                                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __INITIALIZERS_HPP_INCLUDED__
#define __INITIALIZERS_HPP_INCLUDED__





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


/* initialize_char_array: sets all entries of the linearized array of doubles equal to 'i'       */
/*-----------------------------------------------------------------------------------------------*/

int initialize_char_array(char *array, long length);

/*-----------------------------------------------------------------------------------------------*/



/* initialize_evalpoint: sets all entries of the linearized array equal to zero or false         */
/*-----------------------------------------------------------------------------------------------*/

int initialize_evalpoint(EVALPOINT *evalpoint);

/*-----------------------------------------------------------------------------------------------*/



/* initialize_temperature_gas: set the gas temperature to a certain initial value                */
/*-----------------------------------------------------------------------------------------------*/

int initialize_temperature_gas(double *temperature_gas);

/*-----------------------------------------------------------------------------------------------*/



/* initialize_level_populations: sets pops of all line species to the thermal equilibrium value  */
/*-----------------------------------------------------------------------------------------------*/

int initialize_level_populations(double *energy, double *temperature_gas, double *pop);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __INITIALIZERS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
