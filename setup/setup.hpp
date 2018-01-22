/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for initializers.cpp                                                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __SETUP_HPP_INCLUDED__
#define __SETUP_HPP_INCLUDED__

#include <string>



/* write_int_array: write an array of int to the config file                                     */
/*-----------------------------------------------------------------------------------------------*/

int write_int_array(FILE *file, std::string NAME, int *array, long length);

/*-----------------------------------------------------------------------------------------------*/



/* write_long_array: write an array of long to the config file                                     */
/*-----------------------------------------------------------------------------------------------*/

int write_long_array(FILE *file, std::string NAME, long *array, long length);

/*-----------------------------------------------------------------------------------------------*/


// write_long_matrix: write a matrix of longs to config file
// -------------------------------------------------------

int write_long_matrix (FILE *file, std::string NAME, long **array, long nrows, long ncols);


/* write_double_array: write an array of int to the config file                                  */
/*-----------------------------------------------------------------------------------------------*/

int write_double_array(FILE *file, std::string NAME, double *array, long length);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __SETUP_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
