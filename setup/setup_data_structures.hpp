/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for setup_data_structures.cpp                                                          */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __SETUP_DATA_STRUCTURES_HPP_INCLUDED__
#define __SETUP_DATA_STRUCTURES_HPP_INCLUDED__

#include <string>



/* setup_data_structures1: set up the first part of the different datastructures                 */
/*-----------------------------------------------------------------------------------------------*/

int setup_data_structures1( std::string *line_datafile, int *nlev, int *nrad, int *cum_nlev,
                            int *cum_nrad, int *cum_nlev2, int *ncolpar, int *cum_ncolpar );

/*-----------------------------------------------------------------------------------------------*/



/* setup_data_structures2: set up the second part of the different datastructures                */
/*-----------------------------------------------------------------------------------------------*/

int setup_data_structures2( std::string *line_datafile, int* ncolpar, int *cum_ncolpar,
                            int *ncoltran, int *ncoltemp,
                            int *cum_ncoltran, int *cum_ncoltemp, int *cum_ncoltrantemp,
                            int *tot_ncoltran, int *tot_ncoltemp, int *tot_ncoltrantemp,
                            int *cum_tot_ncoltran, int *cum_tot_ncoltemp,
                            int *cum_tot_ncoltrantemp );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __SETUP_DATA_STRUCTURES_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/