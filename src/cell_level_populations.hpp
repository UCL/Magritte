/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for level_populations_otf.cpp                                                          */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __CELL_LEVEL_POPULATIONS_HPP_INCLUDED__
#define __CELL_LEVEL_POPULATIONS_HPP_INCLUDED__



#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"



#if (CELL_BASED)

/* level_populations: iteratively calculates the level populations                               */
/*-----------------------------------------------------------------------------------------------*/

int cell_level_populations( CELL *cell, int *irad, int*jrad, double *frequency,
                            double *A_coeff, double *B_coeff, double *pop,
                            double *C_data, double *coltemp, int *icol, int *jcol,
                            double *temperature_gas, double *temperature_dust,
                            double *weight, double *energy, double *mean_intensity,
                            double *Lambda_diagonal, double *mean_intensity_eff );

/*-----------------------------------------------------------------------------------------------*/

#endif



#endif /* __CELL_LEVEL_POPULATIONS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
