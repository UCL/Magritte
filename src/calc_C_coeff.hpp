/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for calc_C_coeff.cpp                                                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __CALC_C_COEFF_HPP_INCLUDED__
#define __CALC_C_COEFF_HPP_INCLUDED__



/* calc_C_coeff: calculates the collisional coefficients (C_ij) from the line data               */
/*-----------------------------------------------------------------------------------------------*/

void calc_C_coeff( double *C_data, double *coltemp, int *icol, int *jcol, double *temperature,
                   double *weight, double *energy, double *C_coeff, long gridp, int lspec );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __CALC_C_COEFF_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
