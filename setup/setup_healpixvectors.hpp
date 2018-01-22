/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for setup_healpixvectors.cpp                                                           */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __SETUP_HEALPIXVECTORS_HPP_INCLUDED__
#define __SETUP_HEALPIXVECTORS_HPP_INCLUDED__



/* create_healpixvector: store the HEALPix vectors and find the antipodal pairs                  */
/*-----------------------------------------------------------------------------------------------*/

int setup_healpixvectors (long nrays, double *healpixvector, long *antipod, long *n_aligned, long **aligned);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __SETUP_HEALPIXVECTORS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
