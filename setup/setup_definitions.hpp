/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Definitions                                                                                   */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __SETUP_DEFINITIONS_HPP_INCLUDED__
#define __SETUP_DEFINITIONS_HPP_INCLUDED__


#include <string>

#include "../parameters.hpp"


#define EXIT_SUCCESS 0


#define TOL 1.0E-9                                               /* tolerance for antipodal rays */


#define LSPECPAR(lspec,par)   ( (par) + cum_ncolpar[(lspec)] )                                    \
                   /* when first index is line producing species and second is collision partner */

#define VINDEX(r,c) ( (c) + (r)*3 )                 /* when the second index is a 3-vector index */



#endif /* __SETUP_DEFINITIONS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
