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



#define LSPECPAR(lspec,par)   ( (par) + cum_ncolpar[(lspec)] )                                    \
                   /* when first index is line producing species and second is collision partner */



#endif /* __SETUP_DEFINITIONS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
