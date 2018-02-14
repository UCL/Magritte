// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __feautrier_HPP_INCLUDED__
#define __feautrier_HPP_INCLUDED__

#include "declarations.hpp"


// feautrier: fill Feautrier matrix, solve it
// ------------------------------------------

int feautrier( EVALPOINT *evalpoint, long *key, long *raytot, long *cum_raytot, long gridp, long r,
               double *S, double *dtau, double *u, double *L_diag_approx );


#endif // __feautrier_HPP_INCLUDED__
