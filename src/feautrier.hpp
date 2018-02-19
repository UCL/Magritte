// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELL_FEAUTRIER_HPP_INCLUDED__
#define __CELL_FEAUTRIER_HPP_INCLUDED__

#include "declarations.hpp"

#if (CELL_BASED)


// feautrier: fill Feautrier matrix, solve it
// ------------------------------------------

int feautrier (long ndep, long o, long r, double *S, double *dtau,
                    double *u, double *L_diag_approx);


#endif // if CELL_BASED


#endif // __CELL_FEAUTRIER_HPP_INCLUDED__
