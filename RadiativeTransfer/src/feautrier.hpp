// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELL_FEAUTRIER_HPP_INCLUDED__
#define __CELL_FEAUTRIER_HPP_INCLUDED__


// feautrier: fill Feautrier matrix, solve it
// ------------------------------------------

int feautrier (long ndep, double *S, double *dtau, double *u, double *L_diag_approx);


#endif // __CELL_FEAUTRIER_HPP_INCLUDED__
