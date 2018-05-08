// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELL_FEAUTRIER_HPP_INCLUDED__
#define __CELL_FEAUTRIER_HPP_INCLUDED__

#include <Eigen/Core>


// feautrier: fill Feautrier matrix, solve it
// ------------------------------------------

int feautrier (long n_r, double *S_r, double *dtau_r,
	             long n_ar, double *S_ar, double *dtau_ar,
	             double *u, Eigen::Ref <Eigen::MatrixXd> Lambda, long ndiag);


#endif // __CELL_FEAUTRIER_HPP_INCLUDED__
