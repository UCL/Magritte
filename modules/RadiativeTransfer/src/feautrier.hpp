// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELL_FEAUTRIER_HPP_INCLUDED__
#define __CELL_FEAUTRIER_HPP_INCLUDED__

#include <Eigen/Core>


/// feautrier: solve radiative transfer equation using the Feautrier method
/// and the numerical scheme devised by Rybicki & Hummer (1991)
/// @param[in] n_r: number of points on ray r
/// @param[in] *S_r: pointer to source function data along ray r
/// @param[in] *dtau_r: pointer to optical depth increments along ray r
/// @param[in] n_ar: number of points on ray ar
/// @param[in] *S_ar: pointer to source function data along ray ar
/// @param[in] *dtau_ar: pointer to optical depth increments along ray ar
/// @param[out] *u: pointer to resulting Feautrier mean intensity vector
/// @param[out] Lambda: approximate Lambda operator (ALO) for this ray pair
/// @param[in] ndiag: degree of approximation in ALO (e.g. 0->diag, 1->tridiag)
///////////////////////////////////////////////////////////////////////////////

int feautrier (const long n_r, double *S_r, double *dtau_r,
	             const long n_ar, double *S_ar, double *dtau_ar,
	             double *u, Eigen::Ref <Eigen::MatrixXd> Lambda, const long ndiag);


/// feautrier: solve radiative transfer equation using the Feautrier method
/// and the numerical scheme devised by Rybicki & Hummer (1991)
/// @param[in] n_r: number of points on ray r
/// @param[in] *S_r: pointer to source function data along ray r
/// @param[in] *dtau_r: pointer to optical depth increments along ray r
/// @param[in] n_ar: number of points on ray ar
/// @param[in] *S_ar: pointer to source function data along ray ar
/// @param[in] *dtau_ar: pointer to optical depth increments along ray ar
/// @param[out] *u: pointer to resulting Feautrier mean intensity vector
///////////////////////////////////////////////////////////////////////////////

int feautrier (const long n_r, double *S_r, double *dtau_r,
	             const long n_ar, double *S_ar, double *dtau_ar,
	             double *v);


#endif // __CELL_FEAUTRIER_HPP_INCLUDED__
