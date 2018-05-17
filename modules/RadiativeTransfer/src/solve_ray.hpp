// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SOLVE_RAY_HPP_INCLUDED__
#define __SOLVE_RAY_HPP_INCLUDED__

#include <Eigen/Core>


///  solve_ray: solve radiative transfer equation using the Feautrier method
///  and the numerical scheme devised by Rybicki & Hummer (1991)
///    @param[in] n_r: number of points on ray r
///    @param[in] *Su_r: pointer to source function for u along ray r
///    @param[in] *Sv_r: pointer to source function for v along ray r
///    @param[in] *dtau_r: pointer to optical depth increments along ray r
///    @param[in] n_ar: number of points on ray ar
///    @param[in] *Su_ar: pointer to source function for u along ray ar
///    @param[in] *Sv_ar: pointer to source function for v along ray ar
///    @param[in] *dtau_ar: pointer to optical depth increments along ray ar
///    @param[in] ndep: total number of points on the combined ray r and ar
///    @param[out] *u: pointer to resulting Feautrier mean intensity vector
///    @param[out] *v: pointer to resulting Feautrier flux intensity vector
///    @param[in] ndiag: degree of approximation in ALO (e.g. 0->diag, 1->tridiag)
///    @param[out] Lambda: approximate Lambda operator (ALO) for this ray pair
//////////////////////////////////////////////////////////////////////////////////

int solve_ray (const long n_r,  double *Su_r,  double *Sv_r,  double *dtau_r,
	             const long n_ar, double *Su_ar, double *Sv_ar, double *dtau_ar,
	             const long ndep, double *u,     double *v,
							 const long ndiag, Eigen::Ref <Eigen::MatrixXd> Lambda);


#endif // __SOLVE_RAY_HPP_INCLUDED__
