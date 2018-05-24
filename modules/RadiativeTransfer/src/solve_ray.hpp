// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SOLVE_RAY_HPP_INCLUDED__
#define __SOLVE_RAY_HPP_INCLUDED__

#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;


///  solve_ray: solve radiative transfer equation using the Feautrier method
///  and the numerical scheme devised by Rybicki & Hummer (1991)
///    @param[in] n_r: number of points on ray r
///    @param[in] Su_r: reference to source function for u along ray r
///    @param[in] Sv_r: reference to source function for v along ray r
///    @param[in] dtau_r: reference to optical depth increments along ray r
///    @param[in] n_ar: number of points on ray ar
///    @param[in] Su_ar: reference to source function for u along ray ar
///    @param[in] Sv_ar: reference to source function for v along ray ar
///    @param[in] dtau_ar: pointer to optical depth increments along ray ar
///    @param[in] ndep: total number of points on the combined ray r and ar
///    @param[in] nfreq: number of frequency bins
///    @param[out] u: reference to resulting Feautrier mean intensity vector
///    @param[out] v: reference to resulting Feautrier flux intensity vector
///    @param[in] ndiag: degree of approximation in ALO (e.g. 0->diag, 1->tridiag)
///    @param[out] Lambda: approximate Lambda operator (ALO) for this ray pair
//////////////////////////////////////////////////////////////////////////////////

int solve_ray (const long n_r,  vector<double>& Su_r,  vector<double>& Sv_r,  vector<double>& dtau_r,
	             const long n_ar, vector<double>& Su_ar, vector<double>& Sv_ar, vector<double>& dtau_ar,
	             const long ndep, const long nfreq,      vector<double>& u,     vector<double>& v,
							 const long ndiag, Ref<MatrixXd> Lambda);


#endif // __SOLVE_RAY_HPP_INCLUDED__
