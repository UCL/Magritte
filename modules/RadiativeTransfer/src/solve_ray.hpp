// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SOLVE_RAY_HPP_INCLUDED__
#define __SOLVE_RAY_HPP_INCLUDED__

#include <Eigen/Core>
using namespace Eigen;

#include "types.hpp"
#include "GridTypes.hpp"


///  solve_ray: solve radiative transfer equation using the Feautrier method
///  and the numerical scheme devised by Rybicki & Hummer (1991)
///    @param[in] n_r: number of points on ray r
///    @param[in] Su_r: reference to source function for u along ray r
///    @param[in] Sv_r: reference to source function for v along ray r
///    @param[in] dtau_r: reference to optical depth increments along ray r
///    @param[in] n_ar: number of points on ray ar
///    @param[in] Su_ar: reference to source function for u along ray ar
///    @param[in] Sv_ar: reference to source function for v along ray ar
///    @param[in] dtau_ar: reference to optical depth increments along ray ar
///    @param[in] ndep: total number of points on the combined ray r and ar
///    @param[in] f: frequency index under consideration
///    @param[out] u: reference to resulting Feautrier mean intensity vector
///    @param[out] v: reference to resulting Feautrier flux intensity vector
///    @param[in] ndiag: degree of approximation in ALO (e.g. 0->diag, 1->tridiag)
///    @param[out] Lambda: approximate Lambda operator (ALO) for this ray pair
//////////////////////////////////////////////////////////////////////////////////

int solve_ray (const long n_r,  const vReal* Su_r,  const vReal* Sv_r,  const vReal* dtau_r,
	             const long n_ar, const vReal* Su_ar, const vReal* Sv_ar, const vReal* dtau_ar,
							  /*   vReal* A,          vReal* C,           vReal* F,           vReal* G,
									 vReal& B0,         vReal& B0_min_C0,   vReal& Bd,          vReal& Bd_min_Ad,*/
	             const long ndep,       vReal* u,           vReal* v,
							 const long ndiag,      vReal* Lambda, const long ncells);


#endif // __SOLVE_RAY_HPP_INCLUDED__
