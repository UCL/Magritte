// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SOLVE_RAY_HPP_INCLUDED__
#define __SOLVE_RAY_HPP_INCLUDED__

#include <Eigen/Core>
using namespace Eigen;

#include "GridTypes.hpp"


///  solve_ray: solve radiative transfer equation using the Feautrier method
///  and the numerical scheme devised by Rybicki & Hummer (1991)
///    @param[in] ndep: number of points on ray this ray pair
///    @param[in/out] Su: in pointer to source function for u / out solution for u
///    @param[in/out] Sv: in pointer to source function for v / out solution for v
///    @param[in] dtau: pointer to optical depth increments along ray r
///    @param[in] ndep: total number of points on the combined ray r and ar
///    @param[in] ndiag: degree of approximation in ALO (e.g. 0->diag, 1->tridiag)
///    @param[out] Lambda: approximate Lambda operator (ALO) for this ray pair
//////////////////////////////////////////////////////////////////////////////////

inline int solve_ray (const long ndep, vReal* Su, vReal* Sv, const vReal* dtau,
			        				const long ndiag, vReal* Lambda, const long ncells);


#include "solve_ray.cpp"


#endif // __SOLVE_RAY_HPP_INCLUDED__
