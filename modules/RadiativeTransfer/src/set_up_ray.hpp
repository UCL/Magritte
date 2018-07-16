// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SET_UP_RAY_HPP_INCLUDED__
#define __SET_UP_RAY_HPP_INCLUDED__


#include <vector>
using namespace std;

#include "cells.hpp"
#include "types.hpp"
#include "GridTypes.hpp"
#include "lines.hpp"
#include "scattering.hpp"
#include "radiation.hpp"

enum RAYTYPE {ray, antipod};

///  set_up_ray: extract sources and opacities from the grid on the ray
///    @param[in] cells: reference to the geometric cell data containing the grid
///    @param[in] frequencies: reference to data structure containing freqiencies
///    @param[in] lines: reference to data structure containing line transfer data
///    @param[in] scattering: reference to structure containing scattering data
///    @param[in] radiation: reference to (previously calculated) radiation field
///    @param[in] o: number of the cell from which the ray originates
///    @param[in] r: number of the ray which is being set up
///		 @param[in] R: local number of the ray which is being set up
///    @param[in] raytype: indicates whether we walk forward or backward along ray
///    @param[out] n: reference to the resulting number of points along the ray
///    @param[out] Su: reference to the source for u extracted along the ray
///    @param[out] Sv: reference to the source for v extracted along the ray
///    @param[out] dtau: reference to the optical depth increments along the ray 
//////////////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
int set_up_ray (const CELLS <Dimension, Nrays>& cells, FREQUENCIES& frequencies,
		            const TEMPERATURE& temperature, LINES& lines,
								const SCATTERING& scattering, RADIATION& radiation,
								const long o, const long r, const long R, const RAYTYPE raytype,
	              long& n, vReal2& Su, vReal2& Sv, vReal2& dtau);


#include "set_up_ray.tpp"


#endif // __SET_UP_RAY_HPP_INCLUDED__
