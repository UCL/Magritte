// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SET_UP_RAY_HPP_INCLUDED__
#define __SET_UP_RAY_HPP_INCLUDED__


#include <vector>
using namespace std;

#include "cells.hpp"
#include "medium.hpp"
#include "radiation.hpp"


///  set_up_ray: extract sources and opacities from the grid on the ray
///    @param[in] cells: reference to the geometric cell data containing the grid
///    @param[in] radiation: reference to (previously calculated) radiation field
///    @param[in] medium: reference to the opacity and emissivity data of the medium
///    @param[in] o: number of the cell from which the ray originates
///    @param[in] r: number of the ray which is being set up
///    @param[in] f: number of the frequency bin which is used
///    @param[in] sign: +1  if the ray is in the "right" direction, "-1" if opposite
///    @param[out] n: reference to the resulting number of points along the ray
///    @param[out] Su: reference to the source for u extracted along the ray
///    @param[out] Sv: reference to the source for v extracted along the ray
///    @param[out] dtau: reference to the optical depth increments along the ray 
////////////////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays, long Nfreq>
int set_up_ray (CELLS <Dimension, Nrays>& cells, RADIATION& radiation,
		            MEDIUM& medium, long o, long r, long f, double sign,
	              long& n, vector<double>& Su, vector<double>& Sv, vector<double>& dtau);


#include "set_up_ray.tpp"


#endif // __SET_UP_RAY_HPP_INCLUDED__
