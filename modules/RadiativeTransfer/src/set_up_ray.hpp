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
inline int set_up_ray (const CELLS<Dimension, Nrays>& cells, /*const RAYTYPE raytype,*/
	                     FREQUENCIES& frequencies, const TEMPERATURE& temperature,
								       LINES& lines, const SCATTERING& scattering, RADIATION& radiation,
								       const long f, const long o, const long R,
								       long *lnotch_ar, long *notch_ar, const long *cellNrs_ar,
								       const double *shifts_ar, const double *dZs_ar, const long n_ar,
								       long *lnotch_r, long *notch_r, const long *cellNrs_r,
								       const double *shifts_r, const double *dZs_r, const long n_r,
	                     vReal* Su, vReal* Sv, vReal* dtau, const long ndep);


inline int get_eta_and_chi (FREQUENCIES& frequencies, const TEMPERATURE& temperature,
	                          LINES& lines, const SCATTERING& scattering, RADIATION& radiation,
						                const long f, const long o, long& lnotch, const long cellNrs,
														vReal freq_scaled, vReal& eta, vReal& chi);


inline int get_terms_and_chi (FREQUENCIES& frequencies, const TEMPERATURE& temperature,
	                            LINES& lines, const SCATTERING& scattering, RADIATION& radiation,
						                  const long f, const long o, const long R,
								              long& lnotch, long& notch, const long cellNrs, const double shifts,
	                            vReal& term1, vReal& term2, vReal& chi);


template <int Dimension, long Nrays>
inline int get_terms_chi_and_Ibdy (const CELLS<Dimension, Nrays>& cells, FREQUENCIES& frequencies,
	                                 const TEMPERATURE& temperature, LINES& lines,
																	 const SCATTERING& scattering, RADIATION& radiation,
						                       const long f, const long o, const long R,
								                   long& lnotch, long& notch, const long cellNrs, const double shifts,
	                                 vReal& term1, vReal& term2, vReal& chi, vReal& Ibdy_scaled);

#include "set_up_ray.tpp"


#endif // __SET_UP_RAY_HPP_INCLUDED__
