// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __PROFILE_HPP_INCLUDED__
#define __PROFILE_HPP_INCLUDED__


#include "GridTypes.hpp"


///  profile: line profile function
///    @param[in] temperature_gas: temperature of the gas at this cell
///    @param[in] freq_line: frequency of the line under consideration
///    @param[in] freq: frequency at which we  want evaluate the profile
///    @return profile function evaluated at frequency freq
////////////////////////////////////////////////////////////////////////

vReal profile (const double temperature_gas, const double freq_line, const vReal freq);




///  profile_width: line profile width due to thermal and turbulent Doppler shifts
///    @param[in] temperature_gas: temperature of the gas at this cell
///    @param[in] freq_line: frequency of the line under consideration
///    @return width of the correpsonding line profile
//////////////////////////////////////////////////////////////////////////////////

double profile_width (const double temperature_gas, const double freq_line);




double inverse_profile_width (const double temperature_gas, const double freq_line);


///  Planck: Planck energy distribution function (in freqiencies)
///    @param[in] temperature_gas: temperature of the gad at this cell
///    @param[in] freq: frequency at which we want evaluate the Planck function
///////////////////////////////////////////////////////////////////////////////

vReal Planck (const double temperature_gas, const vReal freq);


vReal vExp (const vReal);


#endif // __PROFILE_HPP_INCLUDED__
