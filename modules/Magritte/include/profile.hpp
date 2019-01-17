// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __PROFILE_HPP_INCLUDED__
#define __PROFILE_HPP_INCLUDED__


#include "GridTypes.hpp"




///  profile: line profile function
///    @param[in] width: profile width
///    @param[in] freq_diff: frequency at which we want evaluate the profile
///    @return profile function evaluated at frequency freq
////////////////////////////////////////////////////////////////////////////

inline vReal profile      (
    const double width,
    const vReal  freq_diff);




///  profile_width: line profile width due to thermal and turbulent Doppler shifts
///    @param[in] temperature_gas: temperature of the gas at this cell
///    @param[in] freq_line: frequency of the line under consideration
///    @return width of the correpsonding line profile
//////////////////////////////////////////////////////////////////////////////////

inline double profile_width      (
    const double temperature_gas,
    const double vturb2,
    const double freq_line       );



///  profile_width: line profile width due to thermal and turbulent Doppler shifts
///    @param[in] temperature_gas: temperature of the gas at this cell
///    @return width of the correpsonding line profile
//////////////////////////////////////////////////////////////////////////////////

inline double profile_width      (
    const double temperature_gas,
    const double vturb2          );




///  Planck: Planck energy distribution function (in freqiencies)
///    @param[in] temperature_gas: temperature of the gad at this cell
///    @param[in] freq: frequency at which we want evaluate the Planck function
///////////////////////////////////////////////////////////////////////////////

inline vReal planck (
    const double temperature_gas,
    const vReal  freq);




///  vExp: exponential function for vReal types
///  !!! Only good for positive exponents !!!
///    @param[in] x: exponent
///    @return exponential of x
/////////////////////////////////////////////////

inline vReal vExp (
    const vReal   );




///  vExpMinus: exponential function for vReal types
///    @param[in] x: exponent
///    @return exponential of minus x
/////////////////////////////////////////////////

inline vReal vExpMinus (
    const vReal x      );




///  vExpm1: exponential minus 1.0 function for vReal types
///    @param[in] x: exponent
///    @return exponential minus 1.0 of x
///////////////////////////////////////////////////////////

inline vReal vExpm1 (
    const vReal x   );


#include "../src/profile.tpp"


#endif // __PROFILE_HPP_INCLUDED__
