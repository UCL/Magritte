// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __PROFILE_HPP_INCLUDED__
#define __PROFILE_HPP_INCLUDED__


///  profile: line profile function
///    @param[in] temperature_gas: temperature of the gas at this cell
///    @param[in] freq_line: frequency of the line under consideration
///    @param[in] freq: frequency at which we  want evaluate the profile
///    @return profile function evaluated at frequency freq
////////////////////////////////////////////////////////////////////////

double profile (double temperature_gas, double freq_line, double freq);




///  profile_width: line profile width due to thermal and turbulent Doppler shifts
///    @param[in] temperature_gas: temperature of the gas at this cell
///    @param[in] freq_line: frequency of the line under consideration
///    @return width of the correpsonding line profile
//////////////////////////////////////////////////////////////////////////////////

double profile_width (double temperature_gas, double freq_line);


#endif // __PROFILE_HPP_INCLUDED__
