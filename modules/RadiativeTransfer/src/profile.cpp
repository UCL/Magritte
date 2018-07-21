// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>

#include "profile.hpp"
#include "GridTypes.hpp"
#include "constants.hpp"


///  profile: line profile function
///    @param[in] temperature_gas: temperature of the gas at this cell
///    @param[in] freq_line: frequency of the line under consideration
///    @param[in] freq: frequency at which we want evaluate the profile
///    @return profile function evaluated at frequency freq
////////////////////////////////////////////////////////////////////////

vReal profile (const double temperature_gas, const double freq_line, const vReal freq)
{

  const double inverse_width = inverse_profile_width (temperature_gas, freq_line);

	const vReal vFreq_line   = freq_line;
	const vReal sqrtExponent = inverse_width * (freq - vFreq_line);
	const vReal exponent     = - sqrtExponent * sqrtExponent;


	return inverse_width * INVERSE_SQRT_PI * vExp (exponent);

}




///  profile: line profile function
///    @param[in] inverse_width: inverse profile width
///    @param[in] freq_diff: frequency at which we want evaluate the profile
///    @return profile function evaluated at frequency freq
////////////////////////////////////////////////////////////////////////////

vReal profile (const double inverse_width, const vReal freq_diff)
{

	const vReal sqrtExponent = inverse_width * freq_diff;
	const vReal exponent     = - sqrtExponent * sqrtExponent;


	return inverse_width * INVERSE_SQRT_PI * vExp (exponent);

}




///  profile_width: line profile width due to thermal and turbulent Doppler shifts
///    @param[in] temperature_gas: temperature of the gas at this cell
///    @param[in] freq_line: frequency of the line under consideration
///    @return width of the correpsonding line profile
//////////////////////////////////////////////////////////////////////////////////

double profile_width (const double temperature_gas, const double freq_line)
{
  return freq_line * sqrt(TWO_KB_OVER_MP_C_SQUARED*temperature_gas
			                    + V_TURB_OVER_C_ALL_SQUARED);
}




///  inverse_profile_width: one over the line profile width
///    @param[in] temperature_gas: temperature of the gas at this cell
///    @param[in] freq_line: frequency of the line under consideration
///    @return width of the correpsonding line profile
//////////////////////////////////////////////////////////////////////////////////

double inverse_profile_width (const double temperature_gas, const double freq_line)
{
  return 1.0 / profile_width (temperature_gas, freq_line);
}




///  Planck: Planck energy distribution function (in freqiencies)
///    @param[in] temperature_gas: temperature of the gad at this cell
///    @param[in] freq: frequency at which we want evaluate the Planck function
///////////////////////////////////////////////////////////////////////////////

vReal Planck (const double temperature_gas, const vReal freq)
{
	return 2.0 * HH * freq*freq*freq
		     / (CC*CC*vExpm1( HH*freq / (KB*temperature_gas)));
}




///  vExp: exponential function for vReal types
///    @param[in] x: exponent
///    @return exponential of x
/////////////////////////////////////////////////

vReal vExp (const vReal x)
{
	const int n = 9;

  vReal result = 1.0;

  for (int i = n; i > 0; i--)
	{
		const double factor = 1.0 / i;
		const vReal vFactor = factor;

    result = vOne + x*result*vFactor;
	}


	return result;

}


///  vExpm1: exponential minus 1.0 function for vReal types
///    @param[in] x: exponent
///    @return exponential minus 1.0 of x
///////////////////////////////////////////////////////////

vReal vExpm1 (const vReal x)
{
	const int n = 9;

  vReal result = 1.0;

  for (int i = n; i > 1; i--)
	{
		const double factor = 1.0 / i;
		const vReal vFactor = factor;

    result = vOne + x*result*vFactor;
	}

  result = x*result;


	return result;

}
