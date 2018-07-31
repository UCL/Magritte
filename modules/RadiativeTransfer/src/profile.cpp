// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "profile.hpp"
#include "GridTypes.hpp"
#include "constants.hpp"


///  profile: line profile function
///    @param[in] inverse_width: inverse profile width
///    @param[in] freq_diff: frequency at which we want evaluate the profile
///    @return profile function evaluated at frequency freq
////////////////////////////////////////////////////////////////////////////

inline vReal profile (const double width, const vReal freq_diff)
{

  const double inverse_width = 1.0 / width;
	const vReal  sqrtExponent  = inverse_width * freq_diff;
	const vReal  exponent      = sqrtExponent * sqrtExponent;


	return inverse_width * INVERSE_SQRT_PI * vExpMinus (exponent);

}




///  profile_width: line profile width due to thermal and turbulent Doppler shifts
///    @param[in] temperature_gas: temperature of the gas at this cell
///    @param[in] freq_line: frequency of the line under consideration
///    @return width of the correpsonding line profile
//////////////////////////////////////////////////////////////////////////////////

inline double profile_width (const double temperature_gas, const double freq_line)
{
  return freq_line * sqrt (TWO_KB_OVER_MP_C_SQUARED * temperature_gas
			                     + V_TURB_OVER_C_ALL_SQUARED);
}




///  Planck: Planck energy distribution function (in freqiencies)
///    @param[in] temperature_gas: temperature of the gad at this cell
///    @param[in] freq: frequency at which we want evaluate the Planck function
///////////////////////////////////////////////////////////////////////////////

inline vReal Planck (const double temperature_gas, const vReal freq)
{
  const double h_over_kbT = HH_OVER_KB / temperature_gas;

	return TWO_HH_OVER_CC_SQUARED * (freq*freq*freq) * vExpm1(h_over_kbT*freq);
}




///  vExp: exponential function for vReal types
///  !!! Only good for positive exponents !!!
///    @param[in] x: exponent
///    @return exponential of x
/////////////////////////////////////////////////

inline vReal vExp (const vReal x)
{

	const int n = 23;

  vReal result = 1.0;

  for (int i = n; i > 1; i--)
	{
		const double factor = 1.0 / i;
		const vReal vFactor = factor;

    result = vOne + x*result*vFactor;
	}

  result = vOne + x*result;


	return result;

}




///  vExpMinus: exponential function for vReal types
///    @param[in] x: exponent
///    @return exponential of minus x
/////////////////////////////////////////////////

inline vReal vExpMinus (const vReal x)
{
	return 1.0 / vExp (x);
}




///  vExpm1: exponential minus 1.0 function for vReal types
///    @param[in] x: exponent
///    @return exponential minus 1.0 of x
///////////////////////////////////////////////////////////

inline vReal vExpm1 (const vReal x)
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
