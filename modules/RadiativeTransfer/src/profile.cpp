// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>

#include "profile.hpp"
#include "constants.hpp"


///  profile: line profile function
///    @param[in] temperature_gas: temperature of the gas at this cell
///    @param[in] freq_line: frequency of the line under consideration
///    @param[in] freq: frequency at which we  want evaluate the profile
///    @return profile function evaluated at frequency freq
////////////////////////////////////////////////////////////////////////

double profile (const double temperature_gas, const double freq_line, const double freq)
{
	const double sqrtPi  = sqrt(PI);    // square root of Pi

  const double width = profile_width (temperature_gas, freq_line);

  return exp( -pow((freq - freq_line)/width, 2) ) / (sqrt(PI) * width);
}




///  profile_width: line profile width due to thermal and turbulent Doppler shifts
///    @param[in] temperature_gas: temperature of the gas at this cell
///    @param[in] freq_line: frequency of the line under consideration
///    @return width of the correpsonding line profile
//////////////////////////////////////////////////////////////////////////////////

double profile_width (const double temperature_gas, const double freq_line)
{
	const double v_turb2 = 100.0;       // turbulent speed squared
	const double factor  = 2.0*KB/MP;   // 2.0*Kb/Mp (with Boltzmann constant and proton mass)

  return freq_line/CC * sqrt(factor*temperature_gas + v_turb2);
}
