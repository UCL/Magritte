// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Tools/constants.hpp"


///  profile: line profile function
///    @param[in] inverse_width: inverse profile width
///    @param[in] freq_diff: frequency at which we want evaluate the profile
///    @return profile function evaluated at frequency freq
////////////////////////////////////////////////////////////////////////////

inline vReal Thermodynamics ::
    profile (
        const double width,
        const vReal  freq_diff) const
{

  const double inverse_width = 1.0 / width;
  const vReal  sqrtExponent  = inverse_width * freq_diff;
  const vReal  exponent      = sqrtExponent * sqrtExponent;


  return inverse_width * INVERSE_SQRT_PI * vExpMinus (exponent);

}


inline vReal Thermodynamics ::
    profile (
        const double inverse_mass,
        const long   p,
        const double freq_line,
        const vReal  freq          ) const
{
  return profile (profile_width (inverse_mass, p, freq_line),
                  freq - (vReal) freq_line                   );
}




///  profile_width: line profile width due to thermal and turbulent Doppler shifts
///    @param[in] temperature_gas: temperature of the gas at this cell
///    @param[in] freq_line: frequency of the line under consideration
///    @return width of the correpsonding line profile
//////////////////////////////////////////////////////////////////////////////////

inline double Thermodynamics ::
    profile_width (
        const double inverse_mass,
        const long   p,
        const double freq_line    ) const
{
  return freq_line * profile_width (inverse_mass, p);
}


inline double Thermodynamics ::
    profile_width (
        const double inverse_mass,
        const long   p            ) const
{
  return sqrt (TWO_KB_OVER_AMU_C_SQUARED * inverse_mass * temperature.gas[p]
               + turbulence.vturb2[p]);
}
