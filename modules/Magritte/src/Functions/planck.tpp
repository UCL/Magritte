// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Tools/constants.hpp"


///  Planck: Planck energy distribution function (in freqiencies)
///    @param[in] temperature_gas: temperature of the gad at this cell
///    @param[in] freq: frequency at which we want evaluate the Planck function
///////////////////////////////////////////////////////////////////////////////

inline vReal planck (
    const double temperature,
    const vReal  freq        )
{

  const double h_over_kbT = HH_OVER_KB / temperature;

  return TWO_HH_OVER_CC_SQUARED * (freq*freq*freq) / vExpm1(h_over_kbT*freq);

}
