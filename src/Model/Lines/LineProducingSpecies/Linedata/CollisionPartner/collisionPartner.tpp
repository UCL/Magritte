// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Functions/interpolation.hpp"


///  interpolate_collision_coefficients:
///    @param[in] temperature_gas: local gas temperature
////////////////////////////////////////////////////////

inline void CollisionPartner ::
    interpolate_collision_coefficients (
        const double temperature_gas   )
{

    const int t = search (tmp, temperature_gas);


    if      (t == 0)
    {
      Ce_intpld = Ce[0];
      Cd_intpld = Cd[0];
    }

    else if (t == ntmp)
    {
      Ce_intpld = Ce[ntmp-1];
      Cd_intpld = Cd[ntmp-1];
    }

    else
    {
      const double step = (temperature_gas - tmp[t-1]) / (tmp[t] - tmp[t-1]);

      for (long k = 0; k < ncol; k++)
      {
        Ce_intpld[k] = Ce[t-1][k] + (Ce[t][k] - Ce[t-1][k]) * step;
        Cd_intpld[k] = Cd[t-1][k] + (Cd[t][k] - Cd[t-1][k]) * step;
      }
    }


}


inline void CollisionPartner ::
    adjust_abundance_for_ortho_or_para (
        const double  temperature_gas,
              double &abundance        ) const
{

  if (orth_or_para_H2 != "n")
  {
    const double frac_H2_para = 1.0 / (1.0 + 9.0*exp (-170.5/temperature_gas));

    if (orth_or_para_H2 == "o")
    {
      abundance *= (1.0 - frac_H2_para);
    }

    if (orth_or_para_H2 == "p")
    {
      abundance *= frac_H2_para;
    }
  }


}
