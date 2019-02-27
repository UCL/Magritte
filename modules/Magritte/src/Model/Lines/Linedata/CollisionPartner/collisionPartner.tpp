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
