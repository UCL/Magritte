// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Tools/debug.hpp"
#include "Model/Radiation/Frequencies/frequencies.hpp"
#include "Model/Lines/LineProducingSpecies/lineProducingSpecies.hpp"


inline double RayPair ::
    get_L_diag (
        const Thermodynamics &thermodynamics,
        const double          inverse_mass,
        const double          freq_line,
        const int             lane           ) const
{
  const vReal profile = thermodynamics.profile (inverse_mass, nrs[n_ar],
                                                frs[n_ar], freq_line    );

  return getlane (frs[n_ar] * profile * L_diag[n_ar] / chi[n_ar], lane);
}




inline double RayPair ::
    get_L_lower (
        const Thermodynamics &thermodynamics,
        const double          inverse_mass,
        const double          freq_line,
        const int             lane,
        const long            m              ) const
{
  const vReal profile = thermodynamics.profile (inverse_mass, nrs[n_ar-m],
                                                frs[n_ar-m], freq_line    );

  return getlane (frs[n_ar-m] * profile * L_lower[m][n_ar-m] / chi[n_ar-m], lane);
}




inline double RayPair ::
    get_L_upper (
        const Thermodynamics &thermodynamics,
        const double          inverse_mass,
        const double          freq_line,
        const int             lane,
        const long            m              ) const
{
  const vReal profile = thermodynamics.profile (inverse_mass, nrs[n_ar+m],
                                                frs[n_ar+m], freq_line    );

  return getlane (frs[n_ar+m] * profile * L_upper[m][n_ar+m] / chi[n_ar+m], lane);
}




inline void RayPair ::
    update_Lambda (
        const Frequencies    &frequencies,
        const Thermodynamics &thermodynamics,
        const long            p,
        const long            f,
        const double          weight_angular,
              Lines          &lines          ) const
{


  GRID_FOR_ALL_LANES (lane)
  {
    const long f_index = f * n_simd_lanes + lane;

    if (frequencies.appears_in_line_integral[f_index])
    {
      const long l = frequencies.corresponding_l_for_spec[f_index];
      const long k = frequencies.corresponding_k_for_tran[f_index];
      const long z = frequencies.corresponding_z_for_line[f_index];

      const double freq_line = lines.lineProducingSpecies[l].linedata.frequency[k];
      const double invr_mass = lines.lineProducingSpecies[l].linedata.inverse_mass;
      const double weight    = lines.lineProducingSpecies[l].quadrature.weights[z] * 2.0 * weight_angular;
      const double factor    = lines.lineProducingSpecies[l].linedata.A[k] * weight;


      double L = factor * get_L_diag (thermodynamics, invr_mass, freq_line, lane);

      const long i   = lines.lineProducingSpecies[l].linedata.irad[k];
      const long ind = lines.lineProducingSpecies[l].index (nrs[n_ar], i);

      lines.lineProducingSpecies[l].lambda[p][k].add_entry (L, nrs[n_ar]);


      for (long m = 1; m < n_off_diag; m++)
      {
        if (n_ar-m >= 0)
        {
          L = factor * get_L_lower (thermodynamics, invr_mass, freq_line, lane, m);

          lines.lineProducingSpecies[l].lambda[p][k].add_entry (L, nrs[n_ar-m]);
        }

        if (n_ar+m < ndep-m)
        {
          L = factor * get_L_upper (thermodynamics, invr_mass, freq_line, lane, m);

          lines.lineProducingSpecies[l].lambda[p][k].add_entry (L, nrs[n_ar+m]);
        }
      }
     }
  }


}
