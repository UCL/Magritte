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
        const double          freq_line,
        const int             lane           ) const
{
  const vReal profile = thermodynamics.profile (nrs[n_ar], frs[n_ar], freq_line);

  return getlane (frs[n_ar] * profile * L_diag[n_ar] / chi[n_ar], lane);
}




inline double RayPair ::
    get_L_lower (
        const Thermodynamics &thermodynamics,
        const double          freq_line,
        const int             lane,
        const long            m              ) const
{
  const vReal profile = thermodynamics.profile (nrs[n_ar-m], frs[n_ar-m], freq_line);

  return getlane (frs[n_ar-m] * profile * L_lower[m][n_ar-m] / chi[n_ar-m], lane);
}




inline double RayPair ::
    get_L_upper (
        const Thermodynamics &thermodynamics,
        const double          freq_line,
        const int             lane,
        const long            m              ) const
{
  const vReal profile = thermodynamics.profile (nrs[n_ar+m], frs[n_ar+m], freq_line);

  return getlane (frs[n_ar+m] * profile * L_upper[m][n_ar+m] / chi[n_ar+m], lane);
}




inline void RayPair ::
    update_Lambda (
        const Frequencies                       &frequencies,
        const Thermodynamics                    &thermodynamics,
        const long                               p,
        const long                               f,
        const double                             weight_angular,
              std::vector<LineProducingSpecies> &lineProducingSpecies   ) const
{


  GRID_FOR_ALL_LANES (lane)
  {
    const long f_index = f * n_simd_lanes + lane;

    if (frequencies.appears_in_line_integral[f_index])
    {
      const long l = frequencies.corresponding_l_for_spec[f_index];
      const long k = frequencies.corresponding_k_for_tran[f_index];
      const long z = frequencies.corresponding_z_for_line[f_index];

      const double freq_line = lineProducingSpecies[l].linedata.frequency[k];
      const double weight    = lineProducingSpecies[l].quadrature.weights[z] * 2.0 * weight_angular;
      const double factor    = lineProducingSpecies[l].linedata.A[k] * weight;


      double L = factor * get_L_diag (thermodynamics, freq_line, lane);

      const long i   = lineProducingSpecies[l].linedata.irad[k];
      const long ind = lineProducingSpecies[l].index (nrs[n_ar], i);

      lineProducingSpecies[l].lambda[p][k].add_entry (L, nrs[n_ar]);


      for (long m = 1; m < n_off_diag; m++)
      {
        if (n_ar-m >= 0)
        {
          L = factor * get_L_lower (thermodynamics, freq_line, lane, m);

          lineProducingSpecies[l].lambda[p][k].add_entry (L, nrs[n_ar-m]);
        }

        if (n_ar+m < ndep-m)
        {
          L = factor * get_L_upper (thermodynamics, freq_line, lane, m);

          lineProducingSpecies[l].lambda[p][k].add_entry (L, nrs[n_ar+m]);
        }
      }
     }
  }


}
