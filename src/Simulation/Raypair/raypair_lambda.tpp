// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Tools/debug.hpp"
#include "Model/Radiation/Frequencies/frequencies.hpp"
#include "Model/Lines/LineProducingSpecies/lineProducingSpecies.hpp"


///  Getter for the diagonal entries of the Lambda operator
///    @param[in]     thermodynamics : thermodynamics object of the model
///    @param[in]     inverse_mass   : one over the mass of line producing species
///    @param[in]     freq_line      : frequency of the line
///    @param[in]     lane           : lane of the vector element
//////////////////////////////////////////////////////////////////////////////////

inline double RayPair ::
    get_L_diag (
        const Thermodynamics &thermodynamics,
        const double          inverse_mass,
        const double          freq_line,
        const int             lane           ) const
{
  const vReal profile = thermodynamics.profile (inverse_mass, nrs[n_ar],
                                                freq_line, frs[n_ar]    );

  return getlane (frs[n_ar] * profile * L_diag[n_ar] / chi[n_ar], lane);
}




///  Getter for the lower diagonal entries of the Lambda operator
///    @param[in]     thermodynamics : thermodynamics object of the model
///    @param[in]     inverse_mass   : one over the mass of line producing species
///    @param[in]     freq_line      : frequency of the line
///    @param[in]     lane           : lane of the vector element
///    @param[in]     m              : index for position above the diagonal
//////////////////////////////////////////////////////////////////////////////////

inline double RayPair ::
    get_L_lower (
        const Thermodynamics &thermodynamics,
        const double          inverse_mass,
        const double          freq_line,
        const int             lane,
        const long            m              ) const
{
  const vReal profile = thermodynamics.profile (inverse_mass, nrs[n_ar-m],
                                                freq_line, frs[n_ar-m]    );

  return getlane (frs[n_ar-m] * profile * L_lower[m][n_ar-m] / chi[n_ar-m], lane);
}




///  Getter for the upper diagonal entries of the Lambda operator
///    @param[in]     thermodynamics : thermodynamics object of the model
///    @param[in]     inverse_mass   : one over the mass of line producing species
///    @param[in]     freq_line      : frequency of the line
///    @param[in]     lane           : lane of the vector element
///    @param[in]     m              : index for position above the diagonal
//////////////////////////////////////////////////////////////////////////////////

inline double RayPair ::
    get_L_upper (
        const Thermodynamics &thermodynamics,
        const double          inverse_mass,
        const double          freq_line,
        const int             lane,
        const long            m              ) const
{
  const vReal profile = thermodynamics.profile (inverse_mass, nrs[n_ar+m],
                                                freq_line, frs[n_ar+m]    );

  return getlane (frs[n_ar+m] * profile * L_upper[m][n_ar+m] / chi[n_ar+m], lane);
}




///  Updater for the Lambda operators of the line producing species
///    @param[in]     frequencies    : frequencies object of the model
///    @param[in]     thermodynamics : thermodynamics object of the model
///    @param[in]     p              : index of the cell
///    @param[in]     f              : index of the frequency
///    @param[in]     weight_angular : angular weight for this entry of the Lambda operator
///    @param[in/out] lines          : lines object of the model
///////////////////////////////////////////////////////////////////////////////////////////

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

//      const long i   = lines.lineProducingSpecies[l].linedata.irad[k];
//      const long ind = lines.lineProducingSpecies[l].index (nrs[n_ar], i);

      lines.lineProducingSpecies[l].lambda.add_element (p, k, nrs[n_ar], L);


      for (long m = 1; m < n_off_diag; m++)
      {
        if (n_ar-m >= 0)
        {
          L = factor * get_L_lower (thermodynamics, invr_mass, freq_line, lane, m);

          lines.lineProducingSpecies[l].lambda.add_element (p, k, nrs[n_ar-m], L);
        }

        if (n_ar+m < ndep-m)
        {
          L = factor * get_L_upper (thermodynamics, invr_mass, freq_line, lane, m);

          lines.lineProducingSpecies[l].lambda.add_element (p, k, nrs[n_ar+m], L);
        }
      }
    }
  }


}
