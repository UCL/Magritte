// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Tools/debug.hpp"
#include "Model/Radiation/Frequencies/frequencies.hpp"
#include "Model/Lines/LineProducingSpecies/lineProducingSpecies.hpp"


///  initialize:
///////////////////////////////////////////////////////

inline void RayPair ::
    initialize (
        const long n_ar_local,
        const long n_r_local  )
{

  n_ar = n_ar_local;
  n_r  = n_r_local;

  ndep = n_ar + n_r + 1;

  lnotch_at_origin = 0;


  if (ndep > term1.size())
  {
     term1.resize (ndep+10);
     term2.resize (ndep+10);
         A.resize (ndep+10);
         C.resize (ndep+10);
         F.resize (ndep+10);
         G.resize (ndep+10);
        Su.resize (ndep+10);
        Sv.resize (ndep+10);
      dtau.resize (ndep+10);
    L_diag.resize (ndep+10);
       chi.resize (ndep+10);
       nrs.resize (ndep+10);
       frs.resize (ndep+10);

    if (n_off_diag > 0)
    {
      L_upper.resize (n_off_diag);
      L_lower.resize (n_off_diag);

      for (long m = 0; m < n_off_diag; m++)
      {
        L_upper[m].resize (ndep+10);
        L_lower[m].resize (ndep+10);
      }
    }

  }


}




inline void RayPair ::
    set_term1_and_term2 (
        const vReal &eta,
        const vReal &chi,
        const vReal &U_scaled,
        const vReal &V_scaled,
        const long   n        )
{

  const vReal inverse_chi = 1.0 / chi;

  term1[n] = (U_scaled*0.0 + eta) * inverse_chi;
  term2[n] =  V_scaled*0.0        * inverse_chi;


}





inline void RayPair ::
    set_dtau (
        const vReal  &chi,
        const vReal  &chi_prev,
        const double  dZ,
        const long    n        )
{

  dtau[n] = 0.5 * (chi + chi_prev) * dZ;

}




inline vReal RayPair ::
    get_u_at_origin () const
{
  return Su[n_ar];
}




inline vReal RayPair ::
    get_v_at_origin () const
{
  return Sv[n_ar];
}




inline vReal RayPair ::
    get_I_p () const
{
  return Su[ndep-1] + Sv[ndep-1];
}




inline vReal RayPair ::
    get_I_m () const
{
  return Su[0] - Sv[0];
}
