// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Tools/debug.hpp"
#include "Model/Radiation/Frequencies/frequencies.hpp"
#include "Model/Lines/LineProducingSpecies/lineProducingSpecies.hpp"


///  Initializer for ray pair that allocates the memory
///    @param[in] n_ar_local : number of cells on ray ar
///    @param[in] n_r_local  : number of cells on ray r
////////////////////////////////////////////////////////////

inline void RayPair ::
    initialize (
        const long n_ar_local,
        const long n_r_local  )
{

  n_ar = n_ar_local;
  n_r  = n_r_local;

  ndep = n_ar + n_r + 1;

  first = 0;
  last  = ndep-1;

  lnotch_at_origin = 0;


  //if (ndep > term1.size())
  //{
  //   term1.resize (ndep+10);
  //   term2.resize (ndep+10);
  //       A.resize (ndep+10);
  //       C.resize (ndep+10);
  //       F.resize (ndep+10);
  //       G.resize (ndep+10);
  //      Su.resize (ndep+10);
  //      Sv.resize (ndep+10);
  //    dtau.resize (ndep+10);
  //  L_diag.resize (ndep+10);
  //     chi.resize (ndep+10);
  //     nrs.resize (ndep+10);
  //     frs.resize (ndep+10);

  //     inverse_one_plus_F.resize (ndep+10);
  //     inverse_one_plus_G.resize (ndep+10);
  //      G_over_one_plus_G.resize (ndep+10);
  //              inverse_A.resize (ndep+10);
  //              inverse_C.resize (ndep+10);


  //  if (n_off_diag > 0)
  //  {
  //    L_upper.resize (n_off_diag);
  //    L_lower.resize (n_off_diag);

  //    for (long m = 0; m < n_off_diag; m++)
  //    {
  //      L_upper[m].resize (ndep+10);
  //      L_lower[m].resize (ndep+10);
  //    }
  //  }

  //}


}




///  Setter for Feautrier term 1 and term 2 (considering scattering)
///    @param[in] eta      : emissivity
///    @param[in] chi      : opacity
///    @param[in] U_scaled : scattering U in co-moving frame
///    @param[in] V_scaled : scattering V in co-moving frame
///    @param[in] n        : current index on the ray pair
////////////////////////////////////////////////////////////////////

inline void RayPair ::
    set_term1_and_term2 (
        const vReal &eta,
        const vReal &chi,
        const vReal &U_scaled,
        const vReal &V_scaled,
        const long   n        )
{

  const vReal inverse_chi = 1.0 / chi;

  term1[n] = (U_scaled + eta) * inverse_chi;
  term2[n] =  V_scaled        * inverse_chi;


}




///  Setter for Feautrier term 1 (if no scattering is considered)
///    @param[in] eta      : emissivity
///    @param[in] chi      : opacity
///    @param[in] n        : current index on the ray pair
/////////////////////////////////////////////////////////////////

inline void RayPair ::
    set_term1_and_term2 (
        const vReal &eta,
        const vReal &chi,
        const long   n        )
{

  term1[n] = eta / chi;

}




///  Setter for the optical depth increment (dtau)
///    @param[in] chi      : opacity in this cell
///    @param[in] chi_prev : opacity in previous cell
///    @param[in] dZ       : distance increment between this and previous cell
///    @param[in] n        : current index on the ray pair
//////////////////////////////////////////////////////////////////////////////

inline void RayPair ::
    set_dtau (
        const vReal  &chi,
        const vReal  &chi_prev,
        const double  dZ,
        const long    n        )
{
  dtau[n] = 0.5 * (chi + chi_prev) * dZ;
}




///  Getter for u at the origin
///////////////////////////////

inline vReal RayPair ::
    get_u_at_origin () const
{
  return Su[n_ar];
}




///  Getter for v at the origin
///////////////////////////////

inline vReal RayPair ::
    get_v_at_origin () const
{
  // if      (n_ar == 0)
  // {
  //   return 0;
  // }
  // else if (n_ar == ndep-1)
  // {
  //   return 0;
  // }
  // else
  // {
  //   return (Sv[n_ar+1] - Sv[n_ar]) / dtau[n_ar] ;
  // }

  return Sv[n_ar];
}




///  Getter for the intensity up the ray (I^{+}) at the origin
///    @return Intensity up the ray (I^{+}) at the origin
//////////////////////////////////////////////////////////////

inline vReal RayPair ::
    get_Ip_at_origin () const
{
  return Su[n_ar] + Sv[n_ar];
}




///  Getter for the intensity down the ray (I^{-}) at the origin
///    @return Intensity down the ray (I^{-}) at the origin
////////////////////////////////////////////////////////////////

inline vReal RayPair ::
    get_Im_at_origin () const
{
  return Su[n_ar] - Sv[n_ar];
}




///  Getter for the intensity up the ray (I^{+}) at the end of the ray
///    @return Intensity up the ray (I^{+}) at the end of the ray
//////////////////////////////////////////////////////////////////////

inline vReal RayPair ::
    get_Ip_at_end () const
{
  return Su[ndep-1] - I_bdy_n;
}




///  Getter for the intensity down the ray (I^{-}) at the front of the ray
///    @return Intensity down the ray (I^{-}) at the front of the ray
//////////////////////////////////////////////////////////////////////////

inline vReal RayPair ::
    get_Im_at_front () const
{
  return Su[0] - I_bdy_0;
}



// PROBLEM: The code below code has to problem that the intensity s in the wrong frequency bin i.e. co-moving vs rest frame

//inline int RayPair ::
//    update_radiation (
//        const long      R,
//        const long      f,
//        const Long1     ntimes_encounterd,
//        const double    weight_angular,
//              Radiation &radiation        )
//{
//
//  for (long n = 0; n < ndep; n++)
//  {
//    const long p = nrs[n];
//
//    if (ntimes_encounterd[p] == 0)
//    {
//      const long ind = radiation.index (p,f);
//
//      radiation.J[ind] += 2.0 * weight_angular * Su[n];
//    }
//  }
//
//
//  if (radiation.use_scattering)
//  {
//    for (long n = 0; n < ndep; n++)
//    {
//      const long p = nrs[n];
//
//      if (ntimes_encounterd[p] == 0)
//      {
//        const long ind = radiation.index (p,f);
//
//        radiation.u[R][ind] = Su[n];
//        radiation.v[R][ind] = Sv[n];
//      }
//    }
//  }
//
//
//  return (0);
//
//}
