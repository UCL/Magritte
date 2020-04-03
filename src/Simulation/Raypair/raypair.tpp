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
  return 2.0*Su[ndep-1] - I_bdy_n;
}




///  Getter for the intensity down the ray (I^{-}) at the front of the ray
///    @return Intensity down the ray (I^{-}) at the front of the ray
//////////////////////////////////////////////////////////////////////////

inline vReal RayPair ::
    get_Im_at_front () const
{
  return 2.0*Su[0] - I_bdy_0;
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





//inline void RayPair :: setFrequencies (
//    const Double1 &frequencies,
//    const double   scale,
//    const size_t   index              )
//{
//    size_t f1 = 0;
//
//    for (size_t f2 = 0; f2 < nfreqs; f2++)
//    {
//        const size_t if2 = I(index,f2);
//
//        freqs       [if2] = frequencies[f2];
//        freqs_scaled[if2] = frequencies[f2] * scale;
//
//        while (   (frequencies[f1] < freqs_scaled[if2])
//               && (f1              < nfreqs-1         ) )
//        {
//            f1++;
//        }
//
//        if (f1 > 0)
//        {
//            freqs_lower[if2] = f1-1;
//            freqs_upper[if2] = f1;
//        }
//        else
//        {
//            freqs_lower[if2] = 0;
//            freqs_upper[if2] = 1;
//        }
//    }
//}
//
//
//
//
//inline void RayPair :: setup (
//    const Model &model,
//    const RayData &raydata1,
//    const RayData &raydata2,
//    const size_t   R,
//    const size_t   o         )
//{
//    /// For interchanging raydata
//    RayData raydata_ar;
//    RayData raydata_rr;
//
//    /// Ensure that ray ar is longer than ray rr
//    if (raydata2.size() > raydata1.size())
//    {
//        reverse    =     -1.0;
//        raydata_ar = raydata2;
//        raydata_rr = raydata1;
//    }
//    else
//    {
//        reverse    =     +1.0;
//        raydata_ar = raydata1;
//        raydata_rr = raydata2;
//    }
//
//    /// Extract ray and antipodal ray lengths
//    n_ar = raydata_ar.size();
//    n_rr = raydata_rr.size();
//
//    /// Set total number of depth points
//    ndep = n_ar + n_rr + 1;
//
//    /// Initialize index
//    size_t index = n_ar;
//
//    /// Temporarily set first and last
//    first = 0;
//    last  = n_ar;
//
//    /// Set origin
//    setFrequencies (model.radiation.frequencies.nu[o], 1.0, index);
//    nrs[index] = o;
//
//    /// Temporary boundary numbers
//    size_t bdy_0 = model.geometry.boundary.cell2boundary_nr[o];
//    size_t bdy_n = model.geometry.boundary.cell2boundary_nr[o];
//
//    /// Set ray ar
//    if (n_ar > 0)
//    {
//        index = n_ar-1;
//
//        for (const ProjectedCellData &data : raydata_ar)
//        {
//            setFrequencies (model.radiation.frequencies.nu[o], data.shift, index);
//            nrs[index] = data.cellNr;
//            dZs[index] = data.dZ;
//            index--;
//        }
//
//        bdy_0 = model.geometry.boundary.cell2boundary_nr[raydata_ar.back().cellNr];
//        first = index+1;
//    }
//
//    /// Set ray rr
//    if (n_rr > 0)
//    {
//        index = n_ar+1;
//
//        for (const ProjectedCellData &data : raydata_rr)
//        {
//            setFrequencies (model.radiation.frequencies.nu[o], data.shift, index);
//            nrs[index  ] = data.cellNr;
//            dZs[index-1] = data.dZ;
//            index++;
//        }
//
//        bdy_n = model.geometry.boundary.cell2boundary_nr[raydata_rr.back().cellNr];
//        last  = index-1;
//    }
//}
//
//
//
//
//inline void RayPair :: store (
//    Model        &model,
//    const size_t  R,
//    const size_t  r,
//    const size_t  o          )
//{
//    const double weight_ang = 2.0 * model.geometry.rays.weights[r];
//
//    const size_t i0 = model.radiation.index(o,0);
//    const size_t j0 = I(n_ar,0);
//
//    for (size_t f = 0; f < nfreqs; f++)
//    {
//        model.radiation.J[i0+f] += weight_ang * Su[j0+f];
//    }
//
//    if (model.parameters.use_scattering())
//    {
//        for (size_t f = 0; f < nfreqs; f++)
//        {
//            model.radiation.u[R][i0+f] = Su[j0+f];
//            model.radiation.v[R][i0+f] = Sv[j0+f] * reverse;
//        }
//    }
//}
//
//
//
//
//inline void RayPair :: get_eta_and_chi (const size_t d, const size_t f)
//{
//}
//
//
//
//
//inline void RayPair :: solve (const long f)
//{
//    /// SETUP FEAUTRIER RECURSION RELATION
//    //////////////////////////////////////
//
//
//    /// Determine emissivities, opacities and optical depth increments
//    for (size_t n = first; n <= last; n++)
//    {
//        const size_t idf = I(n,f);
//
//        /// Initialize
//        eta[idf] = 0.0E+00;
//        chi[idf] = 1.0E-26;
//
//        /// Set line emissivity and opacity
//        for (size_t l = 0; l < nlines; l++)
//        {
//            const size_t lnl     = L(nrs[d],l);
//            const double diff    = freqs_scaled[idf] - line[l];
//            const double profile = freqs_scaled[idf] * gaussian (width[lnl], diff);
//
//            eta[idf] += profile * line_emissivity[lnl];
//            chi[idf] += profile * line_opacity   [lnl];
//        }
//
//        term1[idf] = eta[idf] / chi[idf];
//    }
//
//    for (size_t n = first; n <  last; n++)
//    {
//        dtau[I(n,f)] = 0.5 * (chi[I(n,f)] + chi[I(n+1,f)]) * dZs[n];
//    }
//
//
//    /// Set boundary conditions
//
//    const double inverse_dtau0 = 1.0 / dtau[I(first, f)];
//    const double inverse_dtaud = 1.0 / dtau[I(last-1,f)];
//
//    C[I(first,f)] = 2.0 * inverse_dtau0 * inverse_dtau0;
//    A[I(last, f)] = 2.0 * inverse_dtaud * inverse_dtaud;
//
//    const double B0_min_C0 = fma (2.0, inverse_dtau0, 1.0);
//    const double Bd_min_Ad = fma (2.0, inverse_dtaud, 1.0);
//
//    const double B0 = B0_min_C0 + C[I(first,f)];
//    const double Bd = Bd_min_Ad + A[I(last, f)];
//
//    const double inverse_B0 = 1.0 / B0;
//
//    const double I_bdy_0 = frequency_interpolate (I_bdy_0_presc, f);
//    const double I_bdy_n = frequency_interpolate (I_bdy_n_presc, f);
//
//    Su[I(first,f)] = fma (2.0*I_bdy_0, inverse_dtau0, term1[I(first,f)]);
//    Su[I(last, f)] = fma (2.0*I_bdy_n, inverse_dtaud, term1[I(last, f)]);
//
//
//    /// Set body of Feautrier matrix
//
//    for (size_t n = first+1; n < last; n++)
//    {
//        inverse_A[I(n,f)] = 0.5 * (dtau[I(n-1,f)] + dtau[I(n,f)]) * dtau[I(n-1,f)];
//        inverse_C[I(n,f)] = 0.5 * (dtau[I(n-1,f)] + dtau[I(n,f)]) * dtau[I(n,  f)];
//
//        A[I(n,f)] = 1.0 / inverse_A[I(n,f)];
//        C[I(n,f)] = 1.0 / inverse_C[I(n,f)];
//
//        Su[I(n,f)] = term1[I(n,f)];
//    }
//
//
//    /// SOLVE FEAUTRIER RECURSION RELATION
//    //////////////////////////////////////
//
//
//    /// ELIMINATION STEP
//    ////////////////////
//
//    Su[I(first,f)] = Su[I(first,f)] * inverse_B0;
//
//    // F[0] = (B[0] - C[0]) / C[0];
//    F[I(first,f)] = 0.5 * B0_min_C0 * dtau[I(first,f)] * dtau[I(first,f)];
//    inverse_one_plus_F[I(first,f)] = 1.0 / (1.0 + F[I(first,f)]);
//
//    for (size_t n = first+1; n < last; n++)
//    {
//        F[I(n,f)] = (1.0 + A[I(n,f)]*F[I(n-1,f)]*inverse_one_plus_F[I(n-1,f)]) * inverse_C[I(n,f)];
//        inverse_one_plus_F[I(n,f)] = 1.0 / (1.0 + F[I(n,f)]);
//
//        Su[I(n,f)] = (Su[I(n,f)] + A[I(n,f)]*Su[I(n-1,f)]) * inverse_one_plus_F[I(n,f)] * inverse_C[I(n,f)];
//    }
//
//    const double denominator = 1.0 / fma (Bd, F[I(last-1,f)], Bd_min_Ad);
//
//    Su[I(last,f)] = fma (A[I(last,f)], Su[I(last-1,f)], Su[I(last,f)]) * (1.0 + F[I(last-1,f)]) * denominator;
//
//
//    /// BACK SUBSTITUTION
//    /////////////////////
//
//    if (n_ar < last)
//    {
//        // G[ndep-1] = (B[ndep-1] - A[ndep-1]) / A[ndep-1];
//        G[I(last,f)] = 0.5 * Bd_min_Ad * dtau[I(last-1,f)] * dtau[I(last-1,f)];
//        G_over_one_plus_G[I(last,f)] = G[I(last,f)] / (1.0 + G[I(last,f)]);
//
//        for (size_t n = last-1; n > n_ar; n--)
//        {
//            Su[I(n,f)] = fma (Su[I(n+1,f)], inverse_one_plus_F[I(n,f)], Su[I(n,f)]);
//
//            G[I(n,f)] = (1.0 + C[I(n,f)]*G_over_one_plus_G[I(n+1,f)]) * inverse_A[I(n,f)];
//            G_over_one_plus_G[I(n,f)] = G[I(n,f)] / (1.0 + G[I(n,f)]);
//        }
//
//        Su[I(n_ar,f)] = fma (Su[I(n_ar+1,f)], inverse_one_plus_F[I(n_ar,f)], Su[I(n_ar,f)]);
//
//        // printf("Su (n_ar) = %le\n", Su[I(n_ar,f)]);
//
//        L_diag[I(n_ar,f)] = inverse_C[I(n_ar,f)] / (F[I(n_ar,f)] + G_over_one_plus_G[I(n_ar+1,f)]);
//    }
//
//    else
//    {
//        L_diag[I(last,f)] = (1.0 + F[I(last-1,f)]) / fma (Bd, F[I(last-1,f)], Bd_min_Ad);
//    }
//}
