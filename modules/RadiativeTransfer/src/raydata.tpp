// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "GridTypes.hpp"
#include "mpiTools.hpp"
#include "interpolation.hpp"


//  initialize: reset all values for a new ray
///    @param[in] cells: data structure containing the geometric cell data
///    @param[in] o: origin of the ray
//////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
inline void RAYDATA ::
    initialize                              (
        const CELLS<Dimension,Nrays> &cells,
        const long                    o     )
{

  // Reset origin and number of cells on the ray

  origin = o;
  n      = 0;     


  // Find projected cells on ray

  double  Z = 0.0;   // distance from origin (o)
  double dZ = 0.0;   // last increment in Z

  long nxt = cells.next (origin, ray, origin, Z, dZ);


  if (nxt != ncells)   // if we are not going out of grid
  {
    cellNrs[n] = nxt;
        dZs[n] = dZ;

    n++;

    while (!cells.boundary[nxt])   // while we have not hit the boundary
    {
      nxt = cells.next (origin, ray, nxt, Z, dZ);

      cellNrs[n] = nxt;
          dZs[n] = dZ;

      n++;
    }
  }


  // Initialize notches and shitfs

  for (long q = 0; q < n; q++)
  {
     notch[q] = 0;
    lnotch[q] = 0;
    shifts[q] = 1.0 - cells.relative_velocity (origin, ray, cellNrs[q]) / CC;
  }
  

  // Put origin informartion at ncells

  cellNrs[ncells] = origin;
   lnotch[ncells] = 0;
    notch[ncells] = 0;

}


inline void RAYDATA ::
            set_current_to_origin              (
                const FREQUENCIES &frequencies,
                const TEMPERATURE &temperature,
                const LINES       &lines,
                const SCATTERING  &scattering,
                const long         f           )
{

  // Gather all contributions to the emissivity and opacity

  compute_next_eta_and_chi (
      frequencies,
      temperature,
      lines,
      scattering,
      frequencies.nu[origin][f],
      ncells               );


  // Define auxiliary term

  vReal inverse_chi_n = 1.0 / chi_n;


  // Compute (current) terms

  term1_c = (U[Ray][index(origin,f)] + eta_n) * inverse_chi_n;
  term2_c =  V[Ray][index(origin,f)]          * inverse_chi_n;

  chi_c = chi_n;

}

inline void RAYDATA ::
            set_current_to_origin_bdy          (
                const FREQUENCIES &frequencies,
                const TEMPERATURE &temperature,
                const LINES       &lines,
                const SCATTERING  &scattering,
                const long         f           )
{

  // Gather all contributions to the emissivity and opacity

  compute_next_eta_and_chi (
      frequencies,
      temperature,
      lines,
      scattering,
      frequencies.nu[origin][f],
      ncells              );

  // Define I_bdy_scaled (which is not scaled in this case)
  
  const long b = cell2boundary_nr[origin];

  Ibdy_scaled = boundary_intensity[Ray][b][f];


  // Define auxiliary term

  vReal inverse_chi_n = 1.0 / chi_n;


  // Compute (current) terms

  term1_c = (U[Ray][index(origin,f)] + eta_n) * inverse_chi_n;
  term2_c =  V[Ray][index(origin,f)]          * inverse_chi_n;

  chi_c = chi_n;

}

inline void RAYDATA ::
            compute_next                       (
                const FREQUENCIES &frequencies,
                const TEMPERATURE &temperature,
                const LINES       &lines,
                const SCATTERING  &scattering,
                const long         f,
                const long         q           )
{

  // Compute new frequency due to Doppler shift
  
  vReal freq_scaled = shifts[q] * frequencies.nu[origin][f];


  // Gather all contributions to the emissivity and opacity

  compute_next_eta_and_chi (
      frequencies,
      temperature,
      lines,
      scattering,
      freq_scaled,
      q                    );


  // Rescale scatterd radiation field U and V

  vReal U_scaled, V_scaled;

  // !!! What to do with antipodal ray ???
  rescale_U_and_V           (
      frequencies,
      cellNrs[q],
      Ray,
      notch[q],
      freq_scaled,
      U_scaled,
      V_scaled              );


  U_scaled = 0.0;
  V_scaled = 0.0;


  compute_next_terms_and_dtau (U_scaled, V_scaled, q);

//  if (f == frequencies.nr_line[origin][0][19][20])
//  {
//    cout << "chi " << chi_c + chi_n << "    chi_c " << chi_c << "    chi_n " << chi_n << "    dZ " << dZs[q] << endl;  
//  }
}

inline void RAYDATA ::
            compute_next_bdy                   (
                const FREQUENCIES &frequencies,
                const TEMPERATURE &temperature,
                const LINES       &lines,
                const SCATTERING  &scattering,
                const long         f           )
{

  // We know q is a boundary point

  long q = n-1;


  // Compute new frequency due to Doppler shift
  
  vReal freq_scaled = shifts[q] * frequencies.nu[origin][f];


  // Gather all contributions to the emissivity and opacity

  compute_next_eta_and_chi (
      frequencies,
      temperature,
      lines,
      scattering,
      freq_scaled,
      q                    );


  // Rescale scatterd radiation field U and V

  vReal U_scaled, V_scaled;

  rescale_U_and_V_and_bdy_I (
      frequencies,
      cellNrs[q],
      Ray,
      notch[q],
      freq_scaled,
      U_scaled,
      V_scaled,
      Ibdy_scaled           );


  U_scaled = 0.0;
  V_scaled = 0.0;


  compute_next_terms_and_dtau (U_scaled, V_scaled, q);

}




inline void RAYDATA ::
            compute_next_eta_and_chi           (
                const FREQUENCIES &frequencies,
                const TEMPERATURE &temperature,
                const LINES       &lines,
                const SCATTERING  &scattering,
                const vReal        freq_scaled,
                const long         q           )
{

  // Reset eta and chi (next)

  eta_n = 0.0;
  chi_n = 0.0;


  // Add line contributions

  lines.add_emissivity_and_opacity (
      frequencies,
      temperature,
      freq_scaled,
      lnotch[q],
      cellNrs[q],
      eta_n,
      chi_n                        );


  // Add scattering contributions

  scattering.add_opacity (
      freq_scaled,
      chi_n              );


  // Set minimal opacity to avoid zero optical depth increments (dtau)

# if (GRID_SIMD)
    for (int lane = 0; lane < n_simd_lanes; lane++)
    {
      if (fabs(chi_n.getlane(lane)) < 1.0E-30)
      {
        chi_n.putlane(1.0E-30, lane);
      }
    }
# else
    if (fabs(chi_n) < 1.0E-30)
    {
      chi_n = 1.0E-30;
    }
# endif


}



inline void RAYDATA ::
    compute_next_terms_and_dtau (
        const vReal U_scaled,
        const vReal V_scaled,
        const long  q           )
{

  // Define auxiliary term

  vReal inverse_chi_n = 1.0 / chi_n;


  // Compute new terms

  term1_n = (U_scaled + eta_n) * inverse_chi_n;
  term2_n =  V_scaled          * inverse_chi_n;


  // Compute dtau and its inverse

  dtau = 0.5 * (chi_n + chi_c) * dZs[q];

  inverse_dtau = 1.0 / dtau;

}





// Getters for source functions

inline vReal RAYDATA ::
    get_Su_r  (void) const
{
  return 0.5 * (term1_n + term1_c) - (term2_n - term2_c) * inverse_dtau;
}

inline vReal RAYDATA ::
    get_Sv_r  (void) const
{
  return 0.5 * (term2_n + term2_c) - (term1_n - term1_c) * inverse_dtau;
}

inline vReal RAYDATA ::
    get_Su_ar (void) const
{
  return 0.5 * (term1_n + term1_c) + (term2_n - term2_c) * inverse_dtau;
}

inline vReal RAYDATA ::
    get_Sv_ar (void) const
{
  return 0.5 * (term2_n + term2_c) + (term1_n - term1_c) * inverse_dtau;
}


// Getters for boundary term

inline vReal RAYDATA ::
    get_boundary_term_Su_r  (void) const
{
  return 2.0 * inverse_dtau * (Ibdy_scaled - 0.5 * (term2_c + term2_n));
}

inline vReal RAYDATA ::
    get_boundary_term_Sv_r  (void) const
{
  return 2.0 * inverse_dtau * (Ibdy_scaled - 0.5 * (term1_c + term1_n));
}

inline vReal RAYDATA ::
    get_boundary_term_Su_ar (void) const
{
  return 2.0 * inverse_dtau * (Ibdy_scaled + 0.5 * (term2_c + term2_n));
}

inline vReal RAYDATA ::
    get_boundary_term_Sv_ar (void) const
{
  return 2.0 * inverse_dtau * (Ibdy_scaled + 0.5 * (term1_c + term1_n));
}


inline void RAYDATA ::
    set_current_to_next (void)
{
  term1_c = term1_n;
  term2_c = term2_n;
    chi_c = chi_n;
}


inline void RAYDATA ::
    rescale_U_and_V                    (
        const FREQUENCIES &frequencies,
        const long         p,
        const long         R,
              long        &notch,
        const vReal       &freq_scaled,
              vReal       &U_scaled,
              vReal       &V_scaled    )

#if (GRID_SIMD)

{

  vReal nu1, nu2, U1, U2, V1, V2;

  for (int lane = 0; lane < n_simd_lanes; lane++)
  {

    const double freq = freq_scaled.getlane (lane);

    search_with_notch (frequencies.nu[p], notch, freq);

    const long f1    =  notch    / n_simd_lanes;
    const  int lane1 =  notch    % n_simd_lanes;

    const long f2    = (notch-1) / n_simd_lanes;
    const  int lane2 = (notch-1) % n_simd_lanes;

    nu1.putlane (frequencies.nu[p][f1].getlane (lane1), lane);
    nu2.putlane (frequencies.nu[p][f2].getlane (lane2), lane);
    
     U1.putlane (U[R][index(p,f1)].getlane (lane1), lane);
     U2.putlane (U[R][index(p,f2)].getlane (lane2), lane);
    
     V1.putlane (V[R][index(p,f1)].getlane (lane1), lane);
     V2.putlane (V[R][index(p,f2)].getlane (lane2), lane);
  }

  U_scaled = interpolate_linear (nu1, U1, nu2, U2, freq_scaled);
  V_scaled = interpolate_linear (nu1, V1, nu2, V2, freq_scaled);


}

#else

{

  search_with_notch (frequencies.nu[p], notch, freq_scaled);

  const long f1    = notch;
  const long f2    = notch-1;

  const double nu1 = frequencies.nu[p][f1];
  const double nu2 = frequencies.nu[p][f2];

  const double U1 = U[R][index(p,f1)];
  const double U2 = U[R][index(p,f2)];

  const double V1 = V[R][index(p,f1)];
  const double V2 = V[R][index(p,f2)];

  U_scaled = interpolate_linear (nu1, U1, nu2, U2, freq_scaled);
  V_scaled = interpolate_linear (nu1, V1, nu2, V2, freq_scaled);


}

#endif




inline void RAYDATA ::
    rescale_U_and_V_and_bdy_I         (
        const FREQUENCIES &frequencies,
        const long         p,
        const long         R,
              long        &notch,
        const vReal       &freq_scaled,
              vReal       &U_scaled,
              vReal       &V_scaled,
              vReal       &Ibdy_scaled )
   
#if (GRID_SIMD)

{

  vReal nu1, nu2, U1, U2, V1, V2, Ibdy1, Ibdy2;

  const long b = cell2boundary_nr[p];


  for (int lane = 0; lane < n_simd_lanes; lane++)
  {
    double freq = freq_scaled.getlane (lane);

    search_with_notch (frequencies.nu[p], notch, freq);

    const long f1    =  notch    / n_simd_lanes;
    const  int lane1 =  notch    % n_simd_lanes;

    const long f2    = (notch-1) / n_simd_lanes;
    const  int lane2 = (notch-1) % n_simd_lanes;

      nu1.putlane (      frequencies.nu[p][f1].getlane (lane1), lane);
      nu2.putlane (      frequencies.nu[p][f2].getlane (lane2), lane);
    
       U1.putlane (           U[R][index(p,f1)].getlane (lane1), lane);
       U2.putlane (           U[R][index(p,f2)].getlane (lane2), lane);
    
       V1.putlane (           V[R][index(p,f1)].getlane (lane1), lane);
       V2.putlane (           V[R][index(p,f2)].getlane (lane2), lane);
    
    Ibdy1.putlane (boundary_intensity[R][b][f1].getlane (lane1), lane);
    Ibdy2.putlane (boundary_intensity[R][b][f2].getlane (lane2), lane);
  }
    
     U_scaled = interpolate_linear (nu1, U1,    nu2,    U2, freq_scaled);
     V_scaled = interpolate_linear (nu1, V1,    nu2,    V2, freq_scaled);
  Ibdy_scaled = interpolate_linear (nu1, Ibdy1, nu2, Ibdy2, freq_scaled);


}

#else

{
  const long b = cell2boundary_nr[p];

  search_with_notch (frequencies.nu[p], notch, freq_scaled);
  
  const long f1    = notch;
  const long f2    = notch-1;

  const double nu1 = frequencies.nu[p][f1];
  const double nu2 = frequencies.nu[p][f2];
  
  const double U1 = U[R][index(p,f1)];
  const double U2 = U[R][index(p,f2)];
  
  const double V1 = V[R][index(p,f1)];
  const double V2 = V[R][index(p,f2)];
  
  const double Ibdy1 = boundary_intensity[R][b][f1];
  const double Ibdy2 = boundary_intensity[R][b][f2];
  
     U_scaled = interpolate_linear (nu1, U1,    nu2, U2,    freq_scaled);
     V_scaled = interpolate_linear (nu1, V1,    nu2, V2,    freq_scaled);
  Ibdy_scaled = interpolate_linear (nu1, Ibdy1, nu2, Ibdy2, freq_scaled);


}

#endif

inline long RAYDATA ::
    index            (
        const long p,
        const long f ) const
{
  return f + p*nfreq_red;
}
