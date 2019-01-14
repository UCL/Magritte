// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________

#include <iomanip>

#include "GridTypes.hpp"
#include "mpiTools.hpp"
#include "interpolation.hpp"


//  initialize: reset all values for a new ray
///    @param[in] cells: data structure containing the geometric cell data
///    @param[in] o: origin of the ray
//////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
inline void RAYDATA ::
    initialize                                    (
        const CELLS<Dimension,Nrays> &cells,
        const TEMPERATURE            &temperature,
        const long                    o           )
{

  // Reset origin and number of cells on the ray

  origin = o;
  n      = 0;


  // Find projected cells on ray

  double  Z = 0.0;   // distance from origin (o)
  double dZ = 0.0;   // last increment in Z

  long nxt = cells.next (origin, ray, origin, Z, dZ);


  // Clear the three vectors that will dynamically grow

  cellNrs.clear ();
   shifts.clear ();
      dZs.clear ();


  if (nxt != ncells)   // if we are not going out of grid
  {

    const double shift_max = 0.5 * profile_width (temperature.gas[o],
                                                  temperature.vturb2[o]);

    double shift_crt = 1.0;
    long   crt       = origin;
    double shift_nxt = cells.doppler_shift (origin, ray, nxt);

    set_projected_cell_data (crt, nxt, dZ, shift_crt, shift_nxt, shift_max);


    while (!cells.boundary[nxt])   // while we have not hit the boundary
    {
      shift_crt = shift_nxt;
      nxt       = cells.next          (origin, ray, nxt, Z, dZ);
      shift_nxt = cells.doppler_shift (origin, ray, nxt);

      set_projected_cell_data (crt, nxt, dZ, shift_crt, shift_nxt, shift_max);
    }
  }


  if (n+1 > notch.size())
  {
    lnotch.resize (n+10);
     notch.resize (n+10);
  }


  // Initialize notches and shitfs

  for (long q = 0; q < n; q++)
  {
    lnotch[q] = 0;
     notch[q] = 0;
  }


  // Put origin informartion at n

  cellNrs.push_back (origin);

  lnotch[n] = 0;
   notch[n] = 0;

}


inline void RAYDATA ::
    set_projected_cell_data    (
        const long   crt,
        const long   nxt,
        const double dZ,
        const double shift_crt,
        const double shift_nxt,
        const double shift_max )
{

  // If velocity gradient is not well-sampled enough

  if (fabs(shift_nxt - shift_crt) > shift_max)
  {

    // Interpolate velocity gradient field

    const int         n_interpl = fabs(shift_nxt - shift_crt) / shift_max + 1;
    const int    half_n_interpl = n_interpl / 2;
    const double     dZ_interpl =                      dZ / n_interpl;
    const double dshift_interpl = (shift_nxt - shift_crt) / n_interpl;


    // Assign current cell to first half of interpolation points

    for (int m = 1; m < half_n_interpl; m++)
    {
      cellNrs.push_back (crt);
    }


    // Assign next cell to second half of interpolation points

    for (int m = half_n_interpl; m <= n_interpl; m++)
    {
      cellNrs.push_back (nxt);
    }


    // Add interpolated shifts and distance increments

    for (int m = 1; m <= n_interpl; m++)
    {
      shifts.push_back (shift_crt + m * dshift_interpl);
         dZs.push_back (dZ_interpl                    );
    }


    // Increment the number of cells on the ray

    n += n_interpl;
  }

  else
  {
    cellNrs.push_back (nxt      );
     shifts.push_back (shift_nxt);
        dZs.push_back (dZ       );

    n++;
  }

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

  compute_next_eta_and_chi      (
      frequencies,
      temperature,
      lines,
      scattering,
      frequencies.nu[origin][f],
      n                         );


  // Set chi at origin

  chi_o = chi_n;


  // Define auxiliary term

  vReal inverse_chi_n = 1.0 / chi_n;


  // Compute (current) terms

  term1 = (U[Ray][index(origin,f)]*0.0 + eta) * inverse_chi_n;
  term2 =  V[Ray][index(origin,f)]*0.0        * inverse_chi_n;

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

  compute_next_eta_and_chi      (
      frequencies,
      temperature,
      lines,
      scattering,
      frequencies.nu[origin][f],
      n                         );


  // Set chi at origin

  chi_o = chi_n;


  // Define I_bdy_scaled (which is not scaled in this case)

  const long b = cell2boundary_nr[origin];

  Ibdy_scaled = boundary_intensity[Ray][b][f];


  // Define auxiliary term

  vReal inverse_chi_n = 1.0 / chi_n;


  // Compute (current) terms

  term1 = (U[Ray][index(origin,f)]*0.0 + eta) * inverse_chi_n;
  term2 =  V[Ray][index(origin,f)]*0.0        * inverse_chi_n;

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
  rescale_U_and_V (
      frequencies,
      cellNrs[q],
      Ray,
      notch[q],
      freq_scaled,
      U_scaled,
      V_scaled    );


  U_scaled = 0.0;
  V_scaled = 0.0;


  compute_next_terms_and_dtau (U_scaled, V_scaled, q);

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

  // Save old chi_n in chi_c

  chi_c = chi_n;


  // Reset eta and chi (next)

  eta   = 0.0;
  chi_n = 0.0;


  // Add line contributions

  lines.add_emissivity_and_opacity (
      frequencies,
      temperature,
      freq_scaled,
      lnotch[q],
      cellNrs[q],
      eta,
      chi_n                        );


  // Add scattering contributions

  scattering.add_opacity (
      freq_scaled,
      chi_n              );


  // Set minimal opacity to avoid zero optical depth increments (dtau)

# if (GRID_SIMD)
    for (int lane = 0; lane < n_simd_lanes; lane++)
    {
      if (fabs(chi_n.getlane(lane)) < 1.0E-99)
      {
        chi_n.putlane(1.0E-99, lane);
          eta.putlane((eta / (chi_n * 1.0E+99)).getlane(lane), lane);

        //cout << "WARNING : Opacity reached lower bound (1.0E-99)" << endl;
      }
    }
# else
    if (fabs(chi_n) < 1.0E-99)
    {
      chi_n = 1.0E-99;
      eta   = eta / (chi_n * 1.0E+99);

      //cout << "WARNING : Opacity reached lower bound (1.0E-99)" << endl;
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

  term1 = (U_scaled*0.0 + eta) * inverse_chi_n;
  term2 =  V_scaled*0.0        * inverse_chi_n;


  // Compute dtau and its inverse

  dtau = 0.5 * (chi_n + chi_c) * dZs[q];

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
  //Ibdy_scaled = planck (T_CMB, freq_scaled); //interpolate_linear (nu1, Ibdy1, nu2, Ibdy2, freq_scaled);
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
  //Ibdy_scaled = planck (T_CMB, freq_scaled); //interpolate_linear (nu1, Ibdy1, nu2, Ibdy2, freq_scaled);
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
