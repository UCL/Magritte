// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "set_up_ray.hpp"
#include "types.hpp"
#include "GridTypes.hpp"
#include "cells.hpp"
#include "lines.hpp"
#include "scattering.hpp"
#include "radiation.hpp"


///  set_up_ray: extract sources and opacities from the grid on the ray
///    @param[in] cells: reference to the geometric cell data containing the grid
///    @param[in] frequencies: reference to data structure containing freqiencies
///    @param[in] lines: reference to data structure containing line transfer data
///    @param[in] scattering: reference to structure containing scattering data
///    @param[in] radiation: reference to (previously calculated) radiation field
///    @param[in] o: number of the cell from which the ray originates
///    @param[in] r: number of the ray which is being set up
///    @param[in] R: local number of the ray which is being set up
///    @param[in] raytype: indicates whether we walk forward or backward along ray
///    @param[out] n: reference to the resulting number of points along the ray
///    @param[out] Su: reference to the source for u extracted along the ray
///    @param[out] Sv: reference to the source for v extracted along the ray
///    @param[out] dtau: reference to the optical depth increments along the ray
//////////////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
inline int set_up_ray (const CELLS<Dimension, Nrays> &cells,
                       const FREQUENCIES             &frequencies,
                       const TEMPERATURE             &temperature,
                       const LINES                   &lines,
                       const SCATTERING              &scattering,
		       const RADIATION               &radiation,
                       const long                     f,
                       const long                     o,
                       const long                     R,
			     long                    *lnotch_ar,
                             long                    *notch_ar,
                       const long                    *cellNrs_ar,
	               const double                  *shifts_ar,
                       const double                  *dZs_ar,
                       const long                     n_ar,
			     long                    *lnotch_r,
                             long                    *notch_r,
                       const long                    *cellNrs_r,
                       const double                  *shifts_r,
                       const double                  *dZs_r,
                       const long                     n_r,
	                     vReal                  *Su,
                             vReal                  *Sv,
                             vReal                  *dtau,
                       const long                    ndep          )
{

  vReal chi_o, term1_o, term2_o, eta_o;
  vReal chi_c, term1_c, term2_c;
  vReal chi_n, term1_n, term2_n;

  vReal Ibdy_scaled;


  get_eta_and_chi (frequencies, temperature, lines, scattering, radiation,
                   lnotch_r[cells.ncells], o, frequencies.nu[o][f], eta_o, chi_o);

    chi_c =   chi_o;
  term1_c = term1_o =(radiation.U[R][radiation.index(o,f)] + eta_o) / chi_o;
  term2_c = term2_o = radiation.V[R][radiation.index(o,f)]          / chi_o;




  if ( (n_ar > 0) && (n_r > 0) )
  {

    for (long q = 0; q < n_ar; q++)
    {
      get_terms_and_chi (frequencies, temperature, lines, scattering, radiation, f, o, R,
                         lnotch_ar[q], notch_ar[q], cellNrs_ar[q], shifts_ar[q], term1_n, term2_n, chi_n);

      dtau[n_ar-1-q] = 0.5 * (chi_n + chi_c) * dZs_ar[q];
        Su[n_ar-1-q] = 0.5 * (term1_n + term1_c) + (term2_n - term2_c) / dtau[n_ar-1-q];
       	Sv[n_ar-1-q] = 0.5 * (term2_n + term2_c) + (term1_n - term1_c) / dtau[n_ar-1-q];

        chi_c =   chi_n;
      term1_c = term1_n;
      term2_c = term2_n;
    }


    get_terms_chi_and_Ibdy (cells, frequencies, temperature, lines, scattering, radiation, f, o, R,
                            lnotch_ar[n_ar-1], notch_ar[n_ar-1], cellNrs_ar[n_ar-1], shifts_ar[n_ar-1],
                            term1_n, term2_n, chi_n, Ibdy_scaled);
 
    Su[0] += 2.0 / dtau[0] * (Ibdy_scaled - 0.5 * (term2_c + term2_n));
    Sv[0] += 2.0 / dtau[0] * (Ibdy_scaled - 0.5 * (term1_c + term1_n));
    

      chi_c =   chi_o;
    term1_c = term1_o;
    term2_c = term2_o;


    for (long q = 0; q < n_r; q++)
    {
       get_terms_and_chi (frequencies, temperature, lines, scattering, radiation, f, o, R,
                          lnotch_r[q], notch_r[q], cellNrs_r[q], shifts_r[q], term1_n, term2_n, chi_n);

      dtau[n_ar+q] = 0.5 * (chi_n + chi_c) * dZs_r[q];
        Su[n_ar+q] = 0.5 * (term1_n + term1_c) + (term2_n - term2_c) / dtau[n_ar+q];
       	Sv[n_ar+q] = 0.5 * (term2_n + term2_c) + (term1_n - term1_c) / dtau[n_ar+q];


        chi_c =   chi_n;
      term1_c = term1_n;
      term2_c = term2_n;
    }


    get_terms_chi_and_Ibdy (cells, frequencies, temperature, lines, scattering, radiation, f, o, R,
                            lnotch_r[n_r-1], notch_r[n_r-1], cellNrs_r[n_r-1], shifts_r[n_r-1],
                            term1_n, term2_n, chi_n, Ibdy_scaled);

    Su[ndep-1] += 2.0 / dtau[ndep-1] * (Ibdy_scaled - 0.5 * (term2_c + term2_n));
    Sv[ndep-1] += 2.0 / dtau[ndep-1] * (Ibdy_scaled - 0.5 * (term1_c + term1_n));
  }




  else if (n_ar > 0)   // and hence n_r == 0
  {

    for (long q = 0; q < n_ar; q++)
    {
      get_terms_and_chi (frequencies, temperature, lines, scattering, radiation, f, o, R,
                         lnotch_ar[q], notch_ar[q], cellNrs_ar[q], shifts_ar[q], term1_n, term2_n, chi_n);

      dtau[n_ar-1-q] = 0.5 * (chi_n + chi_c) * dZs_ar[q];
        Su[n_ar-1-q] = 0.5 * (term1_n + term1_c) - (term2_n - term2_c) / dtau[n_ar-1-q];
        Sv[n_ar-1-q] = 0.5 * (term2_n + term2_c) - (term1_n - term1_c) / dtau[n_ar-1-q];

        chi_c =   chi_n;
      term1_c = term1_n;
      term2_c = term2_n;
    }


    get_terms_chi_and_Ibdy (cells, frequencies, temperature, lines, scattering, radiation, f, o, R,
                            lnotch_ar[n_ar-1], notch_ar[n_ar-1], cellNrs_ar[n_ar-1], shifts_ar[n_ar-1],
                            term1_n, term2_n, chi_n, Ibdy_scaled);

    Su[0] += 2.0 / dtau[0] * (Ibdy_scaled + 0.5 * (term2_c + term2_n));
    Sv[0] += 2.0 / dtau[0] * (Ibdy_scaled + 0.5 * (term1_c + term1_n));


    const long b = cells.cell2boundary_nr[o];

    get_terms_and_chi (frequencies, temperature, lines, scattering, radiation, f, o, R,
                       lnotch_ar[0], notch_ar[0], cellNrs_ar[0], shifts_ar[0], term1_n, term2_n, chi_n);

    Su[n_ar-1] += 2.0 / dtau[n_ar-1] * (radiation.boundary_intensity[R][b][f] - 0.5 * (term2_o + term2_n));
    Sv[n_ar-1] += 2.0 / dtau[n_ar-1] * (radiation.boundary_intensity[R][b][f] - 0.5 * (term1_o + term1_n));

  }




  else if (n_r > 0)   // and hence n_ar == 0
  {

    for (long q = 0; q < n_r; q++)
    {
      get_terms_and_chi (frequencies, temperature, lines, scattering, radiation, f, o, R,
                         lnotch_r[q], notch_r[q], cellNrs_r[q], shifts_r[q], term1_n, term2_n, chi_n);

      dtau[q] = 0.5 * (chi_n + chi_c) * dZs_r[q];
        Su[q] = 0.5 * (term1_n + term1_c) + (term2_n - term2_c) / dtau[q];
       	Sv[q] = 0.5 * (term2_n + term2_c) + (term1_n - term1_c) / dtau[q];

        chi_c =   chi_n;
      term1_c = term1_n;
      term2_c = term2_n;
    }


    get_terms_chi_and_Ibdy (cells, frequencies, temperature, lines, scattering, radiation, f, o, R,
                            lnotch_r[n_r-1], notch_r[n_r-1], cellNrs_r[n_r-1], shifts_r[n_r-1],
                            term1_n, term2_n, chi_n, Ibdy_scaled);

    Su[n_r-1] += 2.0 / dtau[n_r-1] * (Ibdy_scaled - 0.5 * (term2_c + term2_n));
    Sv[n_r-1] += 2.0 / dtau[n_r-1] * (Ibdy_scaled - 0.5 * (term1_c + term1_n));

    const long b = cells.cell2boundary_nr[o];

    get_terms_and_chi (frequencies, temperature, lines, scattering, radiation, f, o, R,
                       lnotch_r[0], notch_r[0], cellNrs_r[0], shifts_r[0], term1_n, term2_n, chi_n);

    Su[0] += 2.0 / dtau[0] * (radiation.boundary_intensity[R][b][f] - 0.5 * (term2_o + term2_n));
    Sv[0] += 2.0 / dtau[0] * (radiation.boundary_intensity[R][b][f] - 0.5 * (term1_o + term1_n));
  }


  return (0);

}


inline int get_eta_and_chi (const FREQUENCIES &frequencies,
                            const TEMPERATURE &temperature,
                            const LINES       &lines,
                            const SCATTERING  &scattering,
                            const RADIATION   &radiation,
				  long        &lnotch,
                            const long         cellNrs,
			    const vReal        freq_scaled,
                                  vReal       &eta,
                                  vReal       &chi          )
{
 
  eta = 0.0;
  chi = 0.0;

  lines.add_emissivity_and_opacity (frequencies, temperature, freq_scaled,
                                    lnotch, cellNrs, eta, chi);

   

  scattering.add_opacity (freq_scaled, chi);


  // Set minimal opacity to avoid zero optical depth increments (dtau)

# if (GRID_SIMD)
    for (int lane = 0; lane < n_simd_lanes; lane++)
    {
      if (fabs(chi.getlane(lane)) < 1.0E-30)
      {
        chi.putlane(1.0E-30, lane);
      }
    }
# else
    if (fabs(chi) < 1.0E-30)
    {
      chi = 1.0E-30;
    }
# endif


  return (0);

}


inline int get_terms_and_chi (const FREQUENCIES &frequencies,
                              const TEMPERATURE &temperature,
                              const LINES       &lines,
                              const SCATTERING  &scattering,
                              const RADIATION   &radiation,
			      const long         f,
                              const long         o,
                              const long         R,
                                    long        &lnotch,
                                    long        &notch,
                              const long         cellNrs,
                              const double       shifts,
	                            vReal       &term1,
                                    vReal       &term2,
                                    vReal       &chi         )
{

  vReal eta, U_scaled, V_scaled;
  
  vReal freq_scaled  = shifts * frequencies.nu[o][f];

  //cout.precision(16);
  //cout << "shifts = " << shifts << fixed << endl;
  //cout << "freq scaled " << freq_scaled - frequencies.nu[o][f] << endl;

  get_eta_and_chi (frequencies, temperature, lines, scattering, radiation,
                   lnotch, cellNrs, freq_scaled, eta, chi);


  // Placeholer for continuum...


  radiation.rescale_U_and_V (frequencies, cellNrs, R, notch, freq_scaled, U_scaled, V_scaled);


  // temporary !!!

  U_scaled = 0.0;
  V_scaled = 0.0;

  term1 = (U_scaled + eta) / chi;
  term2 =  V_scaled        / chi;

  //if (f == frequencies.nr_line[o][0][0][0])
  //{
  //  cout << "Setup : " << cellNrs <<  "    eta = " << eta << "    chi = " << chi << "    S = " << eta/chi << endl;
  //}

  return (0);

}


template <int Dimension, long Nrays>
inline int get_terms_chi_and_Ibdy (const CELLS<Dimension, Nrays> &cells,
                                   const FREQUENCIES             &frequencies,
                                   const TEMPERATURE             &temperature,
                                   const LINES                   &lines,
                                   const SCATTERING              &scattering,
                                   const RADIATION               &radiation,
				   const long                     f,
                                   const long                     o,
                                   const long                     R,
				         long                    &lnotch,
                                         long                    &notch,
                                   const long                     cellNrs,
                                   const double                   shifts,
	                                 vReal                   &term1,
                                         vReal                   &term2,
                                         vReal                   &chi,
                                         vReal                   &Ibdy_scaled)
{

  vReal eta, U_scaled, V_scaled;

  vReal freq_scaled  = shifts * frequencies.nu[o][f];

  get_eta_and_chi (frequencies, temperature, lines, scattering, radiation,
                   lnotch, cellNrs, freq_scaled, eta, chi);

  const long b = cells.cell2boundary_nr[cellNrs];

  radiation.rescale_U_and_V_and_bdy_I (frequencies, cellNrs, R, notch, freq_scaled,
                                       U_scaled, V_scaled, Ibdy_scaled);

  // temporary !!!

  U_scaled = 0.0;
  V_scaled = 0.0;
  //Ibdy_scaled = 0.0;


  term1 = (U_scaled + eta) / chi;
  term2 =  V_scaled        / chi;

  //if (f == frequencies.nr_line[o][0][0][0])
  //{
  //  cout << "Setup : " << cellNrs <<  "   eta = " << eta << "    chi = " << chi << "   S = " << eta/chi << endl;
  //}

  return (0);

}
