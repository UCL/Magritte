// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <mpi.h>
#include <omp.h>

#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

#include "image.hpp"
#include "folders.hpp"
#include "constants.hpp"
#include "GridTypes.hpp"
#include "mpiTools.hpp"


///  Constructor for IMAGE
//////////////////////////

RAYDATA ::
RAYDATA (const long num_of_cells,
         const long origin_cell,
         const long origin_ray   );
  : ncells (num_of_cells)
  , origin (origin_cell)
  , ray    (origin_ray)
{


}   // END OF CONSTRUCTOR




///  print: write out the images
///    @param[in] tag: tag for output file
//////////////////////////////////////////

template <int Dimension, long Nrays>
inline int RAYDATA ::
           initialize (const CELLS<Dimension,Nrays> &cells)
{

  // Reset number of cells on the ray

  n = 0;     


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
  
  lnotch_r[ncells] = 0;


  return (0);
    
}


inline int RAYDATA ::
           get_terms_dtau (const FREQUENCIES &frequencies,
                           const TEMPERATURE &temperature,
                           const LINES       &lines,
                           const SCATTERING  &scattering,
                           const RADIATION   &radiation,
                           const long         f,
                           const long         q,
                                 vReal       &term1,
                                 vReal       &term2,
                                 vReal       &dtau        )
{

  // Compute new frequency due to Doppler shift
  
  vReal freq_scaled = shifts[q] * frequencies.nu[origin][f];


  // Gather all contributions to the emissivity and opacity

  vReal eta,

  get_eta_chi (
      frequencies,
      temperature,
      lines,
      scattering,
      radiation,
      freq_scaled,
      q,
      eta,
      chi     )


  // Rescale scatterd radiation field U and V

  vReal U_scaled, V_scaled;

  radiation.rescale_U_and_V (
      frequencies,
      cellNrs[q],
      Ray,
      notch[q],
      freq_scaled,
      U_scaled,
      V_scaled              );


  U_scaled = 0.0;
  V_scaled = 0.0;


  // Define auxiliary terms

  term1 = (U_scaled + eta_n) / chi;
  term2 =  V_scaled          / chi;


  // Compute dtau
  
  dtau = 0.5 * (chi_n + chi_c) * dZs[q];


  // Store current value

  chi_c = chi_n;


  return (0);

}

inline int RAYDATA ::
           get_terms_dtau_I_bdy (const FREQUENCIES &frequencies,
                                 const TEMPERATURE &temperature,
                                 const LINES       &lines,
                                 const SCATTERING  &scattering,
                                 const RADIATION   &radiation,
                                 const long         f,
                                 const long         q,
                                       double      &dtau,
                                       double      &Su,
                                       double      &Sv         )
{

  // Compute new frequency due to Doppler shift
  
  vReal freq_scaled = shifts[q] * frequencies.nu[origin][f];


  // Gather all contributions to the emissivity and opacity

  vReal eta_n, chi_n;

  get_eta_chi (
      frequencies,
      temperature,
      lines,
      scattering,
      radiation,
      freq_scaled,
      q,
      eta_n,
      chi_n   )


  // Rescale scatterd radiation field U and V

  vReal U_scaled, V_scaled;

  const long b = cells.cell_to_bdy_nr[cellNrs[q]];

  radiation.rescale_U_and_V_and_bdy_I (
      frequencies,
      cellNrs[q],
      b,
      Ray,
      notch[q],
      freq_scaled,
      U_scaled,
      V_scaled,
      Ibdy_scaled                     );


  U_scaled = 0.0;
  V_scaled = 0.0;


  // Define auxiliary terms

  term1 = (U_scaled + eta) / chi;
  term2 =  V_scaled        / chi;


  // Compute dtau
  
  dtau = 0.5 * (chi_n + chi_c) * dZs[q];


  return (0);

}




inline int RAYDATA ::
           get_eta_chi (const FREQUENCIES &frequencies,
                        const TEMPERATURE &temperature,
                        const LINES       &lines,
                        const SCATTERING  &scattering,
                        const RADIATION   &radiation,
                        const vReal        freq_scaled,
                        const long         q,
                              vReal       &eta,
                              vReal       &chi         )
{

  // Reset emissivity and opacity

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


  return (0);

}
