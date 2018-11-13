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

RAYPAIR ::
RAYPAIR (const long num_of_cells,
         const long origin_Ray   )
  : ncells (num_of_cells)
  : Ray    (origin_Ray)
{


}   // END OF CONSTRUCTOR




///  print: write out the images
///    @param[in] tag: tag for output file
//////////////////////////////////////////

///  read: read the cells, neighbors and boundary files
///////////////////////////////////////////////////////

inline int RAYPAIR ::
           initialize (const RAYDATA &raydata_r,
                       const RAYDATA &raydata_ar)
{
  ndep = raydata_r.n + raydata_ar.n;

  return (0);
}

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
inline int setup (const CELLS<Dimension, Nrays> &cells,
                  const FREQUENCIES             &frequencies,
                  const TEMPERATURE             &temperature,
                  const LINES                   &lines,
                  const SCATTERING              &scattering,
	          const RADIATION               &radiation,
                  const long                     f           )
{

  vReal chi_o, term1_o, term2_o, eta_o;
  vReal chi_c, term1_c, term2_c;
  vReal chi_n, term1_n, term2_n;


  if ( (raydata_ar.n > 0) && (raydata_r.n > 0) )
  {
    raydata_ar.get_terms_chi (frequencies, temperature, lines, scattering, radiation, f,
        ncells, term1_o, term2_o, dtau);

    fill_ar (frequencies, temperature, lines, scattering, radiation, f);

    fill_r  (frequencies, temperature, lines, scattering, radiation, f);
  }

  else if (raydata_ar.n > 0) // and hence raydata_r.n == 0
  {
    raydata_ar.get_terms_chi (frequencies, temperature, lines, scattering, radiation, f,
        ncells, term1_c, term2_c, dtau);

    fill_ar (frequencies, temperature, lines, scattering, radiation, f);


  }

  else if (raydata_r.n > 0) // and hence raydata_ar.n == 0
  {
    raydata_r.get_terms_chi (frequencies, temperature, lines, scattering, radiation, f,
        ncells, term1_c, term2_c, dtau);

    fill_r (frequencies, temperature, lines, scattering, radiation, f);


  }


  return (0);

}




inline int RAYPAIR ::
           fill_ar (const FREQUENCIES &frequencies,
                    const TEMPERATURE &temperature,
                    const LINES       &lines,
                    const SCATTERING  &scattering,
                    const RADIATION   &radiation,
                    const long         f           )
{

  // Reset current cell to origin

  term1_c = term1_o;
  term2_c = term2_o;


  for (long q = 0; q < raydata_ar.n-1; q++)
  {
    raydata_ar.get_terms_chi (frequencies, temperature, lines, scattering, radiation, f,
        q, term1_n, term2_n, dtau[raydata_ar.n-1-q]);
  
    Su[raydata_ar.n-1-q] = 0.5 * (term1_n + term1_c) + (term2_n - term2_c) / dtau[raydata_ar.n-1-q];
    Sv[raydata_ar.n-1-q] = 0.5 * (term2_n + term2_c) + (term1_n - term1_c) / dtau[raydata_ar.n-1-q];

    term1_c = term1_n;
    term2_c = term2_n;
  }


  raydata_ar.get_terms_chi_I_bdy (frequencies, temperature, lines, scattering, radiation, f,
      raydata_ar.n-1, term1_n, term2_n, dtau[0], Ibdy_scaled);

  Su[0] = 0.5 * (term1_n + term1_c) + (term2_n - term2_c) / dtau[0];
  Sv[0] = 0.5 * (term2_n + term2_c) + (term1_n - term1_c) / dtau[0];


  // Add boundary condition

  Su[0] += 2.0 / dtau[0] * (Ibdy_scaled - 0.5 * (term2_c + term2_n));
  Sv[0] += 2.0 / dtau[0] * (Ibdy_scaled - 0.5 * (term1_c + term1_n));


  return (0);

}


inline int RAYPAIR ::
           fill_r (const FREQUENCIES &frequencies,
                   const TEMPERATURE &temperature,
                   const LINES       &lines,
                   const SCATTERING  &scattering,
                   const RADIATION   &radiation,
                   const long         f           )
{

  // Reset current cell to origin

  term1_c = term1_o;
  term2_c = term2_o;


  for (long q = 0; q < raydata_r.n-1; q++)
  {
    raydata_r.get_terms_chi (frequencies, temperature, lines, scattering, radiation, f,
        q, term1_n, term2_n, dtau[raydata_ar.n+q]);
  
    Su[raydata_ar.n+q] = 0.5 * (term1_n + term1_c) + (term2_n - term2_c) / dtau[raydata_ar.n+q];
    Sv[raydata_ar.n+q] = 0.5 * (term2_n + term2_c) + (term1_n - term1_c) / dtau[raydata_ar.n+q];

    term1_c = term1_n;
    term2_c = term2_n;
  }


  raydata_r.get_terms_chi_Ibdy (frequencies, temperature, lines, scattering, radiation, f,
      raydata_r.n-1, term1_n, term2_n, dtau[ndep-1], Ibdy_scaled);

  Su[ndep-1] = 0.5 * (term1_n + term1_c) + (term2_n - term2_c) / dtau[ndep-1];
  Sv[ndep-1] = 0.5 * (term2_n + term2_c) + (term1_n - term1_c) / dtau[ndep-1];


  // Add boundary condition

  Su[ndep-1] += 2.0 / dtau[ndep-1] * (Ibdy_scaled - 0.5 * (term2_c + term2_n));
  Sv[ndep-1] += 2.0 / dtau[ndep-1] * (Ibdy_scaled - 0.5 * (term1_c + term1_n));


  return (0);

}



inline int RAYPAIR ::
           solve (void)

{

  vReal A[ncells];   // A coefficient in Feautrier recursion relation
  vReal C[ncells];   // C coefficient in Feautrier recursion relation
  vReal F[ncells];   // helper variable from Rybicki & Hummer (1991)
  vReal G[ncells];   // helper variable from Rybicki & Hummer (1991)

  vReal B0;          // B[0]
  vReal B0_min_C0;   // B[0] - C[0]
  vReal Bd;          // B[ndep-1]
  vReal Bd_min_Ad;   // B[ndep-1] - A[ndep-1]




  // SETUP FEAUTRIER RECURSION RELATION
  // __________________________________


  A[0] = 0.0;
  C[0] = 2.0/(dtau[0]*dtau[0]);

  B0        = vOne + 2.0/dtau[0] + 2.0/(dtau[0]*dtau[0]);
  B0_min_C0 = vOne + 2.0/dtau[0];

  for (long n = 1; n < ndep-1; n++)
  {
    A[n] = 2.0 / ((dtau[n] + dtau[n+1]) * dtau[n]);
    C[n] = 2.0 / ((dtau[n] + dtau[n+1]) * dtau[n+1]);
  }

  A[ndep-1] = 2.0/(dtau[ndep-1]*dtau[ndep-1]);
  C[ndep-1] = 0.0;

  Bd        = vOne + 2.0/dtau[ndep-1] + 2.0/(dtau[ndep-1]*dtau[ndep-1]);
  Bd_min_Ad = vOne + 2.0/dtau[ndep-1];




  // SOLVE FEAUTRIER RECURSION RELATION
  // __________________________________


  // Elimination step

  Su[0] = Su[0] / B0;
  Sv[0] = Sv[0] / B0;

  F[0] = B0_min_C0 / C[0];

  for (long n = 1; n < ndep-1; n++)
  {
    F[n] = (vOne + A[n]*F[n-1]/(vOne + F[n-1])) / C[n];

    Su[n] = (Su[n] + A[n]*Su[n-1]) / ((vOne + F[n]) * C[n]);
    Sv[n] = (Sv[n] + A[n]*Sv[n-1]) / ((vOne + F[n]) * C[n]);
  }

  Su[ndep-1] = (Su[ndep-1] + A[ndep-1]*Su[ndep-2]) * (vOne + F[ndep-2])
               / (Bd_min_Ad + Bd*F[ndep-2]);

  Sv[ndep-1] = (Sv[ndep-1] + A[ndep-1]*Sv[ndep-2])
               / (Bd_min_Ad + Bd*F[ndep-2]) * (vOne + F[ndep-2]);

  G[ndep-1] = Bd_min_Ad / A[ndep-1];


  // Back substitution

  for (long n = ndep-2; n > 0; n--)
  {
    Su[n] = Su[n] + Su[n+1] / (vOne + F[n]);
    Sv[n] = Sv[n] + Sv[n+1] / (vOne + F[n]);

    G[n] = (vOne + C[n] * G[n+1] / (vOne+G[n+1])) / A[n];
  }

  Su[0] = Su[0] + Su[1] / (vOne + F[0]);
  Sv[0] = Sv[0] + Sv[1] / (vOne+F[0]);



  // CALCULATE LAMBDA OPERATOR
  // _________________________


  // Calculate diagonal elements

  ////Lambda[f](0,0) = (1.0 + G[1][f]) / (B0_min_C0[f] + B0[f]*G[1][f]);
  //Lambda[0] = (vOne + G[1]) / (B0_min_C0 + B0*G[1]);

  //for (long n = 1; n < ndep-1; n++)
  //{
  //  //Lambda[f](n,n) = (1.0 + G[n+1][f]) / ((F[n][f] + G[n+1][f] + F[n][f]*G[n+1][f]) * C[n][f]);
  //  Lambda[n] = (vOne + G[n+1]) / ((F[n] + G[n+1] + F[n]*G[n+1]) * C[n]);
  //}

  ////Lambda[f](ndep-1,ndep-1) = (1.0 + F[ndep-2][f]) / (Bd_min_Ad[f] + Bd[f]*F[ndep-2][f]);
  //Lambda[ndep-1] = (vOne + F[ndep-2]) / (Bd_min_Ad + Bd*F[ndep-2]);


  //// Add upper-diagonal elements

  //for (long m = 1; m < ndiag; m++)
  //{
  //  for (long n = 0; n < ndep-m; n++)
  //  {
  //    for (long f = 0; f < nfreq_red; f++)
	//		{
  //      Lambda[f](n,n+m) = Lambda[f](n+1,n+m) / (1.0 + F[n][f]);
	//		}
  //  }
  //}


  //// Add lower-diagonal elements

  //for (long m = 1; m < ndiag; m++)
  //{
  //  for (long n = m; n < ndep; n++)
  //  {
  //    for (long f = 0; f < nfreq_red; f++)
  //    {
  //      Lambda[f](n,n-m) = Lambda[f](n-1,n-m) / (1.0 + G[n][f]);
	//	  }
  //  }
  //}


  return (0);

}
