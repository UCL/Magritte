// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "GridTypes.hpp"
#include <iostream>


///  read: read the cells, neighbors and boundary files
///////////////////////////////////////////////////////

template <int Dimension, long Nrays>
inline void RAYPAIR ::
    initialize                                    (
        const CELLS<Dimension,Nrays> &cells,
        const TEMPERATURE            &temperature,
        const long                    o           )
{
  
  // initialize the ray and its antipodal

  raydata_r.initialize <Dimension, Nrays> (cells, temperature, o);
  raydata_ar.initialize <Dimension, Nrays> (cells, temperature, o);


  // Set total number of depth points

  ndep = raydata_r.n + raydata_ar.n + 1;

  if (ndep > Lambda.size())
  {
    Lambda.resize (ndep+10);
     term1.resize (ndep+10);
     term2.resize (ndep+10);
      dtau.resize (ndep+10);
        Su.resize (ndep+10);
        Sv.resize (ndep+10);
         A.resize (ndep+10);
         C.resize (ndep+10);
         F.resize (ndep+10);
         G.resize (ndep+10);
  }

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

inline void RAYPAIR ::
    setup                              (
        const FREQUENCIES &frequencies,
        const TEMPERATURE &temperature,
        const LINES       &lines,
        const SCATTERING  &scattering,
        const long         f           )
{


  if ( (raydata_ar.n > 0) && (raydata_r.n > 0) )
  {
    raydata_ar.set_current_to_origin (frequencies, temperature, lines, scattering, f);

    term1[raydata_ar.n] = raydata_ar.term1;
    term2[raydata_ar.n] = raydata_ar.term2; 

    fill_ar (frequencies, temperature, lines, scattering, f);

    raydata_r.chi_n = raydata_ar.chi_o;

    fill_r  (frequencies, temperature, lines, scattering, f);
  }

  else if (raydata_ar.n > 0) // and hence raydata_r.n == 0
  {
    // Get boundary condition at origin
    
    raydata_ar.set_current_to_origin_bdy (frequencies, temperature, lines, scattering, f);

    term1[ndep-1] = raydata_ar.term1;
    term2[ndep-1] = raydata_ar.term2; 

    Ibdy_n = raydata_ar.Ibdy_scaled;


    fill_ar (frequencies, temperature, lines, scattering, f);
  }

  else if (raydata_r.n > 0) // and hence raydata_ar.n == 0
  {
    // Get boundary condition at origin
    
    raydata_r.set_current_to_origin_bdy (frequencies, temperature, lines, scattering, f);

    term1[0] = raydata_r.term1;
    term2[0] = raydata_r.term2; 

    Ibdy_0 = raydata_r.Ibdy_scaled;


    fill_r  (frequencies, temperature, lines, scattering, f);
  }


}




inline void RAYPAIR ::
    fill_ar (
        const FREQUENCIES &frequencies,
        const TEMPERATURE &temperature,
        const LINES       &lines,
        const SCATTERING  &scattering,
        const long         f           )
{

  for (long q = 0; q < raydata_ar.n-1; q++)
  {
    raydata_ar.compute_next (frequencies, temperature, lines, scattering, f, q);

     dtau[raydata_ar.n-1-q] = raydata_ar.dtau; 
    term1[raydata_ar.n-1-q] = raydata_ar.term1; 
    term2[raydata_ar.n-1-q] = raydata_ar.term2; 
  }

  raydata_ar.compute_next_bdy (frequencies, temperature, lines, scattering, f);

   dtau[0] = raydata_ar.dtau; 
  term1[0] = raydata_ar.term1; 
  term2[0] = raydata_ar.term2; 

  Ibdy_0 = raydata_ar.Ibdy_scaled;

}


inline void RAYPAIR ::
    fill_r (
        const FREQUENCIES &frequencies,
        const TEMPERATURE &temperature,
        const LINES       &lines,
        const SCATTERING  &scattering,
        const long         f           )
{

  for (long q = 0; q < raydata_r.n-1; q++)
  {
    raydata_r.compute_next (frequencies, temperature, lines, scattering, f, q);

     dtau[raydata_ar.n  +q] = raydata_r.dtau; 
    term1[raydata_ar.n+1+q] = raydata_r.term1; 
    term2[raydata_ar.n+1+q] = raydata_r.term2; 
  }

  raydata_r.compute_next_bdy (frequencies, temperature, lines, scattering, f);

   dtau[ndep-2] = raydata_r.dtau; 
  term1[ndep-1] = raydata_r.term1; 
  term2[ndep-1] = raydata_r.term2; 

  Ibdy_n = raydata_r.Ibdy_scaled;

}




inline void RAYPAIR ::
    solve (void)

{

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

  Su[0] = term1[0] - (term2[1] + term2[0] - 2.0*Ibdy_0) / dtau[0];
  Sv[0] = term2[0] - (term1[1] + term1[0] - 2.0*Ibdy_0) / dtau[0];

  for (long n = 1; n < ndep-1; n++)
  {
    A[n] = 2.0 / ((dtau[n-1] + dtau[n]) * dtau[n-1]);
    C[n] = 2.0 / ((dtau[n-1] + dtau[n]) * dtau[n]);

    Su[n] = term1[n] - 2.0 * (term2[n+1] - term2[n-1]) / (dtau[n] + dtau[n-1]);
    Sv[n] = term2[n] - 2.0 * (term1[n+1] - term1[n-1]) / (dtau[n] + dtau[n-1]);
  }

  A[ndep-1] = 2.0/(dtau[ndep-2]*dtau[ndep-2]);
  C[ndep-1] = 0.0;

  Bd        = vOne + 2.0/dtau[ndep-2] + 2.0/(dtau[ndep-2]*dtau[ndep-2]);
  Bd_min_Ad = vOne + 2.0/dtau[ndep-2];

  Su[ndep-1] = term1[ndep-1] + (term2[ndep-1] + term2[ndep-2] + 2.0*Ibdy_n) / dtau[ndep-2];
  Sv[ndep-1] = term2[ndep-1] + (term1[ndep-1] + term1[ndep-2] - 2.0*Ibdy_n) / dtau[ndep-2];


  // Add third order terms of the boundary condition

//  Su[0] += (term1[1] - term1[0] + (Ibdy_0 - term2[0] - C[1]*term2[2] + (C[1]+A[1])*term2[1] - A[1]*term2[0]) * dtau[0]) / 3.0;
//  Sv[0] += (term2[1] - term2[0] + (Ibdy_0 - term1[0] - C[1]*term1[2] + (C[1]+A[1])*term1[1] - A[1]*term1[0]) * dtau[0]) / 3.0;

//  Su[ndep-1] += (term1[ndep-1] - term1[ndep-2] + (Ibdy_n - term2[ndep-1] - C[ndep-2]*term2[ndep-1] + (C[ndep-2]+A[ndep-2])*term2[ndep-2] - A[ndep-2]*term2[ndep-3]) * dtau[ndep-2]) / 3.0;
//  Sv[ndep-1] += (term2[ndep-1] - term2[ndep-2] + (Ibdy_n - term1[ndep-1] - C[ndep-2]*term1[ndep-1] + (C[ndep-2]+A[ndep-2])*term1[ndep-2] - A[ndep-2]*term1[ndep-3]) * dtau[ndep-2]) / 3.0;

  
  // Add fourth order terms of the boundary condition

  //vReal T0 = 3.0 / (dtau[0]      + dtau[1]      + dtau[2]     );
  //vReal Td = 3.0 / (dtau[ndep-2] + dtau[ndep-3] + dtau[ndep-4]);

  


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

  Sv[ndep-1] = (Sv[ndep-1] + A[ndep-1]*Sv[ndep-2]) * (vOne + F[ndep-2])
               / (Bd_min_Ad + Bd*F[ndep-2]);

  G[ndep-1] = Bd_min_Ad / A[ndep-1];


  // Back substitution

  for (long n = ndep-2; n > 0; n--)
  {
    Su[n] = Su[n] + Su[n+1] / (vOne + F[n]);
    Sv[n] = Sv[n] + Sv[n+1] / (vOne + F[n]);

    G[n] = (vOne + C[n] * G[n+1] / (vOne+G[n+1])) / A[n];
  }

  Su[0] = Su[0] + Su[1] / (vOne + F[0]);
  Sv[0] = Sv[0] + Sv[1] / (vOne + F[0]);



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


  //// Set number of off-diagonals to add
  //
  //const long ndiag = 0;

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

}

inline void RAYPAIR ::
    solve_ndep_is_1 (void)

{


  // SETUP FEAUTRIER RECURSION RELATION
  // __________________________________

  vReal inverse_dtau = 1.0 / dtau[0];

  vReal A0 = 2.0 * inverse_dtau * inverse_dtau;

  vReal        B0 = vOne + 2.0*inverse_dtau + 2.0*inverse_dtau*inverse_dtau;
  vReal B0_min_A0 = vOne + 2.0*inverse_dtau;
  vReal B0_pls_A0 = vOne + 2.0*inverse_dtau + 4.0*inverse_dtau*inverse_dtau;

  vReal inverse_denominator = 1.0 / (B0_min_A0 * B0_pls_A0);




  // SOLVE FEAUTRIER RECURSION RELATION
  // __________________________________


  vReal u0 = (B0*Su[0] + A0*Su[1]) * inverse_denominator;
  vReal u1 = (B0*Su[1] + A0*Su[0]) * inverse_denominator;

  Su[0] = u0;
  Su[1] = u1;

  vReal v0 = (B0*Sv[0] + A0*Sv[1]) * inverse_denominator;
  vReal v1 = (B0*Sv[1] + A0*Sv[0]) * inverse_denominator;
  
  Sv[0] = v0;
  Sv[1] = v1;


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


  //// Set number of off-diagonals to add
  //
  //const long ndiag = 0;

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

}

inline vReal RAYPAIR ::
    get_u_at_origin (void)
{
  return Su[raydata_ar.n];
}

inline vReal RAYPAIR ::
    get_v_at_origin (void)
{
  return Sv[raydata_ar.n];
}


inline vReal RAYPAIR ::
    get_I_p (void)
{
  return Su[ndep-1] + Sv[ndep-1];
}

inline vReal RAYPAIR ::
    get_I_m (void)
{
  return Su[0] - Sv[0];
}
