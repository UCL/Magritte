// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "GridTypes.hpp"


///  read: read the cells, neighbors and boundary files
///////////////////////////////////////////////////////

template <int Dimension, long Nrays>
inline void RAYPAIR ::
    initialize                              (
        const CELLS<Dimension,Nrays> &cells,
        const long                    o     )
{
  
  // initialize the ray and its antipodal

  raydata_r.initialize <Dimension, Nrays> (cells, o);
  raydata_ar.initialize <Dimension, Nrays> (cells, o);


  // Set total number of depth points

  ndep = raydata_r.n + raydata_ar.n;

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
    fill_ar (frequencies, temperature, lines, scattering, f);
    fill_r  (frequencies, temperature, lines, scattering, f);
  }

  else if (raydata_ar.n > 0) // and hence raydata_r.n == 0
  {
    fill_ar (frequencies, temperature, lines, scattering, f);


    // Add extra boundary condition at origin
    
    raydata_ar.set_current_to_origin_bdy (frequencies, temperature, lines, scattering, f);
    raydata_ar.compute_next (frequencies, temperature, lines, scattering, f, 0);

    Su[ndep-1] += raydata_ar.get_boundary_term_Su_r();
    Sv[ndep-1] += raydata_ar.get_boundary_term_Sv_r();
  }

  else if (raydata_r.n > 0) // and hence raydata_ar.n == 0
  {
    fill_r (frequencies, temperature, lines, scattering, f);


    // Add extra boundary condition at origin
    
    raydata_r.set_current_to_origin_bdy (frequencies, temperature, lines, scattering, f);
    raydata_r.compute_next (frequencies, temperature, lines, scattering, f, 0);

    Su[0] += raydata_r.get_boundary_term_Su_ar();
    Sv[0] += raydata_r.get_boundary_term_Sv_ar();
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

  raydata_ar.set_current_to_origin (frequencies, temperature, lines, scattering, f);

  for (long q = 0; q < raydata_ar.n-1; q++)
  {
    raydata_ar.compute_next (frequencies, temperature, lines, scattering, f, q);

    dtau[raydata_ar.n-1-q] = raydata_ar.dtau; 
      Su[raydata_ar.n-1-q] = raydata_ar.get_Su_ar();
      Sv[raydata_ar.n-1-q] = raydata_ar.get_Sv_ar();

    raydata_ar.set_current_to_next();
    
    //if (f == frequencies.nr_line[raydata_ar.origin][0][15][20])
    //{
    //  cout << "dtau " << dtau[raydata_ar.n-1-q] << "   " << "Sv " << Sv[raydata_ar.n-1-q] << endl;  
    //}
  }

  raydata_ar.compute_next_bdy (frequencies, temperature, lines, scattering, f);

  dtau[0] = raydata_ar.dtau; 
    Su[0] = raydata_ar.get_Su_ar() + raydata_ar.get_boundary_term_Su_ar();
    Sv[0] = raydata_ar.get_Sv_ar() + raydata_ar.get_boundary_term_Sv_ar();

}


inline void RAYPAIR ::
            fill_r (
                const FREQUENCIES &frequencies,
                const TEMPERATURE &temperature,
                const LINES       &lines,
                const SCATTERING  &scattering,
                const long         f           )
 {

  raydata_r.set_current_to_origin (frequencies, temperature, lines, scattering, f);

  for (long q = 0; q < raydata_r.n-1; q++)
  {
    raydata_r.compute_next (frequencies, temperature, lines, scattering, f, q);

    dtau[raydata_ar.n+q] = raydata_r.dtau; 
      Su[raydata_ar.n+q] = raydata_r.get_Su_r();
      Sv[raydata_ar.n+q] = raydata_r.get_Sv_r();

    //if (f == frequencies.nr_line[raydata_ar.origin][0][15][20])
    //{
    //  cout << "dtau " << dtau[raydata_ar.n+q] << "   " << "Sv " << Sv[raydata_ar.n+q] << endl;  
    //}
    raydata_r.set_current_to_next();
  }

  raydata_r.compute_next_bdy (frequencies, temperature, lines, scattering, f);

  dtau[ndep-1] = raydata_r.dtau; 
    Su[ndep-1] = raydata_r.get_Su_r() + raydata_r.get_boundary_term_Su_r();
    Sv[ndep-1] = raydata_r.get_Sv_r() + raydata_r.get_boundary_term_Sv_r();

}




inline void RAYPAIR ::
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
inline void RAYPAIR ::
    compute_u_and_v_at_origin (void)
{

  if ( (raydata_ar.n > 0) && (raydata_r.n > 0) )
  {
    u_at_origin = 0.5 * (Su[raydata_ar.n-1] + Su[raydata_ar.n]);
    v_at_origin = 0.5 * (Sv[raydata_ar.n-1] + Sv[raydata_ar.n]);
  }

  else if (raydata_r.n == 0)   // and hence n_ar > 0
  {
    u_at_origin = Su[ndep-1];
    v_at_origin = Sv[ndep-1];
  }

  else if (raydata_ar.n == 0)   // and hence n_r > 0
  {
    u_at_origin = Su[0];
    v_at_origin = Sv[0];
  }

}


inline vReal RAYPAIR ::
    get_I_p (void)
{
//  if ( (raydata_ar.n > 0) && (raydata_r.n > 0) )
//  {
//    u_at_origin = 0.5 * (Su[raydata_ar.n-1] + Su[raydata_ar.n]);
//    v_at_origin = 0.5 * (Sv[raydata_ar.n-1] + Sv[raydata_ar.n]);
//  }
//
//  else if (raydata_r.n == 0)   // and hence n_ar > 0
//  {
//    u_at_origin = Su[ndep-1];
//    v_at_origin = Sv[ndep-1];
//  }
//
//  else if (raydata_ar.n == 0)   // and hence n_r > 0
//  {
//    u_at_origin = Su[0];
//    v_at_origin = Sv[0];
//  }
//
//  return  u_at_origin + v_at_origin;
  return Su[ndep-1] ;//+ Sv[ndep-1];
}

inline vReal RAYPAIR ::
    get_I_m (void)
{
//  if ( (raydata_ar.n > 0) && (raydata_r.n > 0) )
//  {
//    u_at_origin = 0.5 * (Su[raydata_ar.n-1] + Su[raydata_ar.n]);
//    v_at_origin = 0.5 * (Sv[raydata_ar.n-1] + Sv[raydata_ar.n]);
//
//  }
//
//  else if (raydata_r.n == 0)   // and hence n_ar > 0
//  {
//    u_at_origin = Su[ndep-1];
//    v_at_origin = Sv[ndep-1];
//  }
//
//  else if (raydata_ar.n == 0)   // and hence n_r > 0
//  {
//    u_at_origin = Su[0];
//    v_at_origin = Sv[0];
//  }
//
//  return  u_at_origin - v_at_origin;
  return Su[0] ;// - Sv[0];
}
