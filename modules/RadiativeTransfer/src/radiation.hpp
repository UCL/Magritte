// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RADIATION_HPP_INCLUDED__
#define __RADIATION_HPP_INCLUDED__


#include "frequencies.hpp"
#include "GridTypes.hpp"
#include "frequencies.hpp"
#include "scattering.hpp"


///  RADIATION: data structure for the radiation field
//////////////////////////////////////////////////////

struct RADIATION
{

	const long ncells;          ///< number of cells
	const long nrays;           ///< number of rays
	const long nrays_red;       ///< reduced number of rays
	const long nfreq_red;       ///< reduced number of frequencies
	const long nboundary;       ///< number of boundary cells


	vReal2 u;                   ///< u intensity           (r, index(p,f))
	vReal2 v;                   ///< v intensity           (r, index(p,f))

	vReal2 U;                   ///< U scattered intensity (r, index(p,f))
	vReal2 V;                   ///< V scattered intensity (r, index(p,f))

	vReal1 J;                   ///< (angular) mean intensity (index(p,f))

	vReal3 boundary_intensity;   ///< intensity at the boundary (b,r,f)


	RADIATION (const long num_of_cells,    const long num_of_rays,
			       const long num_of_freq_red, const long num_of_bdycells);

  static long get_nrays_red (const long nrays);


	//int initialize ();

  int read (const string boundary_intensity_file);

	int write (const string boundary_intensity_file) const;

  inline long index (const long p, const long f) const;


  int calc_boundary_intensities (const Long1& bdy_to_cell_nr,
			                           const FREQUENCIES& frequencies);


  inline int rescale_U_and_V (FREQUENCIES& frequencies, const long p,
	                            const long R, long& notch, vReal& freq_scaled,
						                  vReal& U_scaled, vReal& V_scaled);

  inline int rescale_U_and_V_and_bdy_I (FREQUENCIES& frequencies, const long p, const long b,
	                                      const long R, long& notch, vReal& freq_scaled,
															          vReal& U_scaled, vReal& V_scaled, vReal& Ibdy_scaled);

	int calc_J (void);

	int calc_U_and_V (const SCATTERING& scattering);

	// Print

	int print (string output_folder, string tag);


};


//#include "radiation.cpp"

inline long RADIATION ::
            index (const long p, const long f) const
{
	return f + p*nfreq_red;
}



#include "interpolation.hpp"

inline int RADIATION ::
           rescale_U_and_V (      FREQUENCIES &frequencies,
                            const long         p,
                            const long         R,
                                  long        &notch,
                                  vReal       &freq_scaled,
                                  vReal       &U_scaled,
                                  vReal       &V_scaled    )

#if (GRID_SIMD)

{

  vReal nu1, nu2, U1, U2, V1, V2;

  for (int lane = 0; lane < n_simd_lanes; lane++)
  {

    double freq = freq_scaled.getlane (lane);

    search_with_notch (frequencies.nu[p], notch, freq);

    const long f1    =  notch    / n_simd_lanes;
    const  int lane1 =  notch    % n_simd_lanes;

    const long f2    = (notch-1) / n_simd_lanes;
    const  int lane2 = (notch-1) % n_simd_lanes;

    //const double nu1 = frequencies.nu[p][f1].getlane(lane1);
    //const double nu2 = frequencies.nu[p][f2].getlane(lane2);
    
    //const double U1 = U[R][index(p,f1)].getlane(lane1);
    //const double U2 = U[R][index(p,f2)].getlane(lane2);
    
    //const double V1 = V[R][index(p,f1)].getlane(lane1);
    //const double V2 = V[R][index(p,f2)].getlane(lane2);
    
    //U_scaled.putlane(interpolate_linear (nu1, U1, nu2, U2, freq), lane);
    //V_scaled.putlane(interpolate_linear (nu1, V1, nu2, V2, freq), lane);
    
    nu1.putlane (frequencies.nu[p][f1].getlane (lane1), lane);
    nu2.putlane (frequencies.nu[p][f2].getlane (lane2), lane);
    
     U1.putlane (U[R][index(p,f1)].getlane (lane1), lane);
     U2.putlane (U[R][index(p,f2)].getlane (lane2), lane);
    
     V1.putlane (V[R][index(p,f1)].getlane (lane1), lane);
     V2.putlane (V[R][index(p,f2)].getlane (lane2), lane);
  }

  U_scaled = interpolate_linear (nu1, U1, nu2, U2, freq_scaled);
  V_scaled = interpolate_linear (nu1, V1, nu2, V2, freq_scaled);


  return (0);

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


  return (0);

}

#endif




inline int RADIATION ::
           rescale_U_and_V_and_bdy_I (      FREQUENCIES &frequencies,
		                      const long         p,
                                      const long         b,
                                      const long         R,
                                            long        &notch,
                                            vReal       &freq_scaled,
                                            vReal       &U_scaled,
                                            vReal       &V_scaled,
                                            vReal       &Ibdy_scaled )
   
#if (GRID_SIMD)

{

  vReal nu1, nu2, U1, U2, V1, V2, Ibdy1, Ibdy2;

  for (int lane = 0; lane < n_simd_lanes; lane++)
  {
    double freq = freq_scaled.getlane (lane);

    search_with_notch (frequencies.nu[p], notch, freq);

    const long f1    =  notch    / n_simd_lanes;
    const  int lane1 =  notch    % n_simd_lanes;

    const long f2    = (notch-1) / n_simd_lanes;
    const  int lane2 = (notch-1) % n_simd_lanes;

    //const double nu1 = frequencies.nu[p][f1].getlane(lane1);
    //const double nu2 = frequencies.nu[p][f2].getlane(lane2);
    
    //const double U1 = U[R][index(p,f1)].getlane(lane1);
    //const double U2 = U[R][index(p,f2)].getlane(lane2);
    
    //const double V1 = V[R][index(p,f1)].getlane(lane1);
    //const double V2 = V[R][index(p,f2)].getlane(lane2);
    
    //const double Ibdy1 = boundary_intensity[R][b][f1].getlane(lane1);
    //const double Ibdy2 = boundary_intensity[R][b][f2].getlane(lane2);
    
    //   U_scaled.putlane (interpolate_linear (nu1, U1,    nu2, U2,    freq), lane);
    //   V_scaled.putlane (interpolate_linear (nu1, V1,    nu2, V2,    freq), lane);
    //Ibdy_scaled.putlane (interpolate_linear (nu1, Ibdy1, nu2, Ibdy2, freq), lane);
    
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


   return (0);
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
  
  const double Ibdy1 = boundary_intensity[R][b][f1];
  const double Ibdy2 = boundary_intensity[R][b][f2];
  
     U_scaled = interpolate_linear (nu1, U1,    nu2, U2,    freq_scaled);
     V_scaled = interpolate_linear (nu1, V1,    nu2, V2,    freq_scaled);
  Ibdy_scaled = interpolate_linear (nu1, Ibdy1, nu2, Ibdy2, freq_scaled);


   return (0);

}

#endif



#endif // __RADIATION_HPP_INCLUDED__
