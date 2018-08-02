// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LINES_HPP_INCLUDED__
#define __LINES_HPP_INCLUDED__


#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "types.hpp"
#include "GridTypes.hpp"
#include "radiation.hpp"
#include "frequencies.hpp"
#include "temperature.hpp"
#include "Lines/src/linedata.hpp"


///  LINES: bridge between Levels and RadiativeTransfer calculations
////////////////////////////////////////////////////////////////////

struct LINES
{

	const long ncells;          ///< number of cells

	const long nlspec;          ///< number of species producing lines

	const Int1 nrad;
	const Int1 nrad_cum;
	const int  nrad_tot;

	Double1 emissivity;   ///< line emissivity (p,l,k)
	Double1 opacity;      ///< line opacity (p,l,k)


	static Int1 get_nrad_cum (const Int1 nrad);
	static int  get_nrad_tot (const Int1 nrad);


  LINES (const long num_of_cells, const LINEDATA& linedata);   ///< Constructor


	int print (string output_folder, string tag) const;

  inline long index (const long p, const int l, const int k) const;

  inline long index (const long p, const long line_index) const;

  inline int add_emissivity_and_opacity (FREQUENCIES& frequencies,
		                                     const TEMPERATURE& temperature,
																 	       vReal& freq_scaled, long& lnotch,
																				 const long p,
																	       vReal& eta, vReal& chi) const;

	int mpi_allgatherv ();

};




#include "profile.hpp"




///  index:
///////////

inline long LINES ::
            index (const long p, const int l, const int k) const
{
	return k + nrad_cum[l] + p*nrad_tot;
}



///  index:
///////////

inline long LINES ::
            index (const long p, const long line_index) const
{
	return line_index + p*nrad_tot;
}




///  add_emissivity_and_opacity
///////////////////////////////

inline int LINES ::
           add_emissivity_and_opacity (FREQUENCIES& frequencies,
						                           const TEMPERATURE& temperature,
		                                   vReal& freq_scaled,
																			 long& lnotch, const long p,
																       vReal& eta, vReal& chi) const
{

	vReal  freq_diff = freq_scaled - (vReal) frequencies.line[lnotch];
	double     width = profile_width (temperature.gas[p], frequencies.line[lnotch]);


# if (GRID_SIMD)
		while (   (freq_diff.getlane(0) > 3.0*width)
		       && (lnotch < nrad_tot) )
# else
		while (   (freq_diff            > 3.0*width)
		       && (lnotch < nrad_tot) )
# endif
	{
	  freq_diff = freq_scaled - (vReal) frequencies.line[lnotch];
	      width = profile_width (temperature.gas[p], frequencies.line[lnotch]);

		lnotch++;
	}


  if (lnotch < nrad_tot)
	{
		vReal line_profile = profile (width, freq_diff);
		long           ind = index   (p, frequencies.line_index[lnotch]);

		eta += emissivity[ind] * line_profile;
		chi +=    opacity[ind] * line_profile;


	  //for (int lane = 0; lane < n_simd_lanes; lane++)
	  //{
	  //	if (isnan(chi.getlane(lane)))
	  //	{
	  //		cout << line_profile << " " << width << " " << temperature.gas[p] << " " << frequencies.line[lnotch] << endl;
	  //  }
	  //}


	}




	long lindex = lnotch+1;

# if (GRID_SIMD)
	  while (   (freq_diff.getlane(n_simd_lanes-1) > -3.0*width)
		       && (lindex < nrad_tot) )
# else
	  while (   (freq_diff                         > -3.0*width)
		       && (lindex < nrad_tot) )
# endif
	{
	  freq_diff = freq_scaled - (vReal) frequencies.line[lindex];
	      width = profile_width (temperature.gas[p], frequencies.line[lindex]);

	  vReal line_profile = profile (width, freq_diff);
	  long           ind = index   (p, frequencies.line_index[lindex]);

	  eta += emissivity[ind] * line_profile;
	  chi +=    opacity[ind] * line_profile;


	  //for (int lane = 0; lane < n_simd_lanes; lane++)
	  //{
	  //	if (isnan(chi.getlane(lane)))
	  //	{
	  //		cout << line_profile << " " << width << " " << temperature.gas[p] << " " << frequencies.line[lnotch] << endl;
	  //	}
	  //}

		lindex++;
	}



	return (0);

}


#endif // __LINES_HPP_INCLUDED__
