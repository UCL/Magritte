// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________
 

#ifndef __SCATTERING_HPP_INCLUDED__
#define __SCATTERING_HPP_INCLUDED__


#include <vector>
using namespace std;


struct SCATTERING
{

	long ncells;                                 ///< number of cells
	long nrays;                                  ///< number of rays
	long nfreq_scat;                             ///< number of frequencies in scattering data

	vector<double> opacity_scat;                 ///< scattering opacity (p,f)

	vector<vector<vector<double>>> phase_scat;   ///< scattering phase function (r1,r2,f)


  SCATTERING (long num_of_rays, long num_of_freq_scat);   ///< Constructor   
	
  int add_opacity (vector<double> frequencies, vector<double>& chi);

};


#endif // __SCATTERING_HPP_INCLUDED__
