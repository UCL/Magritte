// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________
 

#ifndef __SCATTERING_HPP_INCLUDED__
#define __SCATTERING_HPP_INCLUDED__


#include <vector>
using namespace std;

#include "types.hpp"
#include "GridTypes.hpp"


struct SCATTERING
{

	long ncells;            ///< number of cells
	long nrays;             ///< number of rays
	long nfreq_scat;        ///< number of frequencies in scattering data

	Double1 opacity_scat;   ///< scattering opacity (p,f)

	Double3 phase_scat;     ///< scattering phase function (r1,r2,f)


  SCATTERING (const long num_of_rays, const long num_of_freq_scat);   ///< Constructor   
	
  int add_opacity (const vDouble1& frequencies, vDouble1& chi);

};


#endif // __SCATTERING_HPP_INCLUDED__
