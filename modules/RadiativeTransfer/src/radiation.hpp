// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RADIATION_HPP_INCLUDED__
#define __RADIATION_HPP_INCLUDED__


#include "frequencies.hpp"
#include "GridTypes.hpp"
#include "scattering.hpp"
#include "cells.hpp"


///  RADIATION: data structure for the radiation field
//////////////////////////////////////////////////////

struct RADIATION
{

	const long ncells;          ///< number of cells
	const long nrays;           ///< number of rays
	const long nrays_red;       ///< reduced number of rays
	const long nfreq_red;       ///< reduced number of frequencies
	const long nboundary;       ///< number of boundary cells
	const long START_raypair;   ///< reduced number of frequencies


	vReal2 u;   ///< u intensity
	vReal2 v;   ///< v intensity

	vReal2 U;   ///< U scattered intensity
	vReal2 V;   ///< V scattered intensity

	vReal1 J;   ///< (angular) mean intensity

	vReal3 boundary_intensity;


	RADIATION (const long num_of_cells,    const long num_of_rays,
			       const long num_of_rays_red, const long num_of_freq_red,
						 const long num_of_bdycells, const long START_raypair_input);


	//int initialize ();

  int read (const string boundary_intensity_file);

	int write (const string boundary_intensity_file) const;


  long index (const long r, const long p, const long f) const;

  long index (const long p, const long f) const;


  int calc_boundary_intensities (const Long1& bdy_to_cell_nr,
			                           const FREQUENCIES& frequencies);



  int rescale_U_and_V (FREQUENCIES& frequencies, const long p,
	                     const long R, long& notch, vReal& freq_scaled,
						           vReal& U_scaled, vReal& V_scaled);

	int calc_J (void);

	int calc_U_and_V (const SCATTERING& scattering);

	// Print

	int print (string output_folder, string tag);


};


#endif // __RADIATION_HPP_INCLUDED__
