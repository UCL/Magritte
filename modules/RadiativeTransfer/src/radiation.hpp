// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________
 

#ifndef __RADIATION_HPP_INCLUDED__
#define __RADIATION_HPP_INCLUDED__


///  RADIATION: data structure for the radiation field
//////////////////////////////////////////////////////

struct RADIATION
{

	long ncells;           ///< number of cells
	long nrays;            ///< number of rays
  long nfreq;            ///< number of frequency bins

	double *frequencies;   ///< considered frequencies

	double *U_d;           ///< U scattered intensity
	double *V_d;           ///< V scattered intensity


	RADIATION (long ncells, long nrays, long nfreq);   ///< Constructor

	~RADIATION ();                                     ///< Destructor


	double U (long p, long r, double nu);   ///< U frequency interpolator

	double V (long p, long r, double nu);   ///< V frequency interpolator

};


#endif // __RADIATION_HPP_INCLUDED__
