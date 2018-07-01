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

	Double3 emissivity;   ///< line emissivity (p,l,k)
	Double3 opacity;      ///< line opacity (p,l,k)


	Double1 emissivity_vec;   ///< line emissivity (p,l,k)
	Double1 opacity_vec;      ///< line opacity (p,l,k)


  LINES (const long num_of_cells, const LINEDATA& linedata);   ///< Constructor

	static Int1 get_nrad_cum (const Int1 nrad);
	static int  get_nrad_tot (const Int1 nrad);

  long index (const long p, const int l, const int k) const;

  int add_emissivity_and_opacity (FREQUENCIES& frequencies, const TEMPERATURE& temperature,
																 	vReal1& frequencies_scaled, const long p,
			                            vReal1& eta, vReal1& chi) const;

	int mpi_allgatherv ();

};


#endif // __LINES_HPP_INCLUDED__
