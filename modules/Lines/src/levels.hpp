// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LEVELS_HPP_INCLUDED__
#define __LEVELS_HPP_INCLUDED__


#include <string>
#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "linedata.hpp"
#include "RadiativeTransfer/src/types.hpp"
#include "RadiativeTransfer/src/lines.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"


struct LEVELS
{
	
  const long ncells;                ///< number of cells
	
	const int nlspec;                 ///< number of species producing lines

	const Int1 nlev;                  ///< number of levels per species
	const Int1 nrad;                  ///< number of radiative transitions per species


  bool some_not_converged;          ///< true when there are unconverged species

	Bool1            not_converged;   ///< true when species is not converged
  Double1 fraction_not_converged;   ///< fraction of levels that is not converged

	VectorXd2 population;             ///< level population (most recent)

	Double3 J_eff;                    ///< effective mean intensity

  Double2   population_tot;         ///< total level population (sum over levels)

	VectorXd2 population_prev1;       ///< level populations 1 iteration back
	VectorXd2 population_prev2;       ///< level populations 2 iterations back
	VectorXd2 population_prev3;       ///< level populations 3 iterations back


  LEVELS (const long num_of_cells, const LINEDATA& linedata);   ///< Constructor


	int iteration_using_LTE (const LINEDATA& linedata, const SPECIES& species,
	  	                     const TEMPERATURE& temperature, LINES& lines);


	int update_using_LTE (const LINEDATA& linedata, const SPECIES& species,
			                  const TEMPERATURE& temperature, const long p, const int l);


	int update_using_Ng_acceleration ();


	int update_using_statistical_equilibrium (const MatrixXd& R, const long p, const int l);


	// Communication with Radiative Transfer module
	
	int calc_line_emissivity_and_opacity (const LINEDATA& linedata, LINES& lines,
			                                  const long p, const int l) const;

  int calc_J_eff (FREQUENCIES& frequencies, const TEMPERATURE& temperature,
			            vDouble2& J, const long p, const int l);

	// Convergence

	int check_for_convergence (const long p, const int l);

};


#endif // __LEVELS_HPP_INCLUDED__
