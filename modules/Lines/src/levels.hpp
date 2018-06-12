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
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"


struct LEVELS
{
	
  long ncells;                      ///< number of cells
	
	int nlspec;                       ///< number of species producing lines

	Int1 nlev;                        ///< number of levels per species
	Int1 nrad;                        ///< number of radiative transitions per species

	Long1 nlev_tot;                   ///< total number of levels per species    

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


	int set_LTE_populations (const LINEDATA& linedata, const SPECIES& species,
			                     const TEMPERATURE& temperature);

  int calc_J_eff (const FREQUENCIES& frequencies, const TEMPERATURE& temperature,
			            const Double2& J, const long p, const int l);


	int update_using_statistical_equilibrium (const MatrixXd& R, const long p, const int l);

	int update_using_Ng_acceleration ();

	int check_for_convergence (const long p, const int l);

};


#endif // __LEVELS_HPP_INCLUDED__
