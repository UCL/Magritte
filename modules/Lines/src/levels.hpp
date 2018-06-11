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
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"


struct LEVELS
{
	
  long ncells;                                 ///< number of cells
	
	int nlspec;                                  ///< number of species producing lines

	vector<int> nlev;                            ///< number of levels per species
	vector<int> nrad;                            ///< number of radiative transitions per species
	
	vector<long> nlev_tot;                       ///< total number of levels per species    

  bool some_not_converged;                     ///< true when there are unconverged species

	vector<bool>            not_converged;       ///< true when species is not converged
  vector<double> fraction_not_converged;       ///< fraction of levels that is not converged

	vector<vector<VectorXd>> population;         ///< level population (most recent)

	vector<vector<vector<double>>> J_eff;        ///< effective mean intensity

	vector<vector<double>>   population_tot;     ///< total level population (sum over levels)

	vector<vector<VectorXd>> population_prev1;   ///< level populations 1 iteration back
	vector<vector<VectorXd>> population_prev2;   ///< level populations 2 iterations back
	vector<vector<VectorXd>> population_prev3;   ///< level populations 3 iterations back


  LEVELS (long num_of_cells, LINEDATA linedata);   ///< Constructor


	int set_LTE_populations (LINEDATA linedata, SPECIES species,
			                     TEMPERATURE temperature);

  int calc_J_eff (FREQUENCIES& frequencies, TEMPERATURE& temperature,
			            const vector<vector<double>>& J,
  			          const long p, const int l, const int k);


	int update_using_statistical_equilibrium (const MatrixXd& R, const long p, const int l);

	int update_using_Ng_acceleration ();

	int update_previous_populations ();

	int check_for_convergence (const long p, const int l);

};


#endif // __LEVELS_HPP_INCLUDED__
