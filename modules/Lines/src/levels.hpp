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


struct LEVELS
{
	
  long ncells;

	vector<vector<VectorXd>> population;

	vector<vector<VectorXd>> population_prev1;
	vector<vector<VectorXd>> population_prev2;
	vector<vector<VectorXd>> population_prev3;


  LEVELS (long num_of_cells, LINEDATA linedata);   ///< Constructor

	int set_LTE_populations (LINEDATA linedata, SPECIES species, TEMPERATURE temperature);

};


#endif // __LEVELS_HPP_INCLUDED__
