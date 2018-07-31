// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
#include <vector>
#include <iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "catch.hpp"
#include "tools.hpp"

#include "../src/levels.hpp"

#define EPS 1.0E-7


TEST_CASE ("LEVELS constructor")
{

	const long ncells = 50;

	LINEDATA linedata;

	LEVELS levels (ncells, linedata);

	cout << "ncells is " << levels.ncells << endl;
	cout << "nrad[nlspec-1] is " << levels.nrad[levels.nlspec-1] << endl;

	CHECK (true);

}




TEST_CASE ("LEVELS update_using_statistical_equilibrium")
{

	const long ncells = 1;
	const long p = 0;
	const int  l = 0;


	LINEDATA linedata;

	LEVELS levels (ncells, linedata);


	// Set up populations

	levels.population_tot[p][l] = 0.0;

	for (int i = 0; i < linedata.nlev[l]; i++)
	{
    levels.population[p][l](i) = 2.0/(i+1);

		levels.population_tot[p][l] += levels.population[p][l](i);
	}


	// Solve statistical equilibrium eqquation

	MatrixXd R = MatrixXd :: Random (linedata.nlev[l],linedata.nlev[l]);

	levels.update_using_statistical_equilibrium (R, p, l);


	// Check conservation of populations total

	double tot = 0.0;

	for (int i = 0; i < linedata.nlev[l]; i++)
	{
		tot += levels.population[p][l](i);
	}

	CHECK (relative_error (levels.population_tot[p][l], tot) == Approx(0.0).epsilon(EPS));


	// Setup both sides of the stat. equil. equation

	VectorXd lhs = levels.population[p][l].transpose() * R;
	VectorXd rhs = VectorXd :: Zero(linedata.nlev[l]);

	for (int i = 0; i < linedata.nlev[l]; i++)
	{
		double R_i = 0.0;

	  for (int j = 0; j < linedata.nlev[l]; j++)
		{
		  R_i	+= R(i,j);
		}

		rhs(i) = levels.population[p][l](i) * R_i;
	}


	// Check whether the stat. equil. equation is satisfied

	for (int i = 0; i < linedata.nlev[l]; i++)
	{
		CHECK (relative_error(lhs(i), rhs(i)) == Approx(0.0).epsilon(EPS));
	}

}
