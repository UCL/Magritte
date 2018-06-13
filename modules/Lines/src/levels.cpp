// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <omp.h>
#include <iostream>
using namespace std;
#include <Eigen/QR>
using namespace Eigen;

#include "levels.hpp"
#include "linedata.hpp"
#include "RadiativeTransfer/src/constants.hpp"
#include "RadiativeTransfer/src/profile.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"


#define POP_PREC 1.0E-4


///  Constructor for LEVELS
///////////////////////////

LEVELS :: LEVELS (const long num_of_cells, const LINEDATA& linedata)
{
  
	ncells = num_of_cells;
  nlspec = linedata.nlspec;
  nlev   = linedata.nlev;
  nrad   = linedata.nrad;

  some_not_converged = true;

	         not_converged.resize (nlspec);
  fraction_not_converged.resize (nlspec);

	nlev_tot.resize (nlspec);


  for (int l = 0; l < nlspec; l++)
	{	
             not_converged[l] = true;
    fraction_not_converged[l] = 0.0;
	}


  population.resize (ncells);
	     J_eff.resize (ncells);
	
  population_tot.resize (ncells);

  population_prev1.resize (ncells);
  population_prev2.resize (ncells);
  population_prev3.resize (ncells);

	
# pragma omp parallel   \
	shared (linedata)     \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*ncells)/num_threads;
  long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    population[p].resize (nlspec);
		     J_eff[p].resize (nlspec);
		
    population_tot[p].resize (nlspec);

    population_prev1[p].resize (nlspec);
    population_prev2[p].resize (nlspec);
    population_prev3[p].resize (nlspec);


		for (int l = 0; l < nlspec; l++)
		{
			population[p][l].resize (nlev[l]);
			     J_eff[p][l].resize (nrad[l]);

			population_tot[p][l] = 0.0;

			population_prev1[p][l].resize (nlev[l]);
			population_prev2[p][l].resize (nlev[l]);
			population_prev3[p][l].resize (nlev[l]);
		}

	}
	} // end of pragma omp parallel


}   // END OF CONSTRUCTOR




int LEVELS ::
    set_LTE_populations (const LINEDATA& linedata, const SPECIES& species,
		                     const TEMPERATURE& temperature, const long p, const int l)
{

 	// Set population total

 	population_tot[p][l] = species.density[p] * species.abundance[p][linedata.num[l]];


   // Calculate fractional LTE level populations and partition function

   double partition_function = 0.0;

   for (int i = 0; i < linedata.nlev[l]; i++)
   {
     population[p][l](i) = linedata.weight[l](i)
 			                    * exp( -linedata.energy[l](i) / (KB*temperature.gas[p]) );

     partition_function += population[p][l](i);
   }


   // Rescale (normalize) LTE level populations

   for (int i = 0; i < linedata.nlev[l]; i++)
   {
     population[p][l](i) *= population_tot[p][l] / partition_function;
   }


  return (0);

}




int LEVELS ::
    update_using_statistical_equilibrium (const MatrixXd& R, const long p, const int l)
{


  // Statitstical equilibrium requires sum_j ( n_j R_ji - n_i R_ij) = 0 for all i
	
	MatrixXd M = R.transpose();
  VectorXd y = VectorXd :: Zero (nlev[l]);   


	for (int i = 0; i < nlev[l]; i++)
	{
		double R_i = 0.0;

  	for (int j = 0; j < nlev[l]; j++)
		{
      R_i += R(i,j);  
		}

		M(i,i) -= R_i;
	}


	// Replace last row with conservation equation

	for (int j = 0; j < nlev[l]; j++)
	{
		M(nlev[l]-1,j) = 1.0;
	}

	y(nlev[l]-1) = population_tot[p][l];


  // Solve matrix equation M*x=y for x
	
  population[p][l] = M.householderQr().solve(y);


	return (0);

}




int LEVELS ::
    check_for_convergence (const long p, const int l)
{

  // Start by assuming that the populations are converged

           not_converged[l] = false;
  fraction_not_converged[l] = 0.0;


	// Check whether they are indeed converged

	VectorXd dpop = population[p][l] - population_prev1[p][l];
	VectorXd spop = population[p][l] + population_prev1[p][l];

  double min_pop = 1.0E-10 * population_tot[p][l];


  for (int i = 0; i < nlev[l]; i++)
  {
    if (population[p][l](i) > min_pop)
    {
      double relative_change = 2.0 * fabs(dpop(i) / spop(i));

      if (relative_change > POP_PREC)
      {
        not_converged[l] = true;
			
        fraction_not_converged[l] += 1.0/(ncells*nlev[l]);
      }
    }
  }


	return (0);
}




///  calc_J_eff: calculate the effective mean intensity in a line
///    @param[in] frequencies: data structure containing frequencies
///    @param[in] temperature: data structure containing temperatures
///    @param[in] J: (angle averaged) mean intensity for all frequencies 
///    @param[in] p: number of the cell under consideration
///    @param[in] l: number of the line producing species under consideration
/////////////////////////////////////////////////////////////////////////////
 
int LEVELS ::
    calc_J_eff (const FREQUENCIES& frequencies, const TEMPERATURE& temperature,
				        const Double2& J, const long p, const int l)
{

	for (int k = 0; k < nrad[l]; k++)
	{
    const vector<long> freq_nrs = frequencies.nr_line[p][l][k];

    const double freq_line = 0.5 * (   frequencies.all[p][freq_nrs[1]]
                                     + frequencies.all[p][freq_nrs[2]] );

    J_eff[p][l][k] = 0.0;

    for (long z = 0; z < 4; z++)
    {
      J_eff[p][l][k] += H_4_weights[z] / profile_width (temperature.gas[p], freq_line) * J[p][freq_nrs[z]];
    }
  }


  return (0);

}
