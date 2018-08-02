// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
using namespace std;
#include <Eigen/QR>
using namespace Eigen;

#include "levels.hpp"
#include "linedata.hpp"
#include "RadiativeTransfer/src/constants.hpp"
#include "RadiativeTransfer/src/GridTypes.hpp"
#include "RadiativeTransfer/src/types.hpp"
#include "RadiativeTransfer/src/lines.hpp"
#include "RadiativeTransfer/src/profile.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"


#define POP_PREC 1.0E-4


///  Constructor for LEVELS
///////////////////////////

LEVELS :: LEVELS (const long num_of_cells, const LINEDATA& linedata)
  : ncells (num_of_cells)
  , nlspec (linedata.nlspec)
  ,	nlev   (linedata.nlev)
  ,	nrad   (linedata.nrad)
{

  some_not_converged = true;

	         not_converged.resize (nlspec);
  fraction_not_converged.resize (nlspec);


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
    print (string output_folder, string tag)
{

	int world_rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);


	if (world_rank == 0)
	{
	  for (int l = 0; l < nlspec; l++)
	  {
			string pops_file = output_folder + "populations_" + to_string (l) + tag + ".txt";
			string Jeff_file = output_folder + "J_eff_"       + to_string (l) + tag + ".txt";

      ofstream pops_outputFile (pops_file);
      ofstream Jeff_outputFile (Jeff_file);

	    for (long p = 0; p < ncells; p++)
	    {
	    	for (int i = 0; i < nlev[l]; i++)
	    	{
	    		pops_outputFile << population[p][l](i) << "\t";
	    	}

  	    for (int k = 0; k < nrad[l]; k++)
  	    {
  	    	Jeff_outputFile << J_eff[p][l][k] << "\t";
  	    }

	    	pops_outputFile << endl;
  	    Jeff_outputFile << endl;
	    }

	    pops_outputFile.close ();
	    Jeff_outputFile.close ();

      cout << "Written files:" << endl;
      cout << pops_file        << endl;
      cout << Jeff_file        << endl;
	  }
	}


	return (0);

}


int LEVELS ::
    update_using_LTE (const LINEDATA& linedata, const SPECIES& species,
		                  const TEMPERATURE& temperature, const long p, const int l)
{

 	// Set population total

 	population_tot[p][l] = species.density[p] * species.abundance[p][linedata.num[l]];

  if (population_tot[p][l] == 0.0)
  {
    cout << p << " " << l << endl;
  }

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

      //if (isnan(population[p][l](i)))
      //{
      //  cout << population_tot[p][l] << " " << partition_function << endl;
      //}
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


  // Avoid too small (or negative) populations

  for (int i = 0; i < nlev[l]; i++)
  {
    if (population[p][l](i)*population_tot[p][l] < 1.0E-20)
    {
      population[p][l](i) = 0.0;
    }
  }


	return (0);

}




int LEVELS ::
    check_for_convergence (const long p, const int l)
{

  // Start by assuming that the populations are converged

  not_converged[l] = false;


	// Check whether they are indeed converged

	VectorXd dpop = population[p][l] - population_prev1[p][l];
	VectorXd spop = population[p][l] + population_prev1[p][l];

  double min_pop = 1.0E-10 * population_tot[p][l];


  for (int i = 0; i < nlev[l]; i++)
  {
    if (population[p][l](i) > min_pop)
    {
      double relative_change = 2.0 * fabs(dpop(i) / spop(i));
			//cout << relative_change << endl;

      if (relative_change > POP_PREC)
      {
        not_converged[l] = true;

        fraction_not_converged[l] += 1.0/(ncells*nlev[l]);
      }
    }
  }


	return (0);
}




///  get_emissivity_and_opacity
///    @param[in] linedata: data structure containing the line data
///    @param[in] levels: data structure containing the level populations
/////////////////////////////////////////////////////////////////////////

int LEVELS ::
    calc_line_emissivity_and_opacity (const LINEDATA& linedata, LINES& lines,
		                                  const long p, const int l) const
{

	// For all radiative transitions

  for (int k = 0; k < linedata.nrad[l]; k++)
	{
	  const int i = linedata.irad[l][k];
	  const int j = linedata.jrad[l][k];

    const long index = lines.index(p,l,k);

    const double hv_4pi = HH * linedata.frequency[l][k] / (4.0*PI);

	  lines.emissivity[index] = hv_4pi * linedata.A[l](i,j) * population[p][l](i);

	     lines.opacity[index] = hv_4pi * (  population[p][l](j) * linedata.B[l](j,i)
                 		                    - population[p][l](i) * linedata.B[l](i,j) );

     //if (isnan(lines.emissivity[index]))
     //{
    //   cout << p << " " << l << " " << i << " " << population[p][l](i) << endl;
     //}
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
    calc_J_eff (FREQUENCIES& frequencies, const TEMPERATURE& temperature,
				        RADIATION& radiation, const long p, const int l)
{

	for (int k = 0; k < nrad[l]; k++)
	{
    const Long1 freq_nrs = frequencies.nr_line[p][l][k];

#   if (GRID_SIMD)
		  const long    f_line = freq_nrs[NR_LINE_CENTER] / n_simd_lanes;
		  const long lane_line = freq_nrs[NR_LINE_CENTER] % n_simd_lanes;
      const double freq_line = frequencies.all[p][f_line].getlane(lane_line);
#   else
      const double freq_line = frequencies.all[p][freq_nrs[NR_LINE_CENTER]];
#   endif

    J_eff[p][l][k] = 0.0;

    for (long z = 0; z < N_QUADRATURE_POINTS; z++)
    {

#     if (GRID_SIMD)
		    const long    f = freq_nrs[z] / n_simd_lanes;
		    const long lane = freq_nrs[z] % n_simd_lanes;
		  	const double JJ = radiation.J[radiation.index(p,f)].getlane(lane);
#     else
		  	const double JJ = radiation.J[radiation.index(p,freq_nrs[z])];
#     endif

      J_eff[p][l][k] += H_weights[z] / profile_width (temperature.gas[p], freq_line) * JJ;
    }
  }


  return (0);

 }
