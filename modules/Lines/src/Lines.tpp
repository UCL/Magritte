// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <math.h>
#include <omp.h>
#include <vector>
using namespace std;
#include <Eigen/Core>
#include <Eigen/QR>
using namespace Eigen;

#include "Lines.hpp"
#include "levels.hpp"
#include "linedata.hpp"
#include "acceleration_Ng.hpp"
#include "RadiativeTransfer/src/species.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/RadiativeTransfer.hpp"
#include "RadiativeTransfer/src/cells.hpp"
#include "RadiativeTransfer/src/lines.hpp"
#include "RadiativeTransfer/src/radiation.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"


#define POP_PREC        1.0E-4
#define MAX_NITERATIONS 100


///  Lines: iteratively calculates level populations
////////////////////////////////////////////////////

template <int Dimension, long Nrays>
int Lines (CELLS<Dimension, Nrays>& cells, LINEDATA& linedata, SPECIES& species,
		       TEMPERATURE& temperature, FREQUENCIES& frequencies, LEVELS& levels,
					 RADIATION& radiation)
{


	long nfreq_scat = 1;

	LINES lines (cells.ncells, linedata);
  cout << "lines constructed!" << endl;

	SCATTERING scattering (Nrays, nfreq_scat);
  cout << "scattering constructed!" << endl;


  bool some_not_converged = true;                   // true when there are unconverged species

  vector<int> niterations (linedata.nlspec);        // number of iterations

	vector<bool> not_converged (linedata.nlspec);     // true when species is not converged

  vector<long> n_not_converged (linedata.nlspec);   // number of unconverged cells


  cout << "Just before while" << endl;

  // Iterate until level populations converge

  while (some_not_converged)
  {

    // New iteration, assume populations are converged until proven differently...

    for (int l = 0; l < linedata.nlspec; l++)
    {
        not_converged[l] = false;
      n_not_converged[l] = 0;
    }


    // For each line producing species

    for (int l = 0; l < linedata.nlspec; l++)
    {
      niterations[l]++;

			
      // Perform an Ng acceleration step every 4th iteration

      if (niterations[l]%4 == 0)
      {
        acceleration_Ng (linedata, l, levels);
      }


      // Store populations of previous 3 iterations

      store_populations (levels, l);
    }

		cout << "Here we are...." << endl;


    // Calculate source and opacity for all transitions over whole grid

	  lines.get_emissivity_and_opacity (linedata, levels);

		vector<vector<double>> J (levels.ncells, vector<double> (frequencies.nfreq));

		long rays[Nrays];

		for (long r = 0; r < Nrays; r++)
		{
			rays[r] = r;
		}

		cout << "Still fine..." << endl;
    RadiativeTransfer<Dimension, Nrays> (cells, temperature, frequencies, Nrays, rays, lines, scattering, radiation, J);
    cout << "Little harder..." << endl;		
    calc_level_populations (linedata, lines, levels, species, frequencies, temperature,
				                    J, not_converged, n_not_converged, some_not_converged, Nrays);

		cout << "No way I can do this..." << endl;

    // Limit the number of iterations

    for (int l = 0; l < linedata.nlspec; l++)
    {
      if (    (niterations[l] > MAX_NITERATIONS)
           || (n_not_converged[l] < 0.01*levels.ncells*linedata.nlev[l]) )
      {
        not_converged[l] = false;
      }

			cout << "(Lines): Not yet converged for " << endl;
			cout << "          " << n_not_converged[l] << " of " << levels.ncells * linedata.nlev[l] << endl;
    }


    // If some are not converged

    some_not_converged = false;

    for (int l = 0; l < linedata.nlspec; l++)
    {
      if (not_converged[l])
      {
        some_not_converged = true;
      }
    }


  } // end of while loop of iterations



  // Print stats

  for (int l = 0; l < linedata.nlspec; l++)
  {
    cout << "(Lines): populations for " << linedata.sym[l] << " :"           << endl;
		cout << "         converged after " << niterations[l]  << " iterations." << endl;
  }


  return (0);

}







int calc_level_populations (LINEDATA& linedata, LINES& lines, LEVELS& levels, SPECIES& species,
		                        FREQUENCIES& frequencies, TEMPERATURE& temperature, vector<vector<double>>& J,
														vector<bool> not_converged, vector<long> n_not_converged,
														bool some_not_converged, long Nrays)
{

  cout << "We Got here!" << endl;


# pragma omp parallel                                        \
  shared (linedata, lines, levels, species, frequencies, temperature, J, not_converged,   \
			    n_not_converged, some_not_converged, Nrays)               \
  default (none)
  {

  const int num_threads = omp_get_num_threads();
  const int thread_num  = omp_get_thread_num();

  const long start = (thread_num*levels.ncells)/num_threads;
  const long stop  = ((thread_num+1)*levels.ncells)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {

    // For each line producing species

    for (int l = 0; l < linedata.nlspec; l++)
    {

			// Setup transition matrix R
			// -------------------------


      MatrixXd R (linedata.nlev[l],linedata.nlev[l]);   // Transition matrix R_ij
      MatrixXd C (linedata.nlev[l],linedata.nlev[l]);   // Einstein C_ij coefficient


      // Calculate collissional Einstein coefficients

      linedata.calc_Einstein_C (species, temperature.gas[p], p, l, C);


      // Add Einstein A and C to transition matrix

      R = linedata.A[l] + C;


      // Add  B_ij<J_ij> term

      for (int k = 0; k < linedata.nrad[l]; k++)
      {
        const int i = linedata.irad[l][k];   // i index corresponding to transition k
        const int j = linedata.jrad[l][k];   // j index corresponding to transition k

        const double J_eff = lines.J_eff (frequencies, temperature, J, p, l, k);

        R(i,j) += Nrays * linedata.B[l](i,j) * J_eff; // - linedata.A[l](i,j)*Lambda(); 
        R(j,i) += Nrays * linedata.B[l](j,i) * J_eff;
      }




      // Solve statistical equilibrium equation
      // -------------------------------------- 


			MatrixXd M = R.transpose();
      VectorXd y (linedata.nlev[l]);   

			for (int i = 0; i < linedata.nlev[l]; i++)
			{
				double out = 0.0;

  			for (int j = 0; j < linedata.nlev[l]; j++)
				{
          out += R(i,j);  
				}

				M(i,i) = -out;
			}

			for (int j = 0; j < linedata.nlev[l]; j++)
			{
				y(j) = 0.0;

				M(linedata.nlev[l]-1,j) = 1.0;
			}

			y(linedata.nlev[l]-1) = species.density[p] * species.abundance[p][linedata.num[l]];


      // Solve statistical equilibrium equation for level populations
		
      levels.population[p][l] = M.householderQr().solve(y);


      // Check for convergence

			VectorXd dpop = levels.population[p][l] - levels.population_prev1[p][l];
			VectorXd spop = levels.population[p][l] + levels.population_prev1[p][l];

      double min_pop = 1.0E-10 * species.abundance[p][linedata.num[l]];

      for (int i = 0; i < linedata.nlev[l]; i++)
      {
        if ( (levels.population[p][l](i) > min_pop) && (spop(i) != 0.0) )
        {
          double relative_change = 2.0 * fabs(dpop(i) / spop(i));


          // If population of any level is not converged

          if (relative_change > POP_PREC)
          {
              not_converged[l] = true;
            n_not_converged[l]++;
          }
        }

      } // end of i loop over levels


    } // end of lspec loop over line producing species

  } // end of n loop over cells
  } // end of OpenMP parallel region


  return (0);

}
