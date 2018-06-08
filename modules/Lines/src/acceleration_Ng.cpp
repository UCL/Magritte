// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <omp.h>
#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "acceleration_Ng.hpp"
#include "levels.hpp"
#include "linedata.hpp"


///  acceleration_Ng: perform a Ng accelerated iteration for level populations
///  - All variable names are based on lecture notes by C.P. Dullemond
//////////////////////////////////////////////////////////////////////////////

int acceleration_Ng (LINEDATA& linedata, int l, LEVELS& levels)
{


  vector<VectorXd> Q1 (levels.ncells, VectorXd (linedata.nlev[l]));
  vector<VectorXd> Q2 (levels.ncells, VectorXd (linedata.nlev[l]));
  vector<VectorXd> Q3 (levels.ncells, VectorXd (linedata.nlev[l]));

  vector<VectorXd> Wt (levels.ncells, VectorXd (linedata.nlev[l]));


# pragma omp parallel                            \
  shared (linedata, l, levels, Q1, Q2, Q3, Wt)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*levels.ncells)/num_threads;
  long stop  = ((thread_num+1)*levels.ncells)/num_threads;  // Note brackets


  for (long p = start; p < stop; p++)
  {
    Q1[p] = levels.population[p][l] - 2.0*levels.population_prev1[p][l] + levels.population_prev2[p][l];
    Q2[p] = levels.population[p][l] -     levels.population_prev1[p][l] - levels.population_prev2[p][l] + levels.population_prev3[p][l];
    Q3[p] = levels.population[p][l] -     levels.population_prev1[p][l];

    for (int i = 0; i < linedata.nlev[l]; i++)
    {
      if (levels.population[p][l](i) > 0.0)
      {
        Wt[p](i) = 1.0 / fabs(levels.population[p][l](i));
      }

      else
      {
        Wt[p](i) = 1.0;
      }
    }

  } // end of o loop over grid points
  } // end of OpenMP parallel region


  double A1 = 0.0;
  double A2 = 0.0;

  double B1 = 0.0;
  double B2 = 0.0;

  double C1 = 0.0;
  double C2 = 0.0;


# pragma omp parallel for reduction( + : A1, A2, B1, B2, C1, C2)

  for (long p = 0; p < levels.ncells; p++)
  {
    A1      = A1 + Q1[p].dot((Wt[p].asDiagonal())*Q1[p]);
    A2 = B1 = A2 + Q1[p].dot((Wt[p].asDiagonal())*Q2[p]);
    B2      = B2 + Q2[p].dot((Wt[p].asDiagonal())*Q2[p]);
    C1      = C1 + Q1[p].dot((Wt[p].asDiagonal())*Q3[p]);
    C2      = C2 + Q2[p].dot((Wt[p].asDiagonal())*Q3[p]);
  }

  double denominator = A1*B2 - A2*B1;

  if (denominator == 0.0)
  {
    return (0);
  }

  else
  {

    double a = (C1*B2 - C2*B1) / denominator;
    double b = (C2*A1 - C1*A2) / denominator;


#   pragma omp parallel     \
    shared (levels, l, a, b)   \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*levels.ncells)/num_threads;
    long stop  = ((thread_num+1)*levels.ncells)/num_threads;   // Note brackets


    for (long p = start; p < stop; p++)
    {

      VectorXd  pop_tmp = levels.population[p][l];

      levels.population[p][l] = (1.0 - a - b)*levels.population[p][l]
                                          + a*levels.population_prev1[p][l]
		    																	+ b*levels.population_prev2[p][l];

      levels.population_prev3[p][l] = levels.population_prev2[p][l];
      levels.population_prev2[p][l] = levels.population_prev1[p][l];
      levels.population_prev1[p][l] = pop_tmp;

    } // end of o loop over grid points
    } // end of OpenMP parallel region

  }


  return (0);

}




///  store_populations: update previous populations
///////////////////////////////////////////////////

int store_populations (LEVELS& levels, int l)
{

# pragma omp parallel   \
  shared (levels, l)    \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*levels.ncells)/num_threads;
  long stop  = ((thread_num+1)*levels.ncells)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {

    levels.population_prev3[p][l] = levels.population_prev2[p][l];
    levels.population_prev2[p][l] = levels.population_prev1[p][l];
    levels.population_prev1[p][l] = levels.population[p][l];

  } // end of o loop over grid points
  } // end of OpenMP parallel region


  return (0);

}
