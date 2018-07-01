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

#include "levels.hpp"
#include "linedata.hpp"


///  acceleration_Ng: perform a Ng accelerated iteration for level populations
///  - All variable names are based on lecture notes by C.P. Dullemond
//////////////////////////////////////////////////////////////////////////////

int LEVELS :: update_using_Ng_acceleration ()
{

  for (int l = 0; l < nlspec; l++)
	{

    vector<VectorXd> Q1 (ncells, VectorXd (nlev[l]));
    vector<VectorXd> Q2 (ncells, VectorXd (nlev[l]));
    vector<VectorXd> Q3 (ncells, VectorXd (nlev[l]));

    vector<VectorXd> Wt (ncells, VectorXd (nlev[l]));


#   pragma omp parallel          \
    shared (l, Q1, Q2, Q3, Wt)   \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*ncells)/num_threads;
    long stop  = ((thread_num+1)*ncells)/num_threads;  // Note brackets


    for (long p = start; p < stop; p++)
    {
      Q1[p] = population[p][l] - 2.0*population_prev1[p][l] + population_prev2[p][l];
      Q2[p] = population[p][l] -     population_prev1[p][l] - population_prev2[p][l] + population_prev3[p][l];
      Q3[p] = population[p][l] -     population_prev1[p][l];

      for (int i = 0; i < nlev[l]; i++)
      {
        if (population[p][l](i) > 0.0)
        {
          Wt[p](i) = 1.0 / population[p][l](i);
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


#   pragma omp parallel for reduction( + : A1, A2, B1, B2, C1, C2)

    for (long p = 0; p < ncells; p++)
    {
      A1      = A1 + Q1[p].dot((Wt[p].asDiagonal())*Q1[p]);
      A2 = B1 = A2 + Q1[p].dot((Wt[p].asDiagonal())*Q2[p]);
      B2      = B2 + Q2[p].dot((Wt[p].asDiagonal())*Q2[p]);
      C1      = C1 + Q1[p].dot((Wt[p].asDiagonal())*Q3[p]);
      C2      = C2 + Q2[p].dot((Wt[p].asDiagonal())*Q3[p]);
    }

    const double denominator = A1*B2 - A2*B1;

    if (denominator == 0.0)
    {
      return (0);
    }

    else
    {

      const double a = (C1*B2 - C2*B1) / denominator;
      const double b = (C2*A1 - C1*A2) / denominator;


#     pragma omp parallel   \
      shared (l)            \
      default (none)
      {

      int num_threads = omp_get_num_threads();
      int thread_num  = omp_get_thread_num();

      long start = (thread_num*ncells)/num_threads;
      long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


      for (long p = start; p < stop; p++)
      {

        const VectorXd pop_tmp = population[p][l];

        population[p][l] = (1.0 - a - b)*population[p][l]
                                     + a*population_prev1[p][l]
	  	   											 			 + b*population_prev2[p][l];

        population_prev3[p][l] = population_prev2[p][l];
        population_prev2[p][l] = population_prev1[p][l];
        population_prev1[p][l] = pop_tmp;

      } // end of o loop over grid points
      } // end of OpenMP parallel region

    }

  } // end of l loop


  return (0);

}
