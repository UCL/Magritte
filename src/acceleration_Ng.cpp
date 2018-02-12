// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "acceleration_Ng.hpp"



// acceleration_Ng: perform a Ng accelerated iteration for level populations
// -------------------------------------------------------------------------

int acceleration_Ng (long ncells, int lspec,
                     double *prev3_pop, double *prev2_pop, double *prev1_pop, double *pop)
{

  // All variable names are based on lecture notes by C.P. Dullemond


# if (FIXED_NCELLS)

    double Q1[NCELLS*nlev[lspec]];
    double Q2[NCELLS*nlev[lspec]];
    double Q3[NCELLS*nlev[lspec]];

    double Wt[NCELLS*nlev[lspec]];   // weights of inner product

# else

    double *Q1 = new double[ncells*nlev[lspec]];
    double *Q2 = new double[ncells*nlev[lspec]];
    double *Q3 = new double[ncells*nlev[lspec]];

    double *Wt = new double[ncells*nlev[lspec]];   // weights of inner product

# endif


# pragma omp parallel                                                                            \
  shared (ncells, Q1, Q2, Q3, Wt, nlev, cum_nlev, lspec, pop, prev1_pop, prev2_pop, prev3_pop)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;  // Note brackets


  for (long gridp = start; gridp < stop; gridp++)
  {
    for (int i = 0; i < nlev[lspec]; i++)
    {
      long p_i = LSPECGRIDLEV(lspec,gridp,i);
      long w_i = LINDEX(lspec,gridp,i);

      Q1[w_i] = pop[p_i] - 2.0*prev1_pop[p_i] + prev2_pop[p_i];
      Q2[w_i] = pop[p_i] - prev1_pop[p_i] - prev2_pop[p_i] + prev3_pop[p_i];
      Q3[w_i] = pop[p_i] - prev1_pop[p_i];

      if (pop[p_i] > 0.0)
      {
        Wt[w_i] = 1.0 / fabs(pop[p_i]);
      }

      else
      {
        Wt[w_i] = 1.0;
      }

    } // end of i loop over levels

  } // end of gridp loop over grid points
  } // end of OpenMP parallel region


  double A1 = 0.0;
  double A2 = 0.0;

  double B1 = 0.0;
  double B2 = 0.0;

  double C1 = 0.0;
  double C2 = 0.0;


# pragma omp parallel for reduction( + : A1, A2, B1, B2, C1, C2)

  for (long gi = 0; gi < NCELLS*nlev[lspec]; gi++)
  {
    A1      = A1 + Wt[gi]*Q1[gi]*Q1[gi];
    A2 = B1 = A2 + Wt[gi]*Q1[gi]*Q2[gi];
    B2      = B2 + Wt[gi]*Q2[gi]*Q2[gi];
    C1      = C1 + Wt[gi]*Q1[gi]*Q3[gi];
    C2      = C2 + Wt[gi]*Q2[gi]*Q3[gi];
  }


# if (!FIXED_NCELLS)

    delete [] Q1;
    delete [] Q2;
    delete [] Q3;

    delete [] Wt;

# endif


  double denominator = A1*B2 - A2*B1;

  if (denominator == 0.0)
  {
    return (0);
  }

  else
  {

    double a = (C1*B2 - C2*B1) / denominator;
    double b = (C2*A1 - C1*A2) / denominator;


#   pragma omp parallel                                                                  \
    shared (ncells, a, b, nlev, cum_nlev, lspec, pop, prev1_pop, prev2_pop, prev3_pop)   \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


    for (long gridp = start; gridp < stop; gridp++)
    {
      for (int i = 0; i < nlev[lspec]; i++)
      {
        long p_i = LSPECGRIDLEV(lspec,gridp,i);
        long w_i = LINDEX(lspec,gridp,i);

        double pop_tmp = pop[p_i];

        pop[p_i] = (1.0 - a - b)*pop[p_i] + a*prev1_pop[p_i] + b*prev2_pop[p_i];

        prev3_pop[p_i] = prev2_pop[p_i];
        prev2_pop[p_i] = prev1_pop[p_i];
        prev1_pop[p_i] = pop_tmp;

      } // end of i loop over levels

    } // end of gridp loop over grid points
    } // end of OpenMP parallel region

  }


  return (0);

}




// store_populations: update previous populations
// ----------------------------------------------

int store_populations (long ncells, int lspec,
                       double *prev3_pop, double *prev2_pop, double *prev1_pop, double *pop)
{

# pragma omp parallel                                                            \
  shared (ncells, lspec, nlev, cum_nlev, prev3_pop, prev2_pop, prev1_pop, pop)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long gridp = start; gridp < stop; gridp++)
  {
    for (int i = 0; i < nlev[lspec]; i++)
    {
      long p_i = LSPECGRIDLEV(lspec,gridp,i);

      prev3_pop[p_i] = prev2_pop[p_i];
      prev2_pop[p_i] = prev1_pop[p_i];
      prev1_pop[p_i] = pop[p_i];

    }

  } // end of gridp loop over grid points
  } // end of OpenMP parallel region


  return (0);

}
