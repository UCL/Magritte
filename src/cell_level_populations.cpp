// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#if (CELL_BASED)

#include "cell_level_populations.hpp"
#include "acceleration_Ng.hpp"
#include "lines.hpp"
#include "calc_C_coeff.hpp"
#include "initializers.hpp"
#include "sobolev.hpp"
#include "cell_sobolev.hpp"
#include "cell_radiative_transfer.hpp"
#include "level_population_solver_otf.hpp"
#include "ray_tracing.hpp"
#include "write_output.hpp"


// level_populations: iteratively calculates the level populations
// ---------------------------------------------------------------

int cell_level_populations (long ncells, CELL *cell, int *irad, int*jrad, double *frequency,
                            double *A_coeff, double *B_coeff, double *pop,
                            double *C_data, double *coltemp, int *icol, int *jcol,
                            double *weight, double *energy, double *mean_intensity,
                            double *Lambda_diagonal, double *mean_intensity_eff )
{


  double prev1_pop[NCELLS*TOT_NLEV];   // level population n_i 1 iteration ago
  double prev2_pop[NCELLS*TOT_NLEV];   // level population n_i 2 iterations ago
  double prev3_pop[NCELLS*TOT_NLEV];   // level population n_i 3 iterations ago


  bool some_not_converged = true;    // true when not all species are converged

  bool not_converged[NLSPEC];        // true when not converged

  initialize_bool (NLSPEC, true, not_converged);

  bool prev_not_converged[NLSPEC];   // true when not converged last iteration

  initialize_bool (NLSPEC, true, prev_not_converged);

  int niterations[NLSPEC];           // number of iterations

  initialize_int_array (NLSPEC, niterations);

  int n_not_converged[NLSPEC];       // number of not converged cells

  initialize_int_array (NLSPEC, n_not_converged);


  // Iterate until level populations converge

  while (some_not_converged)
  {

    // New iteration, assume populations are converged until proven differently...

    some_not_converged = false;

    for (int lspec = 0; lspec < NLSPEC; lspec++)
    {
      if (prev_not_converged[lspec])
      {
        prev_not_converged[lspec] = not_converged[lspec];
        not_converged[lspec]      = false;
      }
    }


    double source[NCELLS*TOT_NRAD];    // source function

    double opacity[NCELLS*TOT_NRAD];   // opacity


    // For each line producing species

    for (int lspec = 0; lspec < NLSPEC; lspec++)
    {
      if  (prev_not_converged[lspec])
      {
        niterations[lspec]++;

        printf( "(level_populations): Iteration %d for %s\n",
                niterations[lspec], species[lspec_nr[lspec]].sym.c_str() );


        n_not_converged[lspec] = 0;   // number of grid points that are not yet converged


        // Perform an Ng acceleration step every 4th iteration

#       if (ACCELERATION_POP_NG)

          if (niterations[lspec]%4 == 0)
          {
            acceleration_Ng (NCELLS, lspec, prev3_pop, prev2_pop, prev1_pop, pop);
          }

#       endif


        // Store populations of previous 3 iterations

        store_populations (NCELLS, lspec, prev3_pop, prev2_pop, prev1_pop, pop);


        // Calculate source and opacity for all transitions over whole grid

        line_source (NCELLS, irad, jrad, A_coeff, B_coeff, pop, lspec, source);

        line_opacity (NCELLS, irad, jrad, frequency, B_coeff, pop, lspec, opacity);
      }
    } // end of lspec loop over line producing species


    // For every grid point

#   pragma omp parallel                                                                          \
    shared (energy, weight, icol, jcol, coltemp, C_data, pop, ncells, cell, lspec_nr, frequency, \
            opacity, source, mean_intensity, Lambda_diagonal, mean_intensity_eff, species,       \
            prev1_pop, not_converged, n_not_converged, nlev, cum_nlev, cum_nlev2, irad, jrad,    \
            nrad, cum_nrad, A_coeff, B_coeff, prev_not_converged, some_not_converged)            \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


    for (long n = start; n < stop; n++)
    {
      double R[TOT_NLEV2];         // Transition matrix R_ij

      double C_coeff[TOT_NLEV2];   // Einstein C_ij coefficient


      // For each line producing species

      for (int lspec = 0; lspec < NLSPEC; lspec++)
      {
        if (prev_not_converged[lspec])
        {

          // Calculate collisional terms and fill first part of transition matrix
          //  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


          calc_C_coeff (NCELLS, cell, C_data, coltemp, icol, jcol,
                        weight, energy, C_coeff, n, lspec);


          // Fill first part of transition matrix R

          for (int i = 0; i < nlev[lspec]; i++)
          {
            for (int j = 0; j < nlev[lspec]; j++)
            {
              long b_ij = LSPECLEVLEV(lspec,i,j);   // R, A_coeff and C_coeff index

              R[b_ij] = A_coeff[b_ij] + C_coeff[b_ij];
            }
          }


          // Calculate and add  B_ij<J_ij> term
          // _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


          // For all transitions

          for (int kr = 0; kr < nrad[lspec]; kr++)
          {
            int i     = irad[LSPECRAD(lspec,kr)];   // i index corresponding to transition kr
            int j     = jrad[LSPECRAD(lspec,kr)];   // j index corresponding to transition kr

            long b_ij = LSPECLEVLEV(lspec,i,j);     // A_coeff, B_coeff and frequency index
            long b_ji = LSPECLEVLEV(lspec,j,i);     // A_coeff, B_coeff and frequency index

            double A_ij = A_coeff[b_ij];
            double B_ij = B_coeff[b_ij];
            double B_ji = B_coeff[b_ji];

            long m_ij = LSPECGRIDRAD(lspec,n,kr);

            mean_intensity[m_ij] = 0.0;


            // Calculate mean intensity

#           if (SOBOLEV)

              cell_sobolev (NCELLS, cell, mean_intensity, Lambda_diagonal, mean_intensity_eff, source,
                            opacity, frequency, irad, jrad, n, lspec, kr);

#           else

              cell_radiative_transfer (NCELLS, cell, mean_intensity, Lambda_diagonal, mean_intensity_eff,
                                       source, opacity, frequency, irad, jrad, n, lspec, kr);

#           endif


            // Fill i > j part

            R[b_ij] = R[b_ij] - A_ij*Lambda_diagonal[m_ij] + B_ij*mean_intensity_eff[m_ij];


            // Add j > i part

            R[b_ji] = R[b_ji] + B_ji*mean_intensity_eff[m_ij];

          } // end of kr loop over transitions


          // Solve equilibrium equation at each point
          // _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


          // Solve radiative balance equation for level populations

          level_population_solver_otf (NCELLS, cell, n, lspec, R, pop);


          // Check for convergence

          for (int i = 0; i < nlev[lspec]; i++)
          {
            long p_i = LSPECGRIDLEV(lspec,n,i);   // pop and dpop index

            double dpop = pop[p_i] - prev1_pop[p_i];
            double spop = pop[p_i] + prev1_pop[p_i];

            double min_pop = 1.0E-10 * cell[n].abundance[lspec_nr[lspec]];


            if ( (pop[p_i] > min_pop) && (spop != 0.0) )
            {
              double dpop_rel = 2.0 * fabs(dpop) / spop;


              // If population of any level is not converged

              if (dpop_rel > POP_PREC)
              {
                not_converged[lspec] = true;
                some_not_converged   = true;

                n_not_converged[lspec]++;
              }
            }

          } // end of i loop over levels


        }
      } // end of lspec loop over line producing species


    } // end of n loop over cells
    } // end of OpenMP parallel region



    // Limit the number of iterations

    for (int lspec = 0; lspec < NLSPEC; lspec++)
    {
      if (prev_not_converged[lspec])
      {
        if ( (niterations[lspec] > MAX_NITERATIONS) || (n_not_converged[lspec] < NCELLS/10) )
        {
          not_converged[lspec] = false;
          some_not_converged   = false;
        }

        printf( "(level_populations): Not yet converged for %ld of %d\n",
                n_not_converged[lspec], NCELLS*nlev[lspec] );

      }
    }


  } // end of while loop of iterations



  // Print stats for calculations on lspec

  for (int lspec = 0; lspec < NLSPEC; lspec++)
  {
    printf ("(level_populations): population levels for %s converged after %d iterations\n"
            "                     with precision %.1lE\n",
            species[lspec_nr[lspec]].sym.c_str(), niterations[lspec], POP_PREC);
  }


  return(0);
}


#endif
