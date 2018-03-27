// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "declarations.hpp"
#include "level_populations.hpp"
#include "acceleration_Ng.hpp"
#include "lines.hpp"
#include "calc_C_coeff.hpp"
#include "initializers.hpp"
#include "sobolev.hpp"
#include "sobolev.hpp"
#include "radiative_transfer.hpp"
#include "level_population_solver.hpp"


// level_populations: iteratively calculates the level populations
// ---------------------------------------------------------------

int level_populations (long ncells, CELLS *cells, RAYS rays,
                       SPECIES species, LINES lines)
{


# if (FIXED_NCELLS)

    double prev1_pop[NCELLS*TOT_NLEV];   // level population n_i 1 iteration ago
    double prev2_pop[NCELLS*TOT_NLEV];   // level population n_i 2 iterations ago
    double prev3_pop[NCELLS*TOT_NLEV];   // level population n_i 3 iterations ago

    double mean_intensity_eff[NCELLS*TOT_NRAD];   // mean intensity for a ray
    double Lambda_diagonal[NCELLS*TOT_NRAD];      // mean intensity for a ray


# else

    double *prev1_pop = new double[ncells*TOT_NLEV];   // level population n_i 1 iteration ago
    double *prev2_pop = new double[ncells*TOT_NLEV];   // level population n_i 2 iterations ago
    double *prev3_pop = new double[ncells*TOT_NLEV];   // level population n_i 3 iterations ago

    double *mean_intensity_eff = new double[ncells*TOT_NRAD];   // mean intensity for a ray
    double *Lambda_diagonal    = new double[ncells*TOT_NRAD];   // mean intensity for a ray

# endif


  initialize_double_array (NCELLS*TOT_NRAD, mean_intensity_eff);
  initialize_double_array (NCELLS*TOT_NRAD, Lambda_diagonal);


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

    for (int ls = 0; ls < NLSPEC; ls++)
    {
      if (prev_not_converged[ls])
      {
        prev_not_converged[ls] = not_converged[ls];
        not_converged[ls]      = false;
      }
    }


    double source[NCELLS*TOT_NRAD];    // source function

    double opacity[NCELLS*TOT_NRAD];   // opacity


    // For each line producing species

    for (int ls = 0; ls < NLSPEC; ls++)
    {
      if  (prev_not_converged[ls])
      {
        niterations[ls]++;

        printf( "(level_populations): Iteration %d for %s\n",
                niterations[ls], lines.sym[ls].c_str() );


        n_not_converged[ls] = 0;   // number of grid points that are not yet converged


        // Perform an Ng acceleration step every 4th iteration

#       if (ACCELERATION_POP_NG)

          if (niterations[ls]%4 == 0)
          {
            acceleration_Ng (NCELLS, cells, ls, prev3_pop, prev2_pop, prev1_pop);
          }

#       endif


        // Store populations of previous 3 iterations

        store_populations (NCELLS, cells, ls, prev3_pop, prev2_pop, prev1_pop);


        // Calculate source and opacity for all transitions over whole grid

        lines.source (NCELLS, cells, ls, source);

        lines.opacity (NCELLS, cells, ls, opacity);
      }
    } // end of lspec loop over line producing species


    // For every grid point


#   pragma omp parallel                                                             \
    shared (lines, ncells, cells, rays, species,                          \
            opacity, source, Lambda_diagonal, mean_intensity_eff,                   \
            prev1_pop, not_converged, n_not_converged, nlev, cum_nlev, cum_nlev2,   \
            nrad, cum_nrad, prev_not_converged, some_not_converged)                 \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets



    for (long p = start; p < stop; p++)
    {
      if (!cells->boundary[p])
      {
        double R[TOT_NLEV2];         // Transition matrix R_ij
        double C_coeff[TOT_NLEV2];   // Einstein C_ij coefficient

        if (thread_num == 0)
        {
          // printf("thread 0 on point %ld of %ld\n", n , stop);
        }


        // For each line producing species

        for (int ls = 0; ls < NLSPEC; ls++)
        {
          if (prev_not_converged[ls])
          {

            // Calculate collisional terms and fill first part of transition matrix
            //  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


            calc_C_coeff (NCELLS, cells, species, lines, C_coeff, p, ls);


            // Fill first part of transition matrix R

            for (int i = 0; i < nlev[ls]; i++)
            {
              for (int j = 0; j < nlev[ls]; j++)
              {
                long b_ij = LSPECLEVLEV(ls,i,j);   // R, A_coeff and C_coeff index

                R[b_ij] = lines.A_coeff[b_ij] + C_coeff[b_ij];
              }
            }




            // Calculate and add  B_ij<J_ij> term
            // _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


            // For all transitions

            for (int kr = 0; kr < nrad[ls]; kr++)
            {
              long mm_ij = LSPECRAD(ls,kr);

              int i = lines.irad[mm_ij];   // i index corresponding to transition kr
              int j = lines.jrad[mm_ij];   // j index corresponding to transition kr

              long b_ij = LSPECLEVLEV(ls,i,j);   // A_coeff, B_coeff and frequency index
              long b_ji = LSPECLEVLEV(ls,j,i);   // A_coeff, B_coeff and frequency index

              double A_ij = lines.A_coeff[b_ij];
              double B_ij = lines.B_coeff[b_ij];
              double B_ji = lines.B_coeff[b_ji];

              long m_ij = LSPECGRIDRAD(ls,p,kr);

              cells->mean_intensity[KINDEX(p,mm_ij)] = 0.0;


              // Calculate mean intensity

#             if (SOBOLEV)

            if (thread_num == 0)
            {
              // printf("YESfdg");
            }

                sobolev (NCELLS, cells, rays, lines, Lambda_diagonal, mean_intensity_eff,
                         source, opacity, p, ls, kr);


            if (thread_num == 0)
            {
              // printf("NOsgsf");
            }


#             else

                radiative_transfer (NCELLS, cells, rays, lines, Lambda_diagonal,
                                    mean_intensity_eff, source, opacity, p, ls, kr);

#             endif


              // Fill i > j part

              R[b_ij] = R[b_ij] - A_ij*Lambda_diagonal[m_ij] + NRAYS*B_ij*mean_intensity_eff[m_ij];


              // Add j > i part

              R[b_ji] = R[b_ji] + NRAYS*B_ji*mean_intensity_eff[m_ij];

            } // end of kr loop over transitions




            // Solve equilibrium equation at each point
            // _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

            if (thread_num == 0)
            {
              // printf("YES");
            }

            // Solve radiative balance equation for level populations

            level_population_solver (NCELLS, cells, lines, p, ls, R);


            if (thread_num == 0)
            {
              // printf("NO");
            }

            // Check for convergence

            for (int i = 0; i < nlev[ls]; i++)
            {
              long p_i  = LSPECGRIDLEV(ls,p,i);   // dpop index
              long pp_i = LSPECLEV(ls,i);         // pop index

              double dpop = cells->pop[LINDEX(p,pp_i)] - prev1_pop[p_i];
              double spop = cells->pop[LINDEX(p,pp_i)] + prev1_pop[p_i];

              double min_pop = 1.0E-10 * cells->abundance[SINDEX(p,lines.nr[ls])];


              if ( (cells->pop[LINDEX(p,pp_i)] > min_pop) && (spop != 0.0) )
              {
                double dpop_rel = 2.0 * fabs(dpop) / spop;


                // If population of any level is not converged

                if (dpop_rel > POP_PREC)
                {
                  not_converged[ls]  = true;
                  some_not_converged = true;

                  n_not_converged[ls]++;
                }
              }

            } // end of i loop over levels


          }
        } // end of lspec loop over line producing species

      } // end if not boundary point

    } // end of n loop over cells
    } // end of OpenMP parallel region



    // Limit the number of iterations

    for (int ls = 0; ls < NLSPEC; ls++)
    {
      if (prev_not_converged[ls])
      {
        if (    (niterations[ls] > MAX_NITERATIONS)
             || (n_not_converged[ls] < 0.02*NCELLS*nlev[ls]) )
        {
          not_converged[ls]  = false;
          some_not_converged = false;
        }

        printf ("(level_populations): Not yet converged for %ld of %d (NCELLS = %ld)\n",
                n_not_converged[ls], NCELLS*nlev[ls], NCELLS);

      }
    }


  } // end of while loop of iterations



  // Print stats for calculations on lspec

  for (int ls = 0; ls < NLSPEC; ls++)
  {
    printf ("(level_populations): population levels for %s converged after %d iterations\n"
            "                     with precision %.1lE\n",
            lines.sym[ls].c_str(), niterations[ls], POP_PREC);
  }


# if (!FIXED_NCELLS)

    delete [] prev1_pop;
    delete [] prev2_pop;
    delete [] prev3_pop;

    delete [] mean_intensity_eff;
    delete [] Lambda_diagonal;

# endif


  return(0);
}
