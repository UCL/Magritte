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

#if (!CELL_BASED)

#include "level_populations_otf.hpp"
#include "acceleration_Ng.hpp"
#include "lines.hpp"
#include "calc_C_coeff.hpp"
#include "initializers.hpp"
#include "sobolev.hpp"
#include "radiative_transfer_otf.hpp"
#include "level_population_solver_otf.hpp"
#include "ray_tracing.hpp"
#include "write_output.hpp"


// level_populations: iteratively calculates level populations
// -----------------------------------------------------------

int level_populations_otf (long ncells, CELL *cell, int *irad, int*jrad, double *frequency,
                           double *A_coeff, double *B_coeff, double *pop,
                           double *C_data, double *coltemp, int *icol, int *jcol,
                           double *weight, double *energy, double *mean_intensity,
                           double *Lambda_diagonal, double *mean_intensity_eff)
{


  double prev1_pop[NCELLS*TOT_NLEV];   // level population n_i 1 iteration ago
  double prev2_pop[NCELLS*TOT_NLEV];   // level population n_i 2 iterations ago
  double prev3_pop[NCELLS*TOT_NLEV];   // level population n_i 3 iterations ago


  bool some_not_converged = true;            /*  true when some of the species are not converged */

  bool not_converged[NLSPEC];            /* true when popualations are not converged per species */

  initialize_bool(true, not_converged, NLSPEC);

  bool prev_not_converged[NLSPEC];   /* true when popualations were not converged last iteration */

  initialize_bool(true, prev_not_converged, NLSPEC);

  int niterations[NLSPEC];                                   /* number of iterations per species */

  initialize_int_array(niterations, NLSPEC);

  int n_not_converged[NLSPEC];                 /* number of not converged cells per species */

  initialize_int_array(n_not_converged, NLSPEC);



  /* Iterate until the level populations converge */

  while (some_not_converged)
  {

    /* New iteration, assume populations are converged until proven differently... */

    some_not_converged = false;

    for (int lspec=0; lspec<NLSPEC; lspec++)
    {
      if (prev_not_converged[lspec])
      {
        prev_not_converged[lspec] = not_converged[lspec];
        not_converged[lspec]      = false;
      }
    }


    double source[NCELLS*TOT_NRAD];                                            /* source function */

    double opacity[NCELLS*TOT_NRAD];                                                   /* opacity */


    /* For each line producing species */

    for (int lspec=0; lspec<NLSPEC; lspec++)
    {
      if  (prev_not_converged[lspec])
      {
        niterations[lspec]++;

        printf( "(level_populations): Iteration %d for %s\n",
                niterations[lspec], species[lspec_nr[lspec]].sym.c_str() );


        n_not_converged[lspec] = 0;          /* number of grid points that are not yet converged */


        /* Perform an Ng acceleration step every 4th iteration */

#       if (ACCELERATION_POP_NG)

          if (niterations[lspec]%4 == 0)
          {
            acceleration_Ng(NCELLS, lspec, prev3_pop, prev2_pop, prev1_pop, pop);
          }

#       endif


        /* Store the populations of the previous 3 iterations */

        store_populations(NCELLS, lspec, prev3_pop, prev2_pop, prev1_pop, pop);


        /* Calculate the source and opacity for all transitions over the whole grid */

        line_source(NCELLS, irad, jrad, A_coeff, B_coeff, pop, lspec, source );

        line_opacity(NCELLS, irad, jrad, frequency, B_coeff, pop, lspec, opacity );
      }
    } /* end of lspec loop over line producing species */


    /* For every grid point */

#   pragma omp parallel                                                                         \
    shared (energy, weight, icol, jcol, coltemp, C_data, pop, cell, lspec_nr, frequency,        \
            opacity, source, mean_intensity, Lambda_diagonal, mean_intensity_eff, species,      \
            prev1_pop, not_converged, n_not_converged, nlev, cum_nlev, cum_nlev2, irad, jrad,   \
            nrad, cum_nrad, A_coeff, B_coeff, prev_not_converged, some_not_converged)           \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;     /* Note the brackets are important here */


    for (long n=start; n<stop; n++)
    {

      long key[NCELLS];                /* stores the nrs. of the grid points on the rays in order */
      long raytot[NRAYS];                  /* cumulative nr. of evaluation points along each ray */
      long cum_raytot[NRAYS];              /* cumulative nr. of evaluation points along each ray */

      long first_velo[NRAYS/2];                    /* grid point with lowest velocity on the ray */

      EVALPOINT evalpoint[NCELLS];


      find_evalpoints(cell, evalpoint, key, raytot, cum_raytot, n);

      get_velocities(cell, evalpoint, key, raytot, cum_raytot, n, first_velo);


      double R[TOT_NLEV2];                                             /* Transition matrix R_ij */

      double C_coeff[TOT_NLEV2];                                    /* Einstein C_ij coefficient */


      /* For each line producing species */

      for (int lspec=0; lspec<NLSPEC; lspec++)
      {
        if  (prev_not_converged[lspec])
        {

          /* Calculate the collisional terms and fill the first part of the transition matrix      */
          /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


          calc_C_coeff (NCELLS, cell, C_data, coltemp, icol, jcol,
                        weight, energy, C_coeff, n, lspec );


          /* Fill the first part of the transition matrix R */

          for (int i=0; i<nlev[lspec]; i++)
          {
            for (int j=0; j<nlev[lspec]; j++)
            {
              long b_ij = LSPECLEVLEV(lspec,i,j);                  /* R, A_coeff and C_coeff index */

              R[b_ij] = A_coeff[b_ij] + C_coeff[b_ij];
            }
          }


          /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/




          /* Calculate and add the B_ij<J_ij> term */
          /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


          /* For all transitions */

          for (int kr=0; kr<nrad[lspec]; kr++)
          {
            int i     = irad[LSPECRAD(lspec,kr)];        /* i index corresponding to transition kr */
            int j     = jrad[LSPECRAD(lspec,kr)];        /* j index corresponding to transition kr */

            long b_ij = LSPECLEVLEV(lspec,i,j);            /* A_coeff, B_coeff and frequency index */
            long b_ji = LSPECLEVLEV(lspec,j,i);            /* A_coeff, B_coeff and frequency index */

            double A_ij = A_coeff[b_ij];
            double B_ij = B_coeff[b_ij];
            double B_ji = B_coeff[b_ji];

            long m_ij = LSPECGRIDRAD(lspec,n,kr);

            mean_intensity[m_ij] = 0.0;


            /* Calculate the mean intensity */

#           if (SOBOLEV)

            sobolev (NCELLS, cell, evalpoint, key, raytot, cum_raytot, mean_intensity, Lambda_diagonal,
                     mean_intensity_eff, source, opacity, frequency, irad, jrad, n, lspec, kr);

#           else

            radiative_transfer_otf (NCELLS, cell, evalpoint, key, raytot, cum_raytot, mean_intensity,
                                    Lambda_diagonal, mean_intensity_eff, source, opacity, frequency,
                                    irad, jrad, n, lspec, kr);

#           endif


            /* Fill the i>j part */

            R[b_ij] = R[b_ij] - A_ij*Lambda_diagonal[m_ij] + B_ij*mean_intensity_eff[m_ij];


            /* Add the j>i part */

            R[b_ji] = R[b_ji] + B_ji*mean_intensity_eff[m_ij];

          } /* end of kr loop over transitions */


          /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/




          /* Solve the equilibrium equation at each point */
          /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


          /* Solve the radiative balance equation for the level populations */

          level_population_solver_otf(NCELLS, cell, n, lspec, R, pop);


          /* Check for convergence */

          for (int i=0; i<nlev[lspec]; i++)
          {
            long p_i = LSPECGRIDLEV(lspec,n,i);                            /* pop and dpop index */

            double dpop = pop[p_i] - prev1_pop[p_i];
            double spop = pop[p_i] + prev1_pop[p_i];

            double min_pop = 1.0E-10 * cell[n].abundance[lspec_nr[lspec]];


            if ( (pop[p_i] > min_pop) && (spop != 0.0) )
            {
              double dpop_rel = 2.0 * fabs(dpop) / spop;


              /* If the population of any of the levels is not converged */

              if (dpop_rel > POP_PREC)
              {
                not_converged[lspec] = true;
                some_not_converged   = true;

                n_not_converged[lspec]++;
              }
            }

          } /* end of i loop over levels */


          /*_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/


        }
      } /* end of lspec loop over line producing species */


    } /* end of n loop over cells */
    } /* end of OpenMP parallel region */



    /* For each line producing species */

    for (int lspec=0; lspec<NLSPEC; lspec++)
    {
      if (prev_not_converged[lspec])
      {

        /* Limit the number of iterations */

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

  for (int lspec=0; lspec<NLSPEC; lspec++)
  {
    printf( "(level_populations): population levels for %s converged after %d iterations\n"
            "                     with precision %.1lE\n",
            species[lspec_nr[lspec]].sym.c_str(), niterations[lspec], POP_PREC );
  }


  return(0);
}


#endif // if not CELL_BASED
