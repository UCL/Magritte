// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <iostream>
#include <omp.h>

#include "declarations.hpp"
#include "chemistry.hpp"
#include "calc_column_density.hpp"
#include "reaction_rates.hpp"
#include "sundials/rate_equation_solver.hpp"
#include "write_output.hpp"


// abundances: calculate abundances for each species at each grid point
// --------------------------------------------------------------------

int chemistry (CELLS *cells, RAYS rays, SPECIES species, REACTIONS reactions)
{

  // Calculate column densities

  calc_column_densities (cells, rays, species);


  // For all cells

# pragma omp parallel                  \
  shared (cells, species, reactions)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {

    // Calculate reaction rates

    reaction_rates (cells, reactions, p);


    // Solve rate equations

    rate_equation_solver (cells, species, p);

  } // end of o loop over grid points
  } // end of OpenMP parallel region


  return (0);

}
