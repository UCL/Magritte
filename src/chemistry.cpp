// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "chemistry.hpp"
#include "calc_column_density.hpp"
#include "reaction_rates.hpp"
#include "sundials/rate_equation_solver.hpp"
#include "write_output.hpp"


// abundances: calculate abundances for each species at each grid point
// --------------------------------------------------------------------

int chemistry (long ncells, CELL *cell,
               double *temperature_gas, double *temperature_dust, double *rad_surface, double *AV,
               double *column_H2, double *column_HD, double *column_C, double *column_CO)
{

  // Calculate column densities

  calc_column_densities (NCELLS, cell, column_H2, column_HD, column_C, column_CO);


  // For all cells

# pragma omp parallel                                                 \
  shared( cell, temperature_gas, temperature_dust, rad_surface, AV,   \
          column_H2, column_HD, column_C, column_CO )                 \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long gridp = start; gridp < stop; gridp++)
  {

    // Calculate reaction rates

    reaction_rates( temperature_gas, temperature_dust, rad_surface, AV,
                    column_H2, column_HD, column_C, column_CO, gridp );


    // Solve rate equations

    rate_equation_solver(cell, gridp);


  } // end of gridp loop over grid points
  } // end of OpenMP parallel region


  return(0);

}
