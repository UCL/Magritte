// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include "declarations.hpp"
#include "thermal_balance.hpp"
#include "initializers.hpp"
#include "chemistry.hpp"
#include "write_output.hpp"
#include "thermal_balance_iteration.hpp"
#include "update_temperature_gas.hpp"


// thermal_balance: perform thermal balance iterations to determine temperature
// ----------------------------------------------------------------------------

int thermal_balance (long ncells, CELLS *cells, RAYS rays, SPECIES species, REACTIONS reactions,
                     LINES lines, TIMERS *timers)
{


# if (FIXED_NCELLS)

    double column_H2[NCELLS*NRAYS];   // H2 column density for each ray and cell
    double column_HD[NCELLS*NRAYS];   // HD column density for each ray and cell
    double column_C[NCELLS*NRAYS];    // C  column density for each ray and cell
    double column_CO[NCELLS*NRAYS];   // CO column density for each ray and cell

    // double thermal_ratio[NCELLS];
    // double thermal_ratio_prev[NCELLS];

# else

    // column.new_column(ncells);

    double *column_H2 = new double[ncells*NRAYS];   // H2 column density for each ray and cell
    double *column_HD = new double[ncells*NRAYS];   // HD column density for each ray and cell
    double *column_C  = new double[ncells*NRAYS];   // C  column density for each ray and cell
    double *column_CO = new double[ncells*NRAYS];   // CO column density for each ray and cell

    // double *thermal_ratio      = new double[ncells];
    // double *thermal_ratio_prev = new double[ncells];


# endif


  initialize_double_array (NCELLS*NRAYS, column_H2);
  initialize_double_array (NCELLS*NRAYS, column_HD);
  initialize_double_array (NCELLS*NRAYS, column_C);
  initialize_double_array (NCELLS*NRAYS, column_CO);

  // initialize_double_array (NCELLS, thermal_ratio);
  // initialize_double_array (NCELLS, thermal_ratio_prev);




  // PRELIMINARY CHEMISTRY ITERATION
  // _______________________________


  for (int chem_iteration = 0; chem_iteration < PRELIM_CHEM_ITER; chem_iteration++)
  {
    printf ("(thermal_balance):   chemistry iteration %d of %d \n", chem_iteration+1, PRELIM_CHEM_ITER);


    // Calculate chemical abundances given current temperatures and radiation field

    timers->chemistry.start();

    chemistry (NCELLS, cells, rays, species, reactions, column_H2, column_HD, column_C, column_CO);

    timers->chemistry.stop();


#   if (WRITE_INTERMEDIATE_OUTPUT)

      write_output (NCELLS, cells, lines);

#   endif


  } // End of chemistry iteration




  // THERMAL BALANCE ITERATIONS
  // __________________________


  printf ("(thermal_balance): starting thermal balance iterations \n\n");


  bool no_thermal_balance = true;

  int niterations = 0;


  // Thermal balance iterations

  while (no_thermal_balance)
  {
    no_thermal_balance = false;

    niterations++;


    printf ("(thermal_balance): thermal balance iteration %d\n", niterations);


    long n_not_converged = 0;   // number of grid points that are not yet converged


    thermal_balance_iteration (NCELLS, cells, rays, species, reactions, lines,
                               column_H2, column_HD, column_C, column_CO, timers);


    // Update temperature for each cell

#   pragma omp parallel                                           \
    shared (ncells, cells, n_not_converged, no_thermal_balance)   \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


    for (long p = start; p < stop; p++)
    {

      // Check for thermal balance

      if (fabs(cells->thermal_ratio[p]) > THERMAL_PREC)
      {
        update_temperature_gas (NCELLS, cells, p);

        if (cells->temperature_gas[p] != T_CMB)
        {
          no_thermal_balance = true;
          n_not_converged++;
        }
      }

    }
    } // end of OpenMP parallel region


    // Limit number of iterations

    if ( (niterations >= MAX_TB_ITER) || (n_not_converged < NCELLS/10) )
    {
      no_thermal_balance = false;
    }


    printf ("(thermal_balance): Not yet converged for %ld of %ld\n", n_not_converged, NCELLS);


#   if (WRITE_INTERMEDIATE_OUTPUT)

      write_output (NCELLS, cells, lines);

#   endif


  } // end of tb_iteration loop


  printf ("(thermal_balance): thermal balance reached in %d iterations \n\n", niterations);


# if (!FIXED_NCELLS)

    delete [] column_H2;
    delete [] column_HD;
    delete [] column_C;
    delete [] column_CO;

# endif


  return(0);

}



//
// // thermal_balance_Brent: perform thermal balance iterations to determine temperature
// // ----------------------------------------------------------------------------------
//
// int thermal_balance_Brent (long ncells, CELL *cell, RAYS rays, SPECIES species, REACTIONS reactions,
//                            LINES lines, TIMERS *timers)
// {
//
//     // COLUMN_DENSITIES column;
//
// # if (FIXED_NCELLS)
//
//     double column_H2[NCELLS*NRAYS];   // H2 column density for each ray and cell
//     double column_HD[NCELLS*NRAYS];   // HD column density for each ray and cell
//     double column_C[NCELLS*NRAYS];    // C  column density for each ray and cell
//     double column_CO[NCELLS*NRAYS];   // CO column density for each ray and cell
//
//     // double thermal_ratio[NCELLS];
//
// # else
//
//     // column.new_column(ncells);
//
//     double *column_H2 = new double[ncells*NRAYS];   // H2 column density for each ray and cell
//     double *column_HD = new double[ncells*NRAYS];   // HD column density for each ray and cell
//     double *column_C  = new double[ncells*NRAYS];   // C  column density for each ray and cell
//     double *column_CO = new double[ncells*NRAYS];   // CO column density for each ray and cell
//
//     // double *thermal_ratio = new double[ncells];
//
//
// # endif
//
//
//   initialize_double_array (NCELLS*NRAYS, column_H2);
//   initialize_double_array (NCELLS*NRAYS, column_HD);
//   initialize_double_array (NCELLS*NRAYS, column_C);
//   initialize_double_array (NCELLS*NRAYS, column_CO);
//
//   // initialize_double_array (NCELLS, thermal_ratio);
//
//
//
//
//   // CALCULATE THERMAL BALANCE (ITERATIVELY)
//   // _______________________________________
//
//
//   printf ("(thermal_balance): starting thermal balance iterations \n\n");
//
//
// # if (FIXED_NCELLS)
//
//     double temperature_a[NCELLS];                 // variable for Brent's algorithm
//     double temperature_b[NCELLS];                 // variable for Brent's algorithm
//     double temperature_c[NCELLS];                 // variable for Brent's algorithm
//     double temperature_d[NCELLS];                 // variable for Brent's algorithm
//     double temperature_e[NCELLS];                 // variable for Brent's algorithm
//
//     double thermal_ratio_a[NCELLS];               // variable for Brent's algorithm
//     double thermal_ratio_b[NCELLS];               // variable for Brent's algorithm
//     double thermal_ratio_c[NCELLS];               // variable for Brent's algorithm
//
// # else
//
//     double *temperature_a = new double[ncells];                 // variable for Brent's algorithm
//     double *temperature_b = new double[ncells];                 // variable for Brent's algorithm
//     double *temperature_c = new double[ncells];                 // variable for Brent's algorithm
//     double *temperature_d = new double[ncells];                 // variable for Brent's algorithm
//     double *temperature_e = new double[ncells];                 // variable for Brent's algorithm
//
//     double *thermal_ratio_a = new double[ncells];               // variable for Brent's algorithm
//     double *thermal_ratio_b = new double[ncells];               // variable for Brent's algorithm
//     double *thermal_ratio_c = new double[ncells];               // variable for Brent's algorithm
//
// # endif
//
//
// // Initialize temperature_a and temperature_b
//
// # pragma omp parallel                                                                     \
//   shared (ncells, cell, temperature_a, temperature_b, thermal_ratio_a, thermal_ratio_b)   \
//   default (none)
//   {
//
//   int num_threads = omp_get_num_threads();
//   int thread_num  = omp_get_thread_num();
//
//   long start = (thread_num*NCELLS)/num_threads;
//   long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets
//
//
//   for (long n = start; n < stop; n++)
//   {
//
//     // if (cell[n].thermal_ratio_prev > 0.0)
//     // {
//       temperature_a[n] = cell[n].temperature.gas_prev;
//       temperature_b[n] = cell[n].temperature.gas;
//
//     thermal_ratio_b[n] = cell[n].thermal_ratio;
//
//
//   }
//   } // end of OpenMP parallel region
//
//
//   initialize_double_array (NCELLS, temperature_c);
//   initialize_double_array (NCELLS, temperature_d);
//   initialize_double_array (NCELLS, temperature_e);
//
//   // initialize_double_array (NCELLS, thermal_ratio_a);
//   // initialize_double_array (NCELLS, thermal_ratio_b);
//   initialize_double_array (NCELLS, thermal_ratio_c);
//
//
// // # pragma omp parallel                    \
// //   shared (ncells, cell, temperature_b)   \
// //   default (none)
// //   {
// //
// //   int num_threads = omp_get_num_threads();
// //   int thread_num  = omp_get_thread_num();
// //
// //   long start = (thread_num*NCELLS)/num_threads;
// //   long stop  = ((thread_num+1)*NCELLS)/num_threads;     // Note brackets
// //
// //
// //   for (long n = start; n < stop; n++)
// //   {
// //     cell[n].temperature.gas = temperature_b[n];
// //   }
// //   } // end of OpenMP parallel region
//
//
//   bool no_thermal_balance = true;
//
//   int niterations = 0;
//
//
//   // Thermal balance iterations
//
//   while (no_thermal_balance)
//   {
//     no_thermal_balance = false;
//
//     niterations++;
//
//
//     printf ("(thermal_balance): thermal balance iteration %d\n", niterations);
//
//
//     long n_not_converged = 0;   // number of grid points that are not yet converged
//
//
//     thermal_balance_iteration (NCELLS, cell, rays, species, reactions, lines,
//                                column_H2, column_HD, column_C, column_CO, timers);
//
//
//     // initialize_double_array_with (NCELLS, thermal_ratio_b, thermal_ratio);
//
//
// #   pragma omp parallel                      \
//     shared (ncells, cell, thermal_ratio_b)   \
//     default (none)
//     {
//
//       int num_threads = omp_get_num_threads();
//       int thread_num  = omp_get_thread_num();
//
//       long start = (thread_num*NCELLS)/num_threads;
//       long stop  = ((thread_num+1)*NCELLS)/num_threads;     // Note brackets
//
//
//       for (long n = start; n < stop; n++)
//       {
//         thermal_ratio_b[n] = cell[n].thermal_ratio;
//       }
//     } // end of OpenMP parallel region
//
//
//
//     // Update temperature for each cell
//
// #   pragma omp parallel                                                                        \
//     shared (ncells, cell, temperature_a, temperature_b, temperature_c,                         \
//             temperature_d, temperature_e, thermal_ratio_a, thermal_ratio_b, thermal_ratio_c,   \
//             n_not_converged, no_thermal_balance)                                               \
//     default (none)
//     {
//
//     int num_threads = omp_get_num_threads();
//     int thread_num  = omp_get_thread_num();
//
//     long start = (thread_num*NCELLS)/num_threads;
//     long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets
//
//
//     for (long o = start; o < stop; o++)
//     {
//       shuffle_Brent (o, temperature_a, temperature_b, temperature_c, temperature_d,
//                      temperature_e, thermal_ratio_a, thermal_ratio_b, thermal_ratio_c);
//
//
//       // Check for thermal balance (convergence)
//
//       if (fabs(cell[o].thermal_ratio) > THERMAL_PREC)
//       {
//         update_temperature_gas_Brent (o, temperature_a, temperature_b, temperature_c,
//                                       temperature_d, temperature_e, thermal_ratio_a,
//                                       thermal_ratio_b, thermal_ratio_c);
//
//         cell[o].temperature.gas = temperature_b[o];
//
//         if (cell[o].temperature.gas != T_CMB)
//         {
//           no_thermal_balance = true;
//           n_not_converged++;
//         }
//       }
//
//     } // end of o loop over grid points
//     } // end of OpenMP parallel region
//
//
//     printf ("(thermal_balance): heating and cooling calculated \n\n");
//
//
//     // Limit number of iterations
//
//     if ( (niterations > MAX_NITERATIONS) || (n_not_converged < NCELLS/10) )
//     {
//       no_thermal_balance = false;
//     }
//
//
//     printf ("(thermal_balance): Not yet converged for %ld of %d\n", n_not_converged, NCELLS);
//
//
//   } // end of thermal balance iterations
//
//
//   printf ("(thermal_balance): thermal balance reached in %d iterations \n\n", niterations);
//
//
// # if (!FIXED_NCELLS)
//
//     delete [] column_H2;
//     delete [] column_HD;
//     delete [] column_C;
//     delete [] column_CO;
//
//     delete [] temperature_a;
//     delete [] temperature_b;
//     delete [] temperature_c;
//     delete [] temperature_d;
//     delete [] temperature_e;
//
//     delete [] thermal_ratio_a;
//     delete [] thermal_ratio_b;
//     delete [] thermal_ratio_c;
//
// # endif
//
//
//   return(0);
//
// }
