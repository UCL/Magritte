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
#include "initializers.hpp"
#include "thermal_balance.hpp"
#include "chemistry.hpp"
#include "thermal_balance_iteration.hpp"
#include "update_temperature_gas.hpp"


// thermal_balance: perform thermal balance iterations to determine temperature
// ----------------------------------------------------------------------------

int thermal_balance (long ncells, CELL *cell, SPECIES *species, REACTION *reaction, LINE_SPECIES line_species,
                     double *pop, double *mean_intensity, TIMERS *timers)
{

  // PRELIMINARY CHEMISTRY ITERATIONS
  // ________________________________


  printf ("(thermal_balance): starting preliminary chemistry iterations \n\n");


    // COLUMN_DENSITIES column;

# if (FIXED_NCELLS)

    double column_H2[NCELLS*NRAYS];   // H2 column density for each ray and cell
    double column_HD[NCELLS*NRAYS];   // HD column density for each ray and cell
    double column_C[NCELLS*NRAYS];    // C  column density for each ray and cell
    double column_CO[NCELLS*NRAYS];   // CO column density for each ray and cell

# else

    // column.new_column(ncells);

    double *column_H2 = new double[ncells*NRAYS];   // H2 column density for each ray and cell
    double *column_HD = new double[ncells*NRAYS];   // HD column density for each ray and cell
    double *column_C  = new double[ncells*NRAYS];   // C  column density for each ray and cell
    double *column_CO = new double[ncells*NRAYS];   // CO column density for each ray and cell

# endif


  initialize_double_array (NCELLS*NRAYS, column_H2);
  initialize_double_array (NCELLS*NRAYS, column_HD);
  initialize_double_array (NCELLS*NRAYS, column_C);
  initialize_double_array (NCELLS*NRAYS, column_CO);


  // Preliminary chemistry iterations

  for (int chem_iteration = 0; chem_iteration < PRELIM_CHEM_ITER; chem_iteration++)
  {
    printf ("(thermal_balance):   chemistry iteration %d of %d \n", chem_iteration+1, PRELIM_CHEM_ITER);


    // Calculate chemical abundances given current temperatures and radiation field

    timers->chemistry.start();

    chemistry (NCELLS, cell, species, reaction, column_H2, column_HD, column_C, column_CO );

    timers->chemistry.stop();


    // Write intermediate output for (potential) restart

#   if   (WRITE_INTERMEDIATE_OUTPUT & (INPUT_FORMAT == '.txt'))

        write_temperature_gas ("", NCELLS, cell);
        write_temperature_dust ("", NCELLS, cell);
        write_temperature_gas_prev ("", NCELLS, cell);

#   elif (WRITE_INTERMEDIATE_OUTPUT & (INPUT_FORMAT == '.vtu'))

        write_vtu_output (NCELLS, cell, inputfile);

#   endif


  } // End of chemistry iteration


  printf ("\n(thermal_balance): preliminary chemistry iterations done \n\n");




  // PRELIMINARY THERMAL BALANCE ITERATIONS
  // ______________________________________


  printf ("(thermal_balance): calculating the minimal and maximal thermal flux \n\n");


# if (FIXED_NCELLS)

    double mean_intensity_eff[NCELLS*TOT_NRAD];   // mean intensity for a ray
    double Lambda_diagonal[NCELLS*TOT_NRAD];      // mean intensity for a ray

    double scatter_u[NCELLS*TOT_NRAD*NFREQ];      // angle averaged u scattering term
    double scatter_v[NCELLS*TOT_NRAD*NFREQ];      // angle averaged v scattering term

    double temperature_a[NCELLS];                 // variable for Brent's algorithm
    double temperature_b[NCELLS];                 // variable for Brent's algorithm
    double temperature_c[NCELLS];                 // variable for Brent's algorithm
    double temperature_d[NCELLS];                 // variable for Brent's algorithm
    double temperature_e[NCELLS];                 // variable for Brent's algorithm

    double thermal_ratio_a[NCELLS];               // variable for Brent's algorithm
    double thermal_ratio_b[NCELLS];               // variable for Brent's algorithm
    double thermal_ratio_c[NCELLS];               // variable for Brent's algorithm

    double thermal_ratio[NCELLS];

# else

    double *mean_intensity_eff = new double[ncells*TOT_NRAD];   // mean intensity for a ray
    double *Lambda_diagonal    = new double[ncells*TOT_NRAD];   // mean intensity for a ray

    double *scatter_u = new double[ncells*TOT_NRAD*NFREQ];      // angle averaged u scattering term
    double *scatter_v = new double[ncells*TOT_NRAD*NFREQ];      // angle averaged v scattering term

    double *temperature_a = new double[ncells];                 // variable for Brent's algorithm
    double *temperature_b = new double[ncells];                 // variable for Brent's algorithm
    double *temperature_c = new double[ncells];                 // variable for Brent's algorithm
    double *temperature_d = new double[ncells];                 // variable for Brent's algorithm
    double *temperature_e = new double[ncells];                 // variable for Brent's algorithm

    double *thermal_ratio_a = new double[ncells];               // variable for Brent's algorithm
    double *thermal_ratio_b = new double[ncells];               // variable for Brent's algorithm
    double *thermal_ratio_c = new double[ncells];               // variable for Brent's algorithm

    double *thermal_ratio = new double[ncells];

# endif


  initialize_double_array (NCELLS*TOT_NRAD, mean_intensity);
  initialize_double_array (NCELLS*TOT_NRAD, mean_intensity_eff);
  initialize_double_array (NCELLS*TOT_NRAD, Lambda_diagonal);

  initialize_double_array (NCELLS*TOT_NRAD*NFREQ, scatter_u);
  initialize_double_array (NCELLS*TOT_NRAD*NFREQ, scatter_v);

  initialize_double_array (NCELLS*TOT_NLEV, pop);

  initialize_double_array_with_value (NCELLS, TEMPERATURE_MIN, temperature_a);
  initialize_double_array_with_value (NCELLS, TEMPERATURE_MAX, temperature_b);

  initialize_double_array (NCELLS, temperature_c);
  initialize_double_array (NCELLS, temperature_d);
  initialize_double_array (NCELLS, temperature_e);

  initialize_double_array (NCELLS, thermal_ratio_a);
  initialize_double_array (NCELLS, thermal_ratio_b);
  initialize_double_array (NCELLS, thermal_ratio_c);

  initialize_double_array (NCELLS, thermal_ratio);


  for (int tb_iteration = 0; tb_iteration < PRELIM_TB_ITER; tb_iteration++)
  {
    printf ("(thermal_balance):   thermal balance iteration %d of %d \n", tb_iteration+1, PRELIM_TB_ITER);


    thermal_balance_iteration (NCELLS, cell, species, reaction, line_species,
                               column_H2, column_HD, column_C, column_CO,
                               pop, mean_intensity, Lambda_diagonal, mean_intensity_eff,
                               thermal_ratio, timers);


    // Average thermal ratio over neighbors

    // double tr_new[NCELLS];
    //
    //
    // for (long p = 0; p < NCELLS; p++)
    // {
    //
    //   tr_new[p] = thermal_ratio[p];
    //
    //   for (long n = 0; n < cell[p].n_neighbors; n++)
    //   {
    //     long nr = cell[p].neighbor[n];
    //
    //     tr_new[p] = tr_new[p] + 0.5 * thermal_ratio[nr];
    //   }
    //
    //   tr_new[p] = tr_new[p] / (cell[p].n_neighbors + 1);
    //
    // }


    update_temperature_gas (NCELLS, cell, thermal_ratio,
                            temperature_a, temperature_b, thermal_ratio_a, thermal_ratio_b);


    // Write intermediate output for (potential) restart

#   if   (WRITE_INTERMEDIATE_OUTPUT && (INPUT_FORMAT == '.txt'))

      write_temperature_gas ("", NCELLS, cell); // should be temperature b !!!!!!!!!!
      write_temperature_dust ("", NCELLS, cell);
      write_temperature_gas_prev ("", NCELLS, cell);

#   elif (WRITE_INTERMEDIATE_OUTPUT && (INPUT_FORMAT == '.vtu'))

      write_vtu_output (NCELLS, cell, inputfile);

#  endif


  } // end of tb_iteration loop


#   pragma omp parallel                    \
    shared (ncells, cell, temperature_b)   \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;     // Note brackets


    for (long n = start; n < stop; n++)
    {
      cell[n].temperature.gas = temperature_b[n];
    }
    } // end of OpenMP parallel region


  // write_double_1("temperature_a", "", NCELLS, temperature_a );
  // write_double_1("temperature_b", "", NCELLS, temperature_b );


  printf ("(thermal_balance): minimal and maximal thermal flux calculated \n\n");

// return(0);


  // CALCULATE THERMAL BALANCE (ITERATIVELY)
  // _______________________________________


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


    thermal_balance_iteration (NCELLS, cell, species, reaction, line_species,
                               column_H2, column_HD, column_C, column_CO,
                               pop, mean_intensity, Lambda_diagonal, mean_intensity_eff,
                               thermal_ratio, timers);


  // Average thermal ratio over neighbors

  // for (long p = 0; p < NCELLS; p++)
  // {
  //
  //   thermal_ratio_b[p] = thermal_ratio[p];
  //
  //   for (long n = 0; n < cell[p].n_neighbors; n++)
  //   {
  //     long nr = cell[p].neighbor[n];
  //
  //     thermal_ratio_b[p] = thermal_ratio_b[p] + 0.5 * thermal_ratio[nr];
  //   }
  //
  //   thermal_ratio_b[p] = thermal_ratio_b[p] / (cell[p].n_neighbors + 1);
  // }


    initialize_double_array_with (NCELLS, thermal_ratio_b, thermal_ratio);





    // Calculate thermal balance for each cell

#   pragma omp parallel                                                                        \
    shared (ncells, cell, thermal_ratio, temperature_a, temperature_b, temperature_c,          \
            temperature_d, temperature_e, thermal_ratio_a, thermal_ratio_b, thermal_ratio_c,   \
            n_not_converged, no_thermal_balance)                                               \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


    for (long gridp = start; gridp < stop; gridp++)
    {
      shuffle_Brent (gridp, temperature_a, temperature_b, temperature_c, temperature_d,
                     temperature_e, thermal_ratio_a, thermal_ratio_b, thermal_ratio_c);


      /* Check for thermal balance (convergence) */

      if (fabs(thermal_ratio[gridp]) > THERMAL_PREC)
      {
        update_temperature_gas_Brent (gridp, temperature_a, temperature_b, temperature_c,
                                      temperature_d, temperature_e, thermal_ratio_a,
                                      thermal_ratio_b, thermal_ratio_c);

        cell[gridp].temperature.gas = temperature_b[gridp];


        if (cell[gridp].temperature.gas != T_CMB)
        {
          no_thermal_balance = true;

          n_not_converged++;
        }

      }


    } /* end of gridp loop over grid points */
    } /* end of OpenMP parallel region */



    // Average over neighbors

    // double temperature_new[NCELLS];
    //
    //
    // for (long p = 0; p < NCELLS; p++)
    // {
    //
    //   temperature_new[p] = cell[p].temperature.gas;
    //
    //   for (long n = 0; n < cell[p].n_neighbors; n++)
    //   {
    //     long nr = cell[p].neighbor[n];
    //
    //     temperature_new[p] = temperature_new[p] + cell[nr].temperature.gas;
    //   }
    //
    //   temperature_new[p] = temperature_new[p] / (cell[p].n_neighbors + 1);
    //
    // }
    //
    //
    // for (long p = 0; p < NCELLS; p++)
    // {
    //   cell[p].temperature.gas = temperature_new[p];
    // }


    printf ("(thermal_balance): heating and cooling calculated \n\n");


    // Limit number of iterations

    if ( (niterations > MAX_NITERATIONS) || (n_not_converged < NCELLS/10) )
    {
      no_thermal_balance = false;
    }


    printf ("(thermal_balance): Not yet converged for %ld of %d\n", n_not_converged, NCELLS);


  } // end of thermal balance iterations


  printf ("(thermal_balance): thermal balance reached in %d iterations \n\n", niterations);


# if (!FIXED_NCELLS)

    delete [] column_H2;
    delete [] column_HD;
    delete [] column_C;
    delete [] column_CO;
    delete [] mean_intensity_eff;
    delete [] Lambda_diagonal;
    delete [] scatter_u;
    delete [] scatter_v;
    delete [] temperature_a;
    delete [] temperature_b;
    delete [] temperature_c;
    delete [] temperature_d;
    delete [] temperature_e;
    delete [] thermal_ratio_a;
    delete [] thermal_ratio_b;
    delete [] thermal_ratio_c;
    delete [] thermal_ratio;

# endif


  return(0);

}
