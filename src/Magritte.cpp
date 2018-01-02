// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
/*#include <mpi.h>*/

#include <string>
#include <iostream>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "definitions.hpp"

#include "initializers.hpp"
#include "species_tools.hpp"

#include "read_input.hpp"
#include "read_chemdata.hpp"
#include "read_linedata.hpp"

#include "ray_tracing.hpp"

#include "calc_rad_surface.hpp"
#include "calc_column_density.hpp"
#include "calc_AV.hpp"
#include "calc_UV_field.hpp"
#include "calc_temperature_dust.hpp"
#include "chemistry.hpp"
#include "thermal_balance.hpp"
#include "update_temperature_gas.hpp"

#include "write_output.hpp"
#include "write_txt_tools.hpp"
#include "write_vtu_tools.hpp"



// main for Magritte
// -----------------

int main ()
{

  // Initialize all timers

  double time_total       = 0.0;   // total time in Magritte
  double time_ray_tracing = 0.0;   // total time in ray_tracing
  double time_chemistry   = 0.0;   // total time in abundances
  double time_level_pop   = 0.0;   // total time in level_populations

  time_total -= omp_get_wtime();


  printf ("                                                                          \n");
  printf ("Magritte: Multidimensional Accelerated General-purpose Radiative Transfer \n");
  printf ("                                                                          \n");
  printf ("Developed by: Frederik De Ceuster - University College London & KU Leuven \n");
  printf ("_________________________________________________________________________ \n");
  printf ("                                                                          \n");
  printf ("                                                                          \n");




  // READ GRID INPUT
  // _______________


  printf ("(Magritte): reading grid input \n");


  // Define cells (using types defined in declarations.hpp

  CELL cell[NCELLS];

  double temperature_gas[NCELLS];                          /* gas temperature at each grid point */

  double prev_temperature_gas[NCELLS];

  double temperature_dust[NCELLS];                 /* temperature of the dust at each grid point */


  // Read input file

# if   (INPUT_FORMAT == '.vtu')

    read_vtu_input (inputfile, NCELLS, cell, temperature_gas,
                    temperature_dust, prev_temperature_gas);

# elif (INPUT_FORMAT == '.txt')

    read_txt_input (inputfile, NCELLS, cell, temperature_gas,
                    temperature_dust, prev_temperature_gas);

# endif


  printf ("(Magritte): grid input read \n\n");




  // READ CHEMISTRY DATA
  // ___________________


  printf ("(Magritte): reading chemistry data \n");


  // Read chemical species data

  double initial_abn[NSPEC];

  read_species (spec_datafile, initial_abn);


  // Get and store the species numbers of some inportant species

  e_nr    = get_species_nr ("e-");     // species nr corresponding to electrons
  H2_nr   = get_species_nr ("H2");     // species nr corresponding to H2
  HD_nr   = get_species_nr ("HD");     // species nr corresponding to HD
  C_nr    = get_species_nr ("C");      // species nr corresponding to C
  H_nr    = get_species_nr ("H");      // species nr corresponding to H
  H2x_nr  = get_species_nr ("H2+");    // species nr corresponding to H2+
  HCOx_nr = get_species_nr ("HCO+");   // species nr corresponding to HCO+
  H3x_nr  = get_species_nr ("H3+");    // species nr corresponding to H3+
  H3Ox_nr = get_species_nr ("H3O+");   // species nr corresponding to H3O+
  Hex_nr  = get_species_nr ("He+");    // species nr corresponding to He+
  CO_nr   = get_species_nr ("CO");     // species nr corresponding to CO


  // Read chemical reaction data

  read_reactions (reac_datafile);


  printf ("(Magritte): chemistry data read \n\n");




  // DECLARE AND INITIALIZE LINE VARIABLES
  // _____________________________________


  printf ("(Magritte): declaring and initializing line variables \n");


  // Define line related variables

  int irad[TOT_NRAD];           /* level index corresponding to radiative transition [0..nrad-1] */

  initialize_int_array(irad, TOT_NRAD);

  int jrad[TOT_NRAD];           /* level index corresponding to radiative transition [0..nrad-1] */

  initialize_int_array(jrad, TOT_NRAD);

  double energy[TOT_NLEV];                                                /* energy of the level */

  initialize_double_array(energy, TOT_NLEV);

  double weight[TOT_NLEV];                                    /* statistical weight of the level */

  initialize_double_array(weight, TOT_NLEV);

  double frequency[TOT_NLEV2];             /* photon frequency corresponing to i -> j transition */

  initialize_double_array(frequency, TOT_NLEV2);

  double A_coeff[TOT_NLEV2];                                        /* Einstein A_ij coefficient */

  initialize_double_array(A_coeff, TOT_NLEV2);

  double B_coeff[TOT_NLEV2];                                        /* Einstein B_ij coefficient */

  initialize_double_array(B_coeff, TOT_NLEV2);




  /* Define the collision related variables */

  double coltemp[TOT_CUM_TOT_NCOLTEMP];               /* Collision temperatures for each partner */
                                                                   /*[NLSPEC][ncolpar][ncoltemp] */
  initialize_double_array(coltemp, TOT_CUM_TOT_NCOLTEMP);

  double C_data[TOT_CUM_TOT_NCOLTRANTEMP];           /* C_data for each partner, tran. and temp. */
                                                        /* [NLSPEC][ncolpar][ncoltran][ncoltemp] */
  initialize_double_array(C_data, TOT_CUM_TOT_NCOLTRANTEMP);

  int icol[TOT_CUM_TOT_NCOLTRAN];     /* level index corresp. to col. transition [0..ncoltran-1] */
                                                                  /* [NLSPEC][ncolpar][ncoltran] */
  initialize_int_array(icol, TOT_CUM_TOT_NCOLTRAN);

  int jcol[TOT_CUM_TOT_NCOLTRAN];     /* level index corresp. to col. transition [0..ncoltran-1] */
                                                                  /* [NLSPEC][ncolpar][ncoltran] */
  initialize_int_array(jcol, TOT_CUM_TOT_NCOLTRAN);


  // Define helper arrays specifying species of collisiopn partners

  initialize_int_array (spec_par, TOT_NCOLPAR);

  initialize_char_array (ortho_para, TOT_NCOLPAR);


  printf("(Magritte): data structures are set up \n\n");




  // READ LINE DATA FOR EACH LINE PRODUCING SPECIES
  // ______________________________________________


  printf ("(Magritte): reading line data \n");


  // Read the line data files stored in the list(!) line_data

  read_linedata (line_datafile, irad, jrad, energy, weight, frequency,
                 A_coeff, B_coeff, coltemp, C_data, icol, jcol);


  printf ("(Magritte): line data read \n\n");




# if (CELL_BASED)


    // FIND NEIGHBORING CELLS
    // ______________________


    printf ("(Magritte): finding neighboring cells \n");


    // Find for each cell the neighboring cells

    find_neighbors (NCELLS, cell);


    printf ("(Magritte): neighboring cells found \n\n");

  //   long origin = 0;
  //   long ray = 0;
  //
  // // Walk along ray
  // {
  //   double Z   = 0.0;
  //   double dZ  = 0.0;
  //
  //   long current = origin;
  //   long next    = next_cell(NCELLS, cell, origin, ray, Z, current, &dZ);
  //
  //
  //   while (next != NCELLS)
  //   {
  //     Z = Z + dZ;
  //
  //     printf("current %ld\n", current);
  //
  //     current = next;
  //     next    = next_cell(NCELLS, cell, origin, ray, Z, current, &dZ);
  //   }
  // }
  //
  // return(0);

# endif




  // CALCULATE EXTERNAL RADIATION FIELD
  // __________________________________


  printf ("(Magritte): calculating external radiation field \n");


  double G_external[3];   // external radiation field vector

  G_external[0] = G_EXTERNAL_X;
  G_external[1] = G_EXTERNAL_Y;
  G_external[2] = G_EXTERNAL_Z;


  double rad_surface[NCELLS*NRAYS];

  initialize_double_array (rad_surface, NCELLS*NRAYS);


  // Calculate radiation surface

  calc_rad_surface (NCELLS, G_external, rad_surface);

  printf ("(Magritte): external radiation field calculated \n\n");




  // MAKE GUESS FOR GAS TEMPERATURE AND CALCULATE DUST TEMPERATURE
  // _____________________________________________________________


  printf("(Magritte): making a guess for gas temperature and calculating dust temperature \n");


  double column_tot[NCELLS*NRAYS];   // total column density

  initialize_double_array (column_tot, NCELLS*NRAYS);

  double AV[NCELLS*NRAYS];           // Visual extinction (only takes into account H)

  initialize_double_array (AV, NCELLS*NRAYS);

  double UV_field[NCELLS];           // External UV field

  initialize_double_array (UV_field, NCELLS);


  // Calculate total column density

  calc_column_density (NCELLS, cell, column_tot, NSPEC-1);

  write_double_2("column_tot", "", NCELLS, NRAYS, column_tot);


  // Calculate visual extinction

  calc_AV (NCELLS, column_tot, AV);


  // Calculcate UV field

  calc_UV_field (NCELLS, AV, rad_surface, UV_field);


# if (!RESTART)

    // Make a guess for gas temperature based on UV field

    guess_temperature_gas (NCELLS, UV_field, temperature_gas);


    // Calculate the dust temperature

    calc_temperature_dust (NCELLS, UV_field, rad_surface, temperature_dust);

# endif


  printf ("(Magritte): gas temperature guessed and dust temperature calculated \n\n");




  // PRELIMINARY CHEMISTRY ITERATIONS
  // ________________________________


  printf("(Magritte): starting preliminary chemistry iterations \n\n");


  double column_H2[NCELLS*NRAYS];   // H2 column density for each ray and grid point

  initialize_double_array (column_H2, NCELLS*NRAYS);

  double column_HD[NCELLS*NRAYS];   // HD column density for each ray and grid point

  initialize_double_array (column_HD, NCELLS*NRAYS);

  double column_C[NCELLS*NRAYS];    // C column density for each ray and grid point

  initialize_double_array (column_C, NCELLS*NRAYS);

  double column_CO[NCELLS*NRAYS];   // CO column density for each ray and grid point

  initialize_double_array (column_CO, NCELLS*NRAYS);


  // Preliminary chemistry iterations

  for (int chem_iteration = 0; chem_iteration < PRELIM_CHEM_ITER; chem_iteration++)
  {
    printf("(Magritte):   chemistry iteration %d of %d \n", chem_iteration+1, PRELIM_CHEM_ITER);


    // Calculate chemical abundances given current temperatures and radiation field

    time_chemistry -= omp_get_wtime();

    chemistry (NCELLS, cell, temperature_gas, temperature_dust, rad_surface, AV,
               column_H2, column_HD, column_C, column_CO );

    time_chemistry += omp_get_wtime();


    // Write intermediate output for (potential) restart

#   if (WRITE_INTERMEDIATE_OUTPUT)

#     if   (INPUT_FORMAT == '.txt')

        write_temperature_gas ("", temperature_gas);
        write_temperature_dust ("", temperature_dust);
        write_prev_temperature_gas ("", prev_temperature_gas);

#     elif (INPUT_FORMAT == '.vtu')

        write_vtu_output (inputfile, temperature_gas, temperature_dust, prev_temperature_gas);

#     endif

#   endif


  } // End of chemistry iteration


  printf ("\n(Magritte): preliminary chemistry iterations done \n\n");




  // PRELIMINARY THERMAL BALANCE ITERATIONS
  // ______________________________________


  printf ("(Magritte): calculating the minimal and maximal thermal flux \n\n");


  double mean_intensity[NCELLS*TOT_NRAD];                             /* mean intensity for a ray */

  initialize_double_array(mean_intensity, NCELLS*TOT_NRAD);


  double mean_intensity_eff[NCELLS*TOT_NRAD];                         /* mean intensity for a ray */

  initialize_double_array(mean_intensity_eff, NCELLS*TOT_NRAD);

  double Lambda_diagonal[NCELLS*TOT_NRAD];                            /* mean intensity for a ray */

  initialize_double_array(Lambda_diagonal, NCELLS*TOT_NRAD);

  double scatter_u[NCELLS*TOT_NRAD*NFREQ];                    /* angle averaged u scattering term */

  initialize_double_array(scatter_u, NCELLS*TOT_NRAD*NFREQ);

  double scatter_v[NCELLS*TOT_NRAD*NFREQ];                    /* angle averaged v scattering term */

  initialize_double_array(scatter_v, NCELLS*TOT_NRAD*NFREQ);

  double pop[NCELLS*TOT_NLEV];                                            /* level population n_i */

  initialize_double_array(pop, NCELLS*TOT_NLEV);

  double temperature_a[NCELLS];                                 /* variable for Brent's algorithm */

  initialize_double_array(temperature_a, NCELLS);

  double temperature_b[NCELLS];                                 /* variable for Brent's algorithm */

  initialize_double_array(temperature_b, NCELLS);

  double temperature_c[NCELLS];                                 /* variable for Brent's algorithm */

  initialize_double_array(temperature_c, NCELLS);

  double temperature_d[NCELLS];                                 /* variable for Brent's algorithm */

  initialize_double_array(temperature_d, NCELLS);

  double temperature_e[NCELLS];                                 /* variable for Brent's algorithm */

  initialize_double_array(temperature_e, NCELLS);

  double thermal_ratio_a[NCELLS];                               /* variable for Brent's algorithm */

  initialize_double_array(thermal_ratio_a, NCELLS);

  double thermal_ratio_b[NCELLS];                               /* variable for Brent's algorithm */

  initialize_double_array(thermal_ratio_b, NCELLS);

  double thermal_ratio_c[NCELLS];                               /* variable for Brent's algorithm */

  initialize_double_array(thermal_ratio_c, NCELLS);


  double thermal_ratio[NCELLS];

  initialize_double_array(thermal_ratio, NCELLS);



  for (int tb_iteration = 0; tb_iteration < PRELIM_TB_ITER; tb_iteration++)
  {
    printf("(Magritte):   thermal balance iteration %d of %d \n", tb_iteration+1, PRELIM_TB_ITER);


    thermal_balance (cell, column_H2, column_HD, column_C, column_CO, UV_field,
                     temperature_gas, temperature_dust, rad_surface, AV, irad, jrad, energy,
                     weight, frequency, A_coeff, B_coeff, C_data, coltemp, icol, jcol, pop,
                     mean_intensity, Lambda_diagonal, mean_intensity_eff, thermal_ratio,
                     initial_abn, &time_chemistry, &time_level_pop);


    initialize_double_array_with (thermal_ratio_b, thermal_ratio, NCELLS);


    update_temperature_gas (thermal_ratio, temperature_gas, prev_temperature_gas,
                            temperature_a, temperature_b, thermal_ratio_a, thermal_ratio_b);


    // Write intermediate output for (potential) restart

#   if (WRITE_INTERMEDIATE_OUTPUT)

#     if   (INPUT_FORMAT == '.txt')

        write_temperature_gas ("", temperature_gas);
        write_temperature_dust ("", temperature_dust);
        write_prev_temperature_gas ("", prev_temperature_gas);

#     elif (INPUT_FORMAT == '.vtu')

        write_vtu_output (inputfile, temperature_gas, temperature_dust, prev_temperature_gas);

#     endif

#   endif


  } // end of tb_iteration loop over preliminary tb iterations


  initialize_double_array_with(temperature_gas, temperature_b, NCELLS);


  // write_double_1("temperature_a", "", NCELLS, temperature_a );
  // write_double_1("temperature_b", "", NCELLS, temperature_b );


  printf ("(Magritte): minimal and maximal thermal flux calculated \n\n");

// return(0);


  // CALCULATE THERMAL BALANCE (ITERATIVELY)
  // _______________________________________


  printf ("(Magritte): starting thermal balance iterations \n\n");


  bool no_thermal_balance = true;

  int niterations = 0;


  // Thermal balance iterations

  while (no_thermal_balance)
  {
    no_thermal_balance = false;

    niterations++;


    printf ("(Magritte): thermal balance iteration %d\n", niterations);


    long n_not_converged = 0;   // number of grid points that are not yet converged


    thermal_balance (cell, column_H2, column_HD, column_C, column_CO, UV_field,
                     temperature_gas, temperature_dust, rad_surface, AV, irad, jrad, energy,
                     weight, frequency, A_coeff, B_coeff, C_data, coltemp, icol, jcol, pop,
                     mean_intensity, Lambda_diagonal, mean_intensity_eff, thermal_ratio,
                     initial_abn, &time_chemistry, &time_level_pop);


    initialize_double_array_with (thermal_ratio_b, thermal_ratio, NCELLS);



    // Calculate thermal balance for each cell

#   pragma omp parallel                                                                           \
    shared( thermal_ratio, temperature_gas, prev_temperature_gas, temperature_a, temperature_b,   \
            temperature_c, temperature_d, temperature_e, thermal_ratio_a, thermal_ratio_b,        \
            thermal_ratio_c, n_not_converged, no_thermal_balance )                                \
    default( none )
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

        temperature_gas[gridp] = temperature_b[gridp];


        if (temperature_gas[gridp] != T_CMB)
        {
          no_thermal_balance = true;

          n_not_converged++;
        }

      }


    } /* end of gridp loop over grid points */
    } /* end of OpenMP parallel region */


    // if (no_thermal_balance)
    // {
    //   update_temperature_gas (thermal_ratio, temperature_gas, prev_temperature_gas,
    //                           temperature_a, temperature_b, thermal_ratio_a, thermal_ratio_b);
    // }


    printf ("(Magritte): heating and cooling calculated \n\n");


    // Limit number of iterations

    if ( (niterations > MAX_NITERATIONS) || (n_not_converged < NCELLS/10) )
    {
      no_thermal_balance = false;
    }


    printf ("(Magritte): Not yet converged for %ld of %d\n", n_not_converged, NCELLS);


  } // end of thermal balance iterations


  printf ("(Magritte): thermal balance reached in %d iterations \n\n", niterations);




  time_total += omp_get_wtime();




  // WRITE OUTPUT
  // ____________


  printf("(Magritte): writing output \n");


# if   (INPUT_FORMAT == '.vtu')

  write_vtu_output (inputfile, temperature_gas, temperature_dust, prev_temperature_gas);

# elif (INPUT_FORMAT == '.txt')

  write_txt_output (pop, mean_intensity, temperature_gas, temperature_dust);

# endif


  write_performance_log (time_total, time_level_pop, time_chemistry, time_ray_tracing, niterations);


  printf("(Magritte): output written \n\n");




  printf ("(Magritte): done \n\n");


  return (0);

}
