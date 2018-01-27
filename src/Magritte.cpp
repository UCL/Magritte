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

#include "../setup/setup_data_tools.hpp"

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

  TIMERS timers;

  timers.initialize();


  timers.total.start();




  printf ("                                                                         \n");
  printf ("Magritte: Multidimensional Accelerated General-purpose Radiative Transfer\n");
  printf ("                                                                         \n");
  printf ("Developed by: Frederik De Ceuster - University College London & KU Leuven\n");
  printf ("_________________________________________________________________________\n");
  printf ("                                                                         \n");
  printf ("                                                                         \n");




  // READ GRID INPUT
  // _______________


  printf ("(Magritte): reading grid input \n");


  // Define cells (using types defined in declarations.hpp

# if   (FIXED_NCELLS)

    long ncells = NCELLS;

    CELL cell[NCELLS];

# elif (!FIXED_NCELLS && (INPUT_FORMAT == '.txt'))

    long ncells = get_NCELLS_txt (inputfile);

    CELL *cell = new CELL[ncells];

# elif (!FIXED_NCELLS && (INPUT_FORMAT == '.vtu'))

    long ncells = get_NCELLS_vtu (inputfile);

    CELL *cell = new CELL[ncells];

# endif


  initialize_cells (NCELLS, cell);


  // Read input file

# if   (INPUT_FORMAT == '.vtu')

    read_vtu_input (inputfile, NCELLS, cell);

# elif (INPUT_FORMAT == '.txt')

    read_txt_input (inputfile, NCELLS, cell);

# endif


  printf ("(Magritte): grid input read \n\n");




  // READ CHEMISTRY DATA
  // ___________________


  printf ("(Magritte): reading chemistry data \n");


  // Read chemical species data

  SPECIES species[NSPEC];

  read_species (spec_datafile, species);


  // Initialize abundances in each cell with initial abundances read above

  initialize_abundances (NCELLS, cell, species);


  // Read chemical reaction data

  REACTION reaction[NREAC];

  read_reactions (reac_datafile, reaction);


  printf ("(Magritte): chemistry data read \n\n");




  // READ LINE DATA FOR EACH LINE PRODUCING SPECIES
  // ______________________________________________


  printf ("(Magritte): reading line data \n");


  // Read line data files stored in list(!) line_data

  LINE_SPECIES line_species;

  read_linedata (line_datafile, &line_species, species);


  printf ("(Magritte): line data read \n\n");




# if (CELL_BASED)

    // FIND NEIGHBORING CELLS
    // ______________________


    printf ("(Magritte): finding neighboring cells \n");


    // Find neighboring cells for each cell

    find_neighbors (NCELLS, cell);


    // Find endpoint of each ray for each cell

    find_endpoints (NCELLS, cell);


    printf ("(Magritte): neighboring cells found \n\n");


    // for (long p = 0; p < NCELLS; p++)
    // {
    //   printf("neighbors %ld   %ld\n", cell[p].neighbor[0], cell[p].neighbor[1]);
    //   if (cell[p].boundary) printf("boundary at %ld\n", p);
    // }
    //
    // for (long p = 0; p < NCELLS; p++)
    // {
    //   printf("end %ld   %ld\n", cell[p].endpoint[0], cell[p].endpoint[1]);
    // }

# endif

// return(0);




  // CALCULATE EXTERNAL RADIATION FIELD
  // __________________________________


  printf ("(Magritte): calculating external radiation field \n");


  double G_external[3];   // external radiation field vector

  G_external[0] = G_EXTERNAL_X;
  G_external[1] = G_EXTERNAL_Y;
  G_external[2] = G_EXTERNAL_Z;


# if (FIXED_NCELLS)

    double rad_surface[NCELLS*NRAYS];

# else

    double *rad_surface = new double[ncells*NRAYS];

# endif


  initialize_double_array (NCELLS*NRAYS, rad_surface);


  // Calculate radiation surface

  calc_rad_surface (NCELLS, G_external, rad_surface);

  printf ("(Magritte): external radiation field calculated \n\n");




  // MAKE GUESS FOR GAS TEMPERATURE AND CALCULATE DUST TEMPERATURE
  // _____________________________________________________________


  printf("(Magritte): making a guess for gas temperature and calculating dust temperature \n");


# if (FIXED_NCELLS)

    double column_tot[NCELLS*NRAYS];   // total column density
    double AV[NCELLS*NRAYS];           // Visual extinction
    double UV_field[NCELLS];           // External UV field

# else

    double *column_tot = new double[ncells*NRAYS];   // total column density
    double *AV         = new double[ncells*NRAYS];   // Visual extinction
    double *UV_field   = new double[ncells];         // External UV field

# endif


  initialize_double_array (NCELLS*NRAYS, column_tot);
  initialize_double_array (NCELLS*NRAYS, AV);
  initialize_double_array (NCELLS, UV_field);


  // Calculate total column density

  calc_column_density (NCELLS, cell, column_tot, NSPEC-1);
  // write_double_2("column_tot", "", NCELLS, NRAYS, column_tot);


  // Calculate visual extinction

  calc_AV (NCELLS, column_tot, AV);


  // Calculcate UV field

  calc_UV_field (NCELLS, AV, rad_surface, UV_field);


# if (!RESTART)

    // Make a guess for gas temperature based on UV field

    guess_temperature_gas (NCELLS, cell, UV_field);

    // for (long n = 0; n < NCELLS; n++)
    // {
    //   std::cout << cell[n].temperature.gas << "\n";
    // }

    // Calculate the dust temperature

    calc_temperature_dust (NCELLS, cell, UV_field, rad_surface);

# endif


  printf ("(Magritte): gas temperature guessed and dust temperature calculated \n\n");




  // PRELIMINARY CHEMISTRY ITERATIONS
  // ________________________________


  printf("(Magritte): starting preliminary chemistry iterations \n\n");


    COLUMN_DENSITIES column;

# if (FIXED_NCELLS)

    double column_H2[NCELLS*NRAYS];   // H2 column density for each ray and cell
    double column_HD[NCELLS*NRAYS];   // HD column density for each ray and cell
    double column_C[NCELLS*NRAYS];    // C  column density for each ray and cell
    double column_CO[NCELLS*NRAYS];   // CO column density for each ray and cell

# else

    column.new_column(ncells);

    double *column_H2 = new double[ncells*NRAYS];   // H2 column density for each ray and cell
    double *column_HD = new double[ncells*NRAYS];   // HD column density for each ray and cell
    double *column_C  = new double[ncells*NRAYS];   // C  column density for each ray and cell
    double *column_CO = new double[ncells*NRAYS];   // CO column density for each ray and cell



# endif


  initialize_double_array (NCELLS*NRAYS, column.H2);
  initialize_double_array (NCELLS*NRAYS, column.HD);
  initialize_double_array (NCELLS*NRAYS, column.C);
  initialize_double_array (NCELLS*NRAYS, column.CO);


  // Preliminary chemistry iterations

  for (int chem_iteration = 0; chem_iteration < PRELIM_CHEM_ITER; chem_iteration++)
  {
    printf("(Magritte):   chemistry iteration %d of %d \n", chem_iteration+1, PRELIM_CHEM_ITER);


    // Calculate chemical abundances given current temperatures and radiation field

    timers.chemistry.start();

    chemistry (NCELLS, cell, species, reaction, rad_surface, AV, column_H2, column_HD, column_C, column_CO );

    timers.chemistry.stop();


    // Write intermediate output for (potential) restart

#   if   (WRITE_INTERMEDIATE_OUTPUT & (INPUT_FORMAT == '.txt'))

        write_temperature_gas ("", NCELLS, cell);
        write_temperature_dust ("", NCELLS, cell);
        write_temperature_gas_prev ("", NCELLS, cell);

#   elif (WRITE_INTERMEDIATE_OUTPUT & (INPUT_FORMAT == '.vtu'))

        write_vtu_output (NCELLS, cell, inputfile);

#   endif


  } // End of chemistry iteration


  printf ("\n(Magritte): preliminary chemistry iterations done \n\n");




  // PRELIMINARY THERMAL BALANCE ITERATIONS
  // ______________________________________


  printf ("(Magritte): calculating the minimal and maximal thermal flux \n\n");


# if (FIXED_NCELLS)

    double mean_intensity[NCELLS*TOT_NRAD];       // mean intensity for a ray
    double mean_intensity_eff[NCELLS*TOT_NRAD];   // mean intensity for a ray
    double Lambda_diagonal[NCELLS*TOT_NRAD];      // mean intensity for a ray

    double scatter_u[NCELLS*TOT_NRAD*NFREQ];      // angle averaged u scattering term
    double scatter_v[NCELLS*TOT_NRAD*NFREQ];      // angle averaged v scattering term

    double pop[NCELLS*TOT_NLEV];                  // level population n_i

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

    double *mean_intensity     = new double[ncells*TOT_NRAD];   // mean intensity for a ray
    double *mean_intensity_eff = new double[ncells*TOT_NRAD];   // mean intensity for a ray
    double *Lambda_diagonal    = new double[ncells*TOT_NRAD];   // mean intensity for a ray

    double *scatter_u = new double[ncells*TOT_NRAD*NFREQ];      // angle averaged u scattering term
    double *scatter_v = new double[ncells*TOT_NRAD*NFREQ];      // angle averaged v scattering term

    double *pop = new double[ncells*TOT_NLEV];                  // level population n_i

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
    printf("(Magritte):   thermal balance iteration %d of %d \n", tb_iteration+1, PRELIM_TB_ITER);


    thermal_balance (NCELLS, cell, species, reaction, line_species,
                     column_H2, column_HD, column_C, column_CO, UV_field,
                     rad_surface, AV, pop, mean_intensity, Lambda_diagonal, mean_intensity_eff,
                     thermal_ratio, &timers);


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


    for (long n = 0; n < NCELLS; n++)
    {
      std::cout << cell[n].temperature.gas << "\n";
    }

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


    thermal_balance (NCELLS, cell, species, reaction, line_species,
                     column_H2, column_HD, column_C, column_CO, UV_field,
                     rad_surface, AV, pop, mean_intensity, Lambda_diagonal, mean_intensity_eff,
                     thermal_ratio, &timers);


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


    printf ("(Magritte): heating and cooling calculated \n\n");


    // Limit number of iterations

    if ( (niterations > MAX_NITERATIONS) || (n_not_converged < NCELLS/10) )
    {
      no_thermal_balance = false;
    }


    printf ("(Magritte): Not yet converged for %ld of %d\n", n_not_converged, NCELLS);


  } // end of thermal balance iterations


  printf ("(Magritte): thermal balance reached in %d iterations \n\n", niterations);




  timers.total.stop();




  // WRITE OUTPUT
  // ____________


  printf("(Magritte): writing output \n");


# if   (INPUT_FORMAT == '.vtu')

  write_vtu_output (NCELLS, cell, inputfile);

# elif (INPUT_FORMAT == '.txt')

  write_txt_output (NCELLS, cell, line_species, pop, mean_intensity);

# endif


  write_performance_log (timers, niterations);


  printf("(Magritte): output written \n\n");




# if (!FIXED_NCELLS)

    delete [] cell;
    delete [] rad_surface;
    delete [] column_tot;
    delete [] AV;
    delete [] UV_field;
    delete [] column_H2;
    delete [] column_HD;
    delete [] column_C;
    delete [] column_CO;
    delete [] mean_intensity;
    delete [] mean_intensity_eff;
    delete [] Lambda_diagonal;
    delete [] scatter_u;
    delete [] scatter_v;
    delete [] pop;
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




  printf ("(Magritte): done \n\n");


  return (0);

}
