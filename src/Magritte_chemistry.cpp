// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

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

#include "ray_tracing.hpp"

#include "calc_rad_surface.hpp"
#include "calc_column_density.hpp"
#include "calc_AV.hpp"
#include "calc_UV_field.hpp"
#include "calc_temperature_dust.hpp"
#include "chemistry.hpp"

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


  // Initialize abundances in each cell

  initialize_abundances (NCELLS, cell, species);


  // Get and store species numbers of some inportant species


  // define as nr_e etc
  //

  nr_e    = get_species_nr (species, "e-");     // species nr corresponding to electrons
  nr_H2   = get_species_nr (species, "H2");     // species nr corresponding to H2
  nr_HD   = get_species_nr (species, "HD");     // species nr corresponding to HD
  nr_C    = get_species_nr (species, "C");      // species nr corresponding to C
  nr_H    = get_species_nr (species, "H");      // species nr corresponding to H
  nr_H2x  = get_species_nr (species, "H2+");    // species nr corresponding to H2+
  nr_HCOx = get_species_nr (species, "HCO+");   // species nr corresponding to HCO+
  nr_H3x  = get_species_nr (species, "H3+");    // species nr corresponding to H3+
  nr_H3Ox = get_species_nr (species, "H3O+");   // species nr corresponding to H3O+
  nr_Hex  = get_species_nr (species, "He+");    // species nr corresponding to He+
  nr_CO   = get_species_nr (species, "CO");     // species nr corresponding to CO


  // Read chemical reaction data

  REACTION reaction[NREAC];

  read_reactions (reac_datafile, reaction);


  printf ("(Magritte): chemistry data read \n\n");




  // DECLARE AND INITIALIZE LINE VARIABLES
  // _____________________________________


  printf ("(Magritte): declaring and initializing line variables \n");


  // Define line related variables

  int irad[TOT_NRAD];            // level index of radiative transition

  initialize_int_array (TOT_NRAD, irad);

  int jrad[TOT_NRAD];            // level index of radiative transition

  initialize_int_array (TOT_NRAD, jrad);

  double energy[TOT_NLEV];       // energy of level

  initialize_double_array (TOT_NLEV, energy);

  double weight[TOT_NLEV];       // statistical weight of level

  initialize_double_array (TOT_NLEV, weight);

  double frequency[TOT_NLEV2];   // frequency corresponing to i -> j transition

  initialize_double_array (TOT_NLEV2, frequency);

  double A_coeff[TOT_NLEV2];     // Einstein A_ij coefficient

  initialize_double_array (TOT_NLEV2, A_coeff);

  double B_coeff[TOT_NLEV2];     // Einstein B_ij coefficient

  initialize_double_array (TOT_NLEV2, B_coeff);


  // Define collision related variables

  double coltemp[TOT_CUM_TOT_NCOLTEMP];      // Collision temperatures for each partner

  initialize_double_array (TOT_CUM_TOT_NCOLTEMP, coltemp);

  double C_data[TOT_CUM_TOT_NCOLTRANTEMP];   // C_data for each partner, tran. and temp.

  initialize_double_array (TOT_CUM_TOT_NCOLTRANTEMP, C_data);

  int icol[TOT_CUM_TOT_NCOLTRAN];            // level index corresp. to col. transition

  initialize_int_array (TOT_CUM_TOT_NCOLTRAN, icol);

  int jcol[TOT_CUM_TOT_NCOLTRAN];            // level index corresp. to col. transition

  initialize_int_array (TOT_CUM_TOT_NCOLTRAN, jcol);


  printf("(Magritte): data structures are set up \n\n");




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


# if (FIXED_NCELLS)

    double column_H2[NCELLS*NRAYS];   // H2 column density for each ray and cell
    double column_HD[NCELLS*NRAYS];   // HD column density for each ray and cell
    double column_C[NCELLS*NRAYS];    // C column density for each ray and cell
    double column_CO[NCELLS*NRAYS];   // CO column density for each ray and cell

# else

    double *column_H2 = new double[ncells*NRAYS];   // H2 column density for each ray and cell
    double *column_HD = new double[ncells*NRAYS];   // HD column density for each ray and cell
    double *column_C  = new double[ncells*NRAYS];   // C column density for each ray and cell
    double *column_CO = new double[ncells*NRAYS];   // CO column density for each ray and cell

# endif


  initialize_double_array (NCELLS*NRAYS, column_H2);
  initialize_double_array (NCELLS*NRAYS, column_HD);
  initialize_double_array (NCELLS*NRAYS, column_C);
  initialize_double_array (NCELLS*NRAYS, column_CO);


  // Preliminary chemistry iterations

  for (int chem_iteration = 0; chem_iteration < PRELIM_CHEM_ITER; chem_iteration++)
  {
    printf("(Magritte):   chemistry iteration %d of %d \n", chem_iteration+1, PRELIM_CHEM_ITER);


    // Calculate chemical abundances given current temperatures and radiation field

    time_chemistry -= omp_get_wtime();

    chemistry (NCELLS, cell, species, reaction, rad_surface, AV, column_H2, column_HD, column_C, column_CO );

    time_chemistry += omp_get_wtime();


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




  time_total += omp_get_wtime();




  // WRITE OUTPUT
  // ____________


  printf("(Magritte): writing output \n");


# if   (INPUT_FORMAT == '.vtu')

  write_vtu_output (NCELLS, cell, inputfile);

# elif (INPUT_FORMAT == '.txt')

  write_abundances ("", NCELLS, cell);

# endif


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

# endif




  printf ("(Magritte): done \n\n");


  return (0);

}
