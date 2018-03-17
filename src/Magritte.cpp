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

#include "declarations.hpp"
#include "definitions.hpp"

#include "../setup/setup_data_tools.hpp"

#include "initializers.hpp"
#include "species_tools.hpp"

#include "read_input.hpp"
#include "read_chemdata.hpp"
#include "read_linedata.hpp"

#include "ray_tracing.hpp"
#include "reduce.hpp"
#include "bound.hpp"

#include "calc_rad_surface.hpp"
#include "calc_column_density.hpp"
#include "calc_AV.hpp"
#include "calc_UV_field.hpp"
#include "calc_temperature_dust.hpp"

#include "thermal_balance.hpp"
#include "write_output.hpp"
#include "write_txt_tools.hpp"
#include "write_vtu_tools.hpp"

// #include "../pySetup/test2.hpp"


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


  printf ("(Magritte): reading grid input file\n\n");


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


  // // HACK FOR WARD's LIME SPHERES
  //
  // // Find boundary radius
  //
  // double boundary_radius2 = 0.0;
  //
  // for (long n = 0; n < NCELLS; n++)
  // {
  //   double r2 = cell[n].x*cell[n].x + cell[n].y*cell[n].y + cell[n].z*cell[n].z;
  //
  //   if (r2 > boundary_radius2)
  //   {
  //     boundary_radius2 = r2;
  //   }
  // }
  //
  // long nboundary_cells = 0;
  //
  // for (long n = 0; n < NCELLS; n++)
  // {
  //   double r2 = cell[n].x*cell[n].x + cell[n].y*cell[n].y + cell[n].z*cell[n].z;
  //
  //   if (r2 >= 0.99*boundary_radius2)
  //   {
  //     cell[n].boundary = true;
  //
  //     nboundary_cells++;
  //   }
  // }
  //
  // printf("nboundary_cells = %ld\n", nboundary_cells);



  printf ("(Magritte): grid input read\n\n");





  // CREATE HEALPIXVECTORS
  // _____________________


  printf ("(Magritte): Creating HEALPix vectors\n\n");

  const HEALPIXVECTORS healpixvectors;   // (created by constructor)

  printf ("(Magritte): HEALPix vectors created\n\n");




  // READ CHEMISTRY DATA
  // ___________________


  printf ("(Magritte): reading chemistry data files\n\n");


  // Read chemical species data

  SPECIES species[NSPEC];

  read_species (spec_datafile, species);


# if (!RESTART)
  // Initialize abundances in each cell with initial abundances read above

  initialize_abundances (NCELLS, cell, species);

# endif


  // Read chemical reaction data

  REACTION reaction[NREAC];

  read_reactions (reac_datafile, reaction);


  printf ("(Magritte): chemistry data read\n\n");




  // READ LINE DATA FOR EACH LINE PRODUCING SPECIES
  // ______________________________________________


  printf ("(Magritte): reading line data file\n");


  // Read line data files stored in list(!) line_data

  LINE_SPECIES line_species;

  read_linedata (line_datafile, &line_species, species);


  // for (int i=0; i<TOT_NLEV2; i++)
  // {
  //
  //   double a = TESTB_coeff[i];
  //   double b = line_species.B_coeff[i];
  //
  //   double nill = 2.0 * (a - b);
  //   if (nill != 0.0)
  //   {
  //     nill = nill / (a + b);
  //   }
  //   // double nill = 2.0 * (TESTA_coeff[i] - line_species.A_coeff[i]);
  //
  //   printf("nill = %lE\n", nill);
  //   // if (n != 0.0)
  //   // {}
  // }

  // for (int i=0; i<TOT_CUM_TOT_NCOLTRANTEMP; i++)
  // {
  //
  //   double a = TESTC_data[i];
  //   double b = line_species.C_data[i];
  //
  //   double nill = 2.0 * (a - b);
  //   if (nill != 0.0)
  //   {
  //     nill = nill / (a + b);
  //   }
  //   // double nill = 2.0 * (TESTA_coeff[i] - line_species.A_coeff[i]);
  //
  //   printf("nill = %lE\n", nill);
  //   // if (n != 0.0)
  //   // {}
  // }
  //
  // return(0);


  printf ("(Magritte): line data read \n\n");




  // FIND NEIGHBORING CELLS
  // ______________________


  printf ("(Magritte): finding neighboring cells \n");


  // Find neighboring cells for each cell

  find_neighbors (NCELLS, cell, healpixvectors);


  // Find endpoint of each ray for each cell

  find_endpoints (NCELLS, cell, healpixvectors);


  printf ("(Magritte): neighboring cells found \n\n");

  // 
  // printf("thing    %lE\n", cell[123].Z[0]);
  // printf("thing    %lE\n", cell[123].Z[1]);
  // printf("thing    %lE\n", cell[123].Z[2]);
  //
  // printf("temp gas %lE\n", cell[123].temperature.gas);
  //
  // printf("x        %lE\n", cell[123].x);
  //
  // printf("denisty  %lE\n", cell[123].density);
  //
  //   printf("763 n_neighbors %ld\n", cell[763].n_neighbors);
  //
  //   printf("763 pos %lE\n", cell[763].x);
  //   printf("763 pos %lE\n", cell[763].y);
  //   printf("763 pos %lE\n", cell[763].z);
  //
  //   if(!cell[763].boundary)
  //   {
  //     printf(" not On boundary!\n");
  //   }


  // write_grid("", NCELLS, cell);


  // return(0);


  // CALCULATE EXTERNAL RADIATION FIELD
  // __________________________________


  printf ("(Magritte): calculating external radiation field \n");


  double G_external[3];   // external radiation field vector

  G_external[0] = G_EXTERNAL_X;
  G_external[1] = G_EXTERNAL_Y;
  G_external[2] = G_EXTERNAL_Z;


  // Calculate radiation surface

  calc_rad_surface (NCELLS, cell, healpixvectors, G_external);

  printf ("(Magritte): external radiation field calculated \n\n");




  // MAKE GUESS FOR GAS TEMPERATURE AND CALCULATE DUST TEMPERATURE
  // _____________________________________________________________


  printf("(Magritte): making a guess for gas temperature and calculating dust temperature \n");


# if (FIXED_NCELLS)

    double column_tot[NCELLS*NRAYS];   // total column density

# else

    double *column_tot = new double[ncells*NRAYS];   // total column density

# endif


  initialize_double_array (NCELLS*NRAYS, column_tot);


  // Calculate total column density

  calc_column_density (NCELLS, cell, healpixvectors, column_tot, NSPEC-1);
  // write_double_2("column_tot", "", NCELLS, NRAYS, column_tot);


  // Calculate visual extinction

  calc_AV (NCELLS, cell, column_tot);


  // Calculcate UV field

  calc_UV_field (NCELLS, cell);


# if (!RESTART)

    // Make a guess for gas temperature based on UV field

    guess_temperature_gas (NCELLS, cell);

    initialize_previous_temperature_gas(NCELLS, cell);


    // Calculate the dust temperature

    calc_temperature_dust (NCELLS, cell);

# endif

// for (long n=0; n<NCELLS; n++)
// {
  // if (cell[n].n_neighbors <= 1)
  // printf("%ld\n", n);
// }


// return(0);

  printf ("(Magritte): gas temperature guessed and dust temperature calculated \n\n");



  // Reduce grid

  // long ncells_red1 = reduce (ncells, cell);
  // CELL *cell_red1 = new CELL[ncells_red1];
  // initialize_reduced_grid (ncells_red1, cell_red1, ncells, cell);
  //
  //
  // long ncells_red2 = reduce (ncells_red1, cell_red1);
  // CELL *cell_red2 = new CELL[ncells_red2];
  // initialize_reduced_grid (ncells_red2, cell_red2, ncells_red1, cell_red1);
  //
  //
  // long ncells_red3 = reduce (ncells_red2, cell_red2);
  // CELL *cell_red3 = new CELL[ncells_red3];
  // initialize_reduced_grid (ncells_red3, cell_red3, ncells_red2, cell_red2);
  //
  //
  // long ncells_red4 = reduce (ncells_red3, cell_red3);
  // CELL *cell_red4 = new CELL[ncells_red4];
  // initialize_reduced_grid (ncells_red4, cell_red4, ncells_red3, cell_red3);
  //
  //
  // long ncells_red5 = reduce (ncells_red4, cell_red4);
  // CELL *cell_red5 = new CELL[ncells_red5];
  // initialize_reduced_grid (ncells_red5, cell_red5, ncells_red4, cell_red4);





  // CALCULATE TEMPERATURE
  // _____________________

  // thermal_balance (ncells_red5, cell_red5, species, reaction, line_species, &timers);

  // interpolate (ncells_red5, cell_red5, ncells_red4, cell_red4);


  // thermal_balance (ncells_red4, cell_red4, species, reaction, line_species, &timers);

  // interpolate (ncells_red4, cell_red4, ncells_red3, cell_red3);


  // thermal_balance (ncells_red3, cell_red3, species, reaction, line_species, &timers);

  // interpolate (ncells_red3, cell_red3, ncells_red2, cell_red2);

  // thermal_balance (ncells_red2, cell_red2, species, reaction, line_species, &timers);

  // interpolate (ncells_red2, cell_red2, ncells_red1, cell_red1);

  // thermal_balance (ncells_red1, cell_red1, species, reaction, line_species, &timers);

  // interpolate (ncells_red1, cell_red1, ncells, cell);

  thermal_balance (ncells, cell, healpixvectors, species, reaction, line_species, &timers);


  // delete [] cell_red5;
  // delete [] cell_red4;
  // delete [] cell_red3;
  // delete [] cell_red2;
  // delete [] cell_red1;


  timers.total.stop();


  printf ("(Magritte): Total calculation time = %lE\n\n", timers.total.duration);
  printf ("(Magritte): - time in chemistry = %lE\n\n", timers.chemistry.duration);
  printf ("(Magritte): - time in level_pop = %lE\n\n", timers.level_pop.duration);




  // WRITE OUTPUT
  // ____________


  printf ("(Magritte): writing output \n");


# if   (INPUT_FORMAT == '.vtu')

    write_vtu_output (NCELLS, cell, inputfile);

# elif (INPUT_FORMAT == '.txt')

    write_txt_output (NCELLS, cell, line_species);

# endif

  write_level_populations("", NCELLS, cell, line_species);


  write_performance_log (timers);


  printf ("(Magritte): output written \n\n");




# if (!FIXED_NCELLS)

    delete [] cell;
    delete [] column_tot;

# endif




  printf ("(Magritte): done \n\n");


  return (0);

}
