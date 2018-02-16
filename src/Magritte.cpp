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
    double mean_intensity[NCELLS*TOT_NRAD];   // mean intensity for a ray
    double pop[NCELLS*TOT_NLEV];              // level population n_i


# elif (!FIXED_NCELLS && (INPUT_FORMAT == '.txt'))

    long ncells = get_NCELLS_txt (inputfile);

    CELL *cell = new CELL[ncells];
    double *mean_intensity = new double[ncells*TOT_NRAD];   // mean intensity for a ray
    double *pop            = new double[ncells*TOT_NLEV];   // level population n_i

# elif (!FIXED_NCELLS && (INPUT_FORMAT == '.vtu'))

    long ncells = get_NCELLS_vtu (inputfile);

    CELL *cell = new CELL[ncells];
    double *mean_intensity = new double[ncells*TOT_NRAD];   // mean intensity for a ray
    double *pop            = new double[ncells*TOT_NLEV];   // level population n_i

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


  printf ("(Magritte): reading chemistry data files\n\n");


  // Read chemical species data

  SPECIES species[NSPEC];

  read_species (spec_datafile, species);


  // Initialize abundances in each cell with initial abundances read above

  initialize_abundances (NCELLS, cell, species);


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

# endif




  // CALCULATE EXTERNAL RADIATION FIELD
  // __________________________________


  printf ("(Magritte): calculating external radiation field \n");


  double G_external[3];   // external radiation field vector

  G_external[0] = G_EXTERNAL_X;
  G_external[1] = G_EXTERNAL_Y;
  G_external[2] = G_EXTERNAL_Z;


  // Calculate radiation surface

  calc_rad_surface (NCELLS, cell, G_external);

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

  calc_column_density (NCELLS, cell, column_tot, NSPEC-1);
  // write_double_2("column_tot", "", NCELLS, NRAYS, column_tot);


  // Calculate visual extinction

  calc_AV (NCELLS, cell, column_tot);


  // Calculcate UV field

  calc_UV_field (NCELLS, cell);


# if (!RESTART)

    // Make a guess for gas temperature based on UV field

    guess_temperature_gas (NCELLS, cell);


    // Calculate the dust temperature

    calc_temperature_dust (NCELLS, cell);

# endif


  printf ("(Magritte): gas temperature guessed and dust temperature calculated \n\n");




  // Specify grid boundaries

  double x_min = X_MIN;
  double x_max = X_MAX;
  double y_min = Y_MIN;
  double y_max = Y_MAX;
  double z_min = Z_MIN;
  double z_max = Z_MAX;

  double threshold = THRESHOLD;   // keep cells if rel_density_change > threshold


  // Reduce grid

  long ncells_red1 = reduce (ncells, cell, threshold, x_min, x_max, y_min, y_max, z_min, z_max);

  CELL *cell_red1 = new CELL[ncells_red1];

  initialize_reduced_grid (ncells_red1, cell_red1, ncells, cell);


  long ncells_red2 = reduce (ncells_red1, cell_red1, threshold, x_min, x_max, y_min, y_max, z_min, z_max);

  CELL *cell_red2 = new CELL[ncells_red2];

  initialize_reduced_grid (ncells_red2, cell_red2, ncells_red1, cell_red1);


  long ncells_red3 = reduce (ncells_red2, cell_red2, threshold, x_min, x_max, y_min, y_max, z_min, z_max);

  CELL *cell_red3 = new CELL[ncells_red3];

  initialize_reduced_grid (ncells_red3, cell_red3, ncells_red2, cell_red2);



  double *mean_intensity_red3 = new double[ncells_red3*TOT_NRAD];   // mean intensity for a ray
  double *pop_red3            = new double[ncells_red3*TOT_NLEV];   // level population n_i



  // CALCULATE TEMPERATURE
  // _____________________

  thermal_balance (ncells_red3, cell_red3, species, reaction, line_species, pop_red3, mean_intensity_red3, &timers);

  // thermal_balance (ncells, cell, species, reaction, line_species, pop, mean_intensity, &timers);


  // Interpolate reduced grid back to original grid

  interpolate (ncells_red3, cell_red3, ncells_red2, cell_red2);
  interpolate (ncells_red2, cell_red2, ncells_red1, cell_red1);
  interpolate (ncells_red1, cell_red1, ncells, cell);




  double *mean_intensity_red = new double[ncells*TOT_NRAD];   // mean intensity for a ray
  double *pop_red            = new double[ncells*TOT_NLEV];   // level population n_i


  // CALCULATE TEMPERATURE
  // _____________________

  thermal_balance (ncells, cell, species, reaction, line_species, pop, mean_intensity, &timers);




  delete [] cell_red2;
  delete [] cell_red1;


  timers.total.stop();


  printf ("(Magritte): Total calculation time is %lE\n\n", timers.total.duration);




  // WRITE OUTPUT
  // ____________


  printf ("(Magritte): writing output \n");


# if   (INPUT_FORMAT == '.vtu')

    write_vtu_output (NCELLS, cell, inputfile);

# elif (INPUT_FORMAT == '.txt')

    write_txt_output (NCELLS, cell, line_species, pop, mean_intensity);

# endif


  write_performance_log (timers);


  printf ("(Magritte): output written \n\n");




# if (!FIXED_NCELLS)

    delete [] cell;
    delete [] column_tot;
    delete [] mean_intensity;
    delete [] pop;

# endif




  printf ("(Magritte): done \n\n");


  return (0);

}
