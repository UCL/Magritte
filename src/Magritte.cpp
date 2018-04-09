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

#include "initializers.hpp"

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


  // Construct cells

    long ncells = NCELLS_INIT;


# if (FIXED_NCELLS)

    CELLS Cells (NCELLS);    // create CELLS object Cells
    CELLS *cells = &Cells;   // pointer to Cells

# else

    CELLS Cells (ncells);    // create CELLS object Cells
    CELLS *cells = &Cells;   // pointer to Cells

# endif


  cells->initialize ();

  cells->read_input (inputfile);


// return(0);


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





  // CREATE RAYS
  // _____________________


  printf ("(Magritte): Creating HEALPix vectors\n\n");

  const RAYS rays;   // (created by constructor)

  printf ("(Magritte): HEALPix vectors created\n\n");




  // READ CHEMICAL SPECIES DATA
  // __________________________


  printf ("(Magritte): creating species, reading species data\n\n");

  const SPECIES species (spec_datafile);   // (created by constructor)

  printf ("(Magritte): species data read, species created\n\n");


# if (!RESTART)

  // Initialize abundances in each cell with initial abundances

  initialize_abundances (cells, species);

# endif




  // READ CHEMICAL REACTION DATA
  // ___________________________


  printf ("(Magritte): creating reactions, reading reaction data\n\n");

  const REACTIONS reactions(reac_datafile);   // (created by constructor)

  printf ("(Magritte): reaction data read, reactions created\n\n");




  // READ LINE DATA FOR EACH LINE PRODUCING SPECIES
  // ______________________________________________


  printf ("(Magritte): reading line data file\n");

  const LINES lines;   // (values defined in line_data.hpp)

  printf ("(Magritte): line data read \n\n");




  // FIND NEIGHBORING CELLS
  // ______________________


  printf ("(Magritte): finding neighboring cells \n");


  // Find neighboring cells for each cell

  find_neighbors (NCELLS, cells, rays);


  // Find endpoint of each ray for each cell

  find_endpoints (NCELLS, cells, rays);


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

  calc_rad_surface (NCELLS, cells, rays, G_external);

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

  calc_column_density (NCELLS, cells, rays, column_tot, NSPEC-1);
  // write_double_2("column_tot", "", NCELLS, NRAYS, column_tot);


  // Calculate visual extinction

  calc_AV (cells, column_tot);


  // Calculcate UV field

  calc_UV_field (cells);


# if (!RESTART)

    // Make a guess for gas temperature based on UV field

    guess_temperature_gas (NCELLS, cells);

    initialize_double_array_with_scale (NCELLS, cells->temperature_gas_prev, cells->temperature_gas, 0.9);

    initialize_double_array_with_value (NCELLS, cells->temperature_gas_prev, 9.0);

    // initialize_previous_temperature_gas (NCELLS, cells);

    calc_temperature_dust (NCELLS, cells);   // depends on UV field

# endif

// for (long n=0; n<NCELLS; n++)
// {
  // if (cell[n].n_neighbors <= 1)
  // printf("%ld\n", n);
// }


// return(0);

  printf ("(Magritte): gas temperature guessed and dust temperature calculated \n\n");



  // Reduce grid

  long ncells_red1 = reduce (cells);
  CELLS Cells_red1 (ncells_red1);
  CELLS *cells_red1 = &Cells_red1;
  initialize_reduced_grid (cells_red1, cells, rays);



  long ncells_red2 = reduce (cells_red1);
  CELLS Cells_red2 (ncells_red2);
  CELLS *cells_red2 = &Cells_red2;
  initialize_reduced_grid (cells_red2, cells_red1, rays);


  long ncells_red3 = reduce (cells_red2);
  CELLS Cells_red3 (ncells_red3);
  CELLS *cells_red3 = &Cells_red3;
  initialize_reduced_grid (cells_red3, cells_red2, rays);


  // long ncells_red4 = reduce (cells_red3);
  // CELLS Cells_red4 (ncells_red4);
  // CELLS *cells_red4 = &Cells_red4;
  // initialize_reduced_grid (cells_red4, cells_red3, rays);
  //
  //
  // long ncells_red5 = reduce (cells_red4);
  // CELLS Cells_red5 (ncells_red5);
  // CELLS *cells_red5 = &Cells_red5;
  // initialize_reduced_grid (cells_red5, cells_red4, rays);
  //
  //



  // CALCULATE TEMPERATURE
  // _____________________

  printf("ncells = %ld\n", ncells);
  printf("ncells_red = %ld\n", ncells_red1);
  printf("ncells_red = %ld\n", ncells_red2);
  printf("ncells_red = %ld\n", ncells_red3);
  // printf("ncells_red = %ld\n", ncells_red4);

// return(0);

  // thermal_balance (ncells_red5, cells_red5, rays, species, reactions, lines, &timers);

  // write_output (cells_red5, lines);


  // interpolate (cells_red5, cells_red4);

  //
  // thermal_balance (ncells_red4, cells_red4, rays, species, reactions, lines, &timers);
  //
  // interpolate (cells_red4, cells_red3);


  thermal_balance (ncells_red3, cells_red3, rays, species, reactions, lines, &timers);

  interpolate (cells_red3, cells_red2);

  // thermal_balance (ncells_red2, cells_red2, rays, species, reactions, lines, &timers);

  interpolate (cells_red2, cells_red1);

  // thermal_balance (ncells_red1, cells_red1, rays, species, reactions, lines, &timers);

  // return(0);

  // interpolate (ncells_red1, cells_red1, ncells, cells);
  //
  // thermal_balance (ncells, cells, rays, species, reactions, lines, &timers);


  // delete [] cells_red5;
  // delete [] cells_red4;
  // delete [] cells_red3;
  // delete [] cells_red2;
  delete [] cells_red1;


  timers.total.stop();


  printf ("(Magritte): Total calculation time = %lE\n\n", timers.total.duration);
  printf ("(Magritte):    - time in chemistry = %lE\n\n", timers.chemistry.duration);
  printf ("(Magritte):    - time in level_pop = %lE\n\n", timers.level_pop.duration);




  // WRITE OUTPUT
  // ____________


  printf ("(Magritte): writing output \n");

  write_output (cells, lines);

  write_output_log ();

  write_performance_log (timers);


  printf ("(Magritte): output written \n\n");




# if (!FIXED_NCELLS)

    cells->~CELLS();

    delete [] column_tot;

# endif




  printf ("(Magritte): done \n\n");


  return (0);

}
