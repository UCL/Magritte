// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#include <string>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"


#include "../../../src/declarations.hpp"
#include "../../../src/definitions.hpp"

#include "../../../src/initializers.hpp"

#include "../../../src/ray_tracing.hpp"
#include "../../../src/reduce.hpp"
#include "../../../src/bound.hpp"

#include "../../../src/calc_rad_surface.hpp"
#include "../../../src/calc_column_density.hpp"
#include "../../../src/calc_AV.hpp"
#include "../../../src/calc_UV_field.hpp"
#include "../../../src/calc_temperature_dust.hpp"

#include "../../../src/chemistry.hpp"
#include "../../../src/heating.hpp"

#include "../../../src/write_output.hpp"



#define EPS 1.0E-4


TEST_CASE ("Einstein Collisional coefficient at different temperatures")
{

  // Construct cells

  long ncells = NCELLS_INIT;

  CELLS Cells (ncells);    // create CELLS object Cells
  CELLS *cells = &Cells;   // pointer to Cells


  cells->initialize ();
  cells->read_input (inputfile);


  const RAYS rays;                            // (created by constructor)
  const SPECIES species (spec_datafile);      // (created by constructor)
  const REACTIONS reactions(reac_datafile);   // (created by constructor)
  const LINES lines;                          // (values defined in line_data.hpp)


  find_neighbors (NCELLS, cells, rays);
  find_endpoints (NCELLS, cells, rays);

  initialize_abundances (cells, species);


  double G_external[3];   // external radiation field vector

  G_external[0] = G_EXTERNAL_X;
  G_external[1] = G_EXTERNAL_Y;
  G_external[2] = G_EXTERNAL_Z;


  // Calculate radiation surface

  calc_rad_surface (NCELLS, cells, rays, G_external);


  double column_tot[NCELLS*NRAYS];   // total column density

  initialize_double_array (NCELLS*NRAYS, column_tot);


  // Calculate total column density

  calc_column_density (NCELLS, cells, rays, column_tot, NSPEC-1);


  // Calculate visual extinction

  calc_AV (cells, column_tot);


  // Calculcate UV field

  calc_UV_field (cells);


  // Make a guess for gas temperature based on UV field

  guess_temperature_gas (NCELLS, cells);

  initialize_double_array_with_scale (NCELLS, cells->temperature_gas_prev, cells->temperature_gas, 1.0);

  calc_temperature_dust (NCELLS, cells);   // depends on UV field



  double column_H2[NCELLS*NRAYS];   // H2 column density for each ray and cell
  double column_HD[NCELLS*NRAYS];   // HD column density for each ray and cell
  double column_C[NCELLS*NRAYS];    // C  column density for each ray and cell
  double column_CO[NCELLS*NRAYS];   // CO column density for each ray and cell

  initialize_double_array (NCELLS*NRAYS, column_H2);
  initialize_double_array (NCELLS*NRAYS, column_HD);
  initialize_double_array (NCELLS*NRAYS, column_C);
  initialize_double_array (NCELLS*NRAYS, column_CO);


  chemistry (NCELLS, cells, rays, species, reactions, column_H2, column_HD, column_C, column_CO);
  write_output(cells, lines);


  chemistry (NCELLS, cells, rays, species, reactions, column_H2, column_HD, column_C, column_CO);
  write_output(cells, lines);

  chemistry (NCELLS, cells, rays, species, reactions, column_H2, column_HD, column_C, column_CO);
  write_output(cells, lines);

  chemistry (NCELLS, cells, rays, species, reactions, column_H2, column_HD, column_C, column_CO);
  write_output(cells, lines);

  // double heating_total = heating (cells, species, reactions, 50);

  // printf("heating = %lE\n", heating_total);

  write_output_log ();

  CHECK(true);

}
