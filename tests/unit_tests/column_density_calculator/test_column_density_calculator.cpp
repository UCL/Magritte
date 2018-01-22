// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <string>
#include <iostream>

#include "catch.hpp"

#include "../../src/declarations.hpp"
#include "../../src/definitions.hpp"
#include "../../src/initializers.hpp"
#include "../../src/read_input.hpp"
#include "../../src/ray_tracing.hpp"
#include "../../src/species_tools.hpp"
#include "../../src/data_tools.hpp"
#include "../../src/read_chemdata.hpp"
#include "../../src/calc_column_density.hpp"


#define EPS 1.0E-7


TEST_CASE("1D regular grid")
{

  // SET UP TEST DATA
  // ________________


  // Since executables are in directory /tests, we have to change paths

  test_inputfile        = "../" + inputfile;
  test_spec_datafile    = "../" + spec_datafile;
  test_line_datafile[0] = "../" + line_datafile[0];


  // Read input file

  CELL cell[NCELLS]; 

  read_input (inputfile, cell);


  // Read species file

  SPECIES species[NSPEC];

  read_species (spec_datafile, species);





  // Define and initialize

  double column_tot[NCELLS*NRAYS];   // total column density

  initialize_double_array (column_tot, NCELLS*NRAYS);

  double AV[NCELLS*NRAYS];           // Visual extinction

  initialize_double_array (AV, NCELLS*NRAYS);



  calc_column_density (NCELLS, cell, evalpoint, column_density, spec);


  CHECK( 1==1 );

}
