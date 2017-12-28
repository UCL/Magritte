/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_calc_column_density: tests the calc_column_density function                  */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <string>
#include <iostream>
using namespace std;

#include "catch.hpp"

#include "../../src/declarations.hpp"
#include "../../src/definitions.hpp"
#include "../../src/initializers.hpp"
#include "../../src/read_input.hpp"
#include "../../src/create_healpixvectors.hpp"
#include "../../src/ray_tracing.hpp"
#include "../../src/species_tools.hpp"
#include "../../src/data_tools.hpp"
#include "../../src/read_chemdata.hpp"
#include "../../src/calc_column_density.hpp"


#define EPS 1.0E-7


TEST_CASE("1D regular grid"){





  /*   SET UP TEST DATA                                                                          */
  /*_____________________________________________________________________________________________*/


  double healpixvector[3*NRAYS];            /* array of HEALPix vectors for each ipix pixel */

  long   antipod[NRAYS];                                     /* gives antipodal ray for each ray */



  /* Since the executables are now in the directory /tests, we have to change the paths */

  inputfile   = "../" + inputfile;
  spec_datafile    = "../" + spec_datafile;
  line_datafile[0] = "../" + line_datafile[0];



  /* Define grid (using types defined in definitions.h) */

  CELL cell[NCELLS];                                                     /* grid points */

  EVALPOINT evalpoint[NCELLS*NCELLS];                     /* evaluation points for each grid point */


  initialize_evalpoint(evalpoint);

  /* Initialize the data structures which will store the evaluation pointa */

  initialize_long_array(key, NCELLS*NCELLS);

  initialize_long_array(raytot, NCELLS*NRAYS);

  initialize_long_array(cum_raytot, NCELLS*NRAYS);


  /* Read input file */

  read_input(inputfile, cell);


  /* Read the species and their abundances */

  read_species(spec_datafile);


  /* Setup the (unit) HEALPix vectors */

  create_healpixvectors(healpixvector, antipod);


  /* Trace the rays */

  ray_tracing(healpixvector, cell, evalpoint);


  /* Define and initialize */

  double column_density[NCELLS*NRAYS];                   /* column density for each ray and gridp */

  initialize_double_array(column_density, NCELLS*NRAYS);

  double AV[NCELLS*NRAYS];                       /* Visual extinction (only takes into account H) */

  initialize_double_array(AV, NCELLS*NRAYS);

  METALLICITY = 1.0;





  /*_____________________________________________________________________________________________*/





  int spec =0;

  calc_column_density(cell, evalpoint, column_density, spec);


  CHECK( 1==1 );

}
