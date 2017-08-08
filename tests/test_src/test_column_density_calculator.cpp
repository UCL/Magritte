/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_column_density_calculator: tests the column_density_calculator function                  */
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
#include "../../src/read_input.hpp"
#include "../../src/create_healpixvectors.hpp"
#include "../../src/ray_tracing.hpp"
#include "../../src/species_tools.hpp"
#include "../../src/data_tools.hpp"
#include "../../src/read_chemdata.hpp"
#include "../../src/column_density_calculator.hpp"


#define EPS 1.0E-7


TEST_CASE("1D regular grid"){





  /*   SET UP TEST DATA                                                                          */
  /*_____________________________________________________________________________________________*/


  double unit_healpixvector[3*NRAYS];            /* array of HEALPix vectors for each ipix pixel */

  long   antipod[NRAYS];                                     /* gives antipodal ray for each ray */



  /* Since the executables are now in the directory /tests, we have to change the paths */

  grid_inputfile   = "../" + grid_inputfile;
  spec_datafile    = "../" + spec_datafile;
  line_datafile[0] = "../" + line_datafile[0];



  /* Define grid (using types defined in definitions.h) */

  GRIDPOINT gridpoint[NGRID];                                                     /* grid points */

  EVALPOINT evalpoint[NGRID*NGRID];                     /* evaluation points for each grid point */



  /* Initialize */

  for (long n1=0; n1<NGRID; n1++){

    for (long n2=0; n2<NGRID; n2++){

      evalpoint[GINDEX(n1,n2)].dZ  = 0.0;
      evalpoint[GINDEX(n1,n2)].Z   = 0.0;
      evalpoint[GINDEX(n1,n2)].vol = 0.0;

      evalpoint[GINDEX(n1,n2)].ray = 0;
      evalpoint[GINDEX(n1,n2)].nr  = 0;

      evalpoint[GINDEX(n1,n2)].eqp = 0;

      evalpoint[GINDEX(n1,n2)].onray = false;

      key[GINDEX(n1,n2)] = 0;
    }

    for (long r=0; r<NRAYS; r++){

      raytot[RINDEX(n1,r)]      = 0;
      cum_raytot[RINDEX(n1,r)]  = 0;
    }

  }


  /* Read input file */

  read_input(grid_inputfile, gridpoint);



  /* Read the species and their abundances */

  read_species(spec_datafile);



  /* Setup the (unit) HEALPix vectors */

  create_healpixvectors(unit_healpixvector, antipod);



  /* Trace the rays */

  ray_tracing(unit_healpixvector, gridpoint, evalpoint);



  /* Test the column_density_calculator */

  double column_density[NGRID*NSPEC*NRAYS];       /* column density for each spec, ray and gridp */

  double AV[NGRID*NRAYS];                       /* Visual extinction (only takes into account H) */

  metallicity = 1.0;


  /* Initialization */

  for (int n=0; n<NGRID; n++){

    for (int r=0; r<NRAYS; r++){

      for (int spec=0; spec<NSPEC; spec++){

        column_density[GRIDSPECRAY(n,spec,r)] = 0.0;
      }

    }
  }


  /*_____________________________________________________________________________________________*/





  int spec =0;

  column_density_calculator( gridpoint, evalpoint, column_density, spec );


  CHECK( 1==1 );

}
