/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_radfield_tools: teat the radfield tools functions                                        */
/*                                                                                               */
/* (based on spline in 3D-PDR)                                                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <iostream>
using namespace std;

#include "catch.hpp"

#include "../../src/declarations.hpp"
#include "../../src/definitions.hpp"
#include "../../src/data_tools.hpp"
#include "../../src/read_chemdata.hpp"
#include "../../src/radfield_tools.hpp"




#define EPS 1.0E-3                                    /* fractional error allowed in computation */




/* Test radfield_tools functions                                                                 */
/*-----------------------------------------------------------------------------------------------*/

TEST_CASE("Test rate_calculations_radfield functions"){


  int n, spec, reac, ray;                                                               /* index */



  /* Test X_lambda */

  SECTION("Test X_lambda calculator"){


    /* Since the executables are now in the directory /tests, we have to change the paths */

    string spec_datafile1  = "../" + spec_datafile;

    string reac_datafile1  = "../" + reac_datafile;


    /* Read the chemical data */

    read_species(spec_datafile1);

    read_reactions(reac_datafile1);


    double lambda;                                                                 /* wavelength */

    int n = 250;                                               /* number of interpolating points */



    /* Write the X_lambda values to a text file (for testing) */

    FILE *xl = fopen("test_output/X_lambda_spline.txt", "w");

    if (xl == NULL){

      printf("Error opening file!\n");
      exit(1);
    }

    for (int i=0; i<n; i++){

      lambda = (35000.0-800.0)/n*i + 800.0;

      fprintf( xl, "%lE\t%lE\n", lambda, X_lambda(lambda) );
    }

    fclose(xl);


    CHECK( 1==1 );
  }



  /* Test self_shielding_CO */

  SECTION("Test self_shielding_CO"){


    /* Since the executables are now in the directory /tests, we have to change the paths */

    string spec_datafile2  = "../" + spec_datafile;

    string reac_datafile2  = "../" + reac_datafile;


    /* Read the chemical data */

    read_species(spec_datafile2);

    read_reactions(reac_datafile2);


    double column_CO_t;                                                     /* CO column density */
    double column_H2_t;                                                     /* H2 column density */

    int n = 20;                                                /* number of interpolating points */



    /* Write the self_shielding_CO values to a text file (for testing) */

    FILE *sCO = fopen("test_output/self_shielding_CO_spline.txt", "w");

    if (sCO == NULL){

      printf("Error opening file!\n");
      exit(1);
    }

    for (int i=0; i<n; i++){

      for (int j=0; j<n; j++){

        column_CO_t = pow(10.0, (19.0-12.0)/n*i + 12.0 );
        column_H2_t = pow(10.0, (23.0-18.0)/n*j + 18.0 );

        fprintf( sCO, "%lE\t%lE\t%lE\n", log10(column_CO_t),
                                         log10(column_H2_t),
                                         log10(self_shielding_CO(column_CO_t, column_H2_t)) );
      }
    }

    fclose(sCO);


    CHECK( 1==1 );
  }


}

/*-----------------------------------------------------------------------------------------------*/
