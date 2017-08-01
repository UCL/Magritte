/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_spline: Calculate splines for tabulated functions                                        */
/*                                                                                               */
/* (based on spline in 3D-PDR)                                                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>

#include <string>
using namespace std;

#include "catch.hpp"

#include "../src/definitions.hpp"
#include "../src/data_tools.cpp"
#include "../src/read_chemdata.cpp"
#include "../src/reaction_rates.cpp"



#define EPS 1.0E-3                                    /* fractional error allowed in computation */




/* Test rate_calculations_radfield functions                                                     */
/*-----------------------------------------------------------------------------------------------*/

TEST_CASE("Test rate_calculations_radfield functions"){


  int n, spec, reac, ray;                                                               /* index */


  void read_species(string spec_datafile);

  read_species(spec_datafile);


  void read_reactions(string reac_datafile);

  read_reactions(reac_datafile);



  /* Test X_lambda */

  SECTION("Test X_lambda calculator"){

    double lambda;

    double X_lambda(double lambda);

    int n = 250;                                               /* number of interpolating points */



    /* Write the X_lambda values to a text file (for testing) */

    FILE *xl = fopen("X_lambda_spline.txt", "w");

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

    double column_CO;
    double column_H2;;

    double self_shielding_CO(double column_CO, double column_H2);

    int n = 20;                                                /* number of interpolating points */



    /* Write the self_shielding_CO values to a text file (for testing) */

    FILE *sCO = fopen("self_shielding_CO_spline.txt", "w");

    if (sCO == NULL){

      printf("Error opening file!\n");
      exit(1);
    }

    for (int i=0; i<n; i++){

      for (int j=0; j<n; j++){

        column_CO = pow(10.0, (19.0-12.0)/n*i + 12.0 );
        column_H2 = pow(10.0, (23.0-18.0)/n*j + 18.0 );

        fprintf( sCO, "%lE\t%lE\t%lE\n", log10(column_CO),
                                         log10(column_H2),
                                         log10(self_shielding_CO(column_CO, column_H2)) );
      }
    }

    fclose(sCO);


    CHECK( 1==1 );
  }


}

/*-----------------------------------------------------------------------------------------------*/