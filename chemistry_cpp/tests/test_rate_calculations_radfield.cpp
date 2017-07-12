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
#include "../src/read_chemdata.cpp"
#include "../src/reaction_rates.cpp"



#define EPS 1.0E-3                                    /* fractional error allowed in computation */




/* Test rate_calculations_radfield functions                                                     */
/*-----------------------------------------------------------------------------------------------*/

TEST_CASE("Test rate_calculations_radfield functions"){


  /* Set up the problem */

  ngrid = 10;

  int n, spec, reac, ray;                                                               /* index */


  /* Specify the file containing the species */

  std::string specdatafile = "../data/species_reduced.d"; /* path to data file containing the species */


  /* Specify the file containing the reactions */

  std::string reacdatafile = "../data/rates_reduced.d"; /* path to data file containing the reactions */


  /* Get the number of species from the species data file */

  nspec = get_nspec(specdatafile);
  printf("(read_chemdata): number of species   %*d\n", MAX_WIDTH, nspec);


  /* Get the number of reactions from the reactions data file */

  nreac = get_nreac(reacdatafile);
  printf("(read_chemdata): number of reactions %*d\n", MAX_WIDTH, nreac);


  /* Declare the species and reactions */

  SPECIES species[nspec];     /* species has a symbol (.sym), mass (.mass), and abundance (.abn) */


  REACTIONS reaction[nreac];         /* reaction has reactants (.R1, .R2, .R3), reaction products \
                                (.P1, .P2, .P3, .P4), alpha (.alpha), beta (.beta), gamma (.gamma)\
                                minimal temperature (.RT_min) and maximal temperature (RT_max) */



  void read_species(string specdatafile, SPECIES *species);

  read_species(specdatafile, species);


  void read_reactions(string reacdatafile, REACTIONS *reaction);

  read_reactions(reacdatafile, reaction);



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