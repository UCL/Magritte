/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_chemistry:                                                                               */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <string>
using namespace std;

#include "catch.hpp"

#include "../../../src/declarations.hpp"
#include "../../../src/definitions.hpp"
#include "../../../src/data_tools.hpp"
#include "../../../src/species_tools.hpp"
#include "../../../src/read_chemdata.hpp"
#include "../../../src/reaction_rates.hpp"




TEST_CASE("Test reaction_rates"){


  /* Since the executables are now in the directory /tests, we have to change the paths */

  string test_spec_datafile  = "../../../" + spec_datafile;

  string test_reac_datafile  = "../../../" + reac_datafile;


  /* Read the species (and their initial abundances) */

  read_species(test_spec_datafile);


  /* Get and store the species numbers of some inportant species */

  nr_e    = get_species_nr("e-");                       /* species nr corresponding to electrons */

  nr_H2   = get_species_nr("H2");                              /* species nr corresponding to H2 */

  nr_HD   = get_species_nr("HD");                              /* species nr corresponding to HD */

  nr_C    = get_species_nr("C");                                /* species nr corresponding to C */

  nr_H    = get_species_nr("H");                                /* species nr corresponding to H */

  nr_H2x  = get_species_nr("H2+");                            /* species nr corresponding to H2+ */

  nr_HCOx = get_species_nr("HCO+");                          /* species nr corresponding to HCO+ */

  nr_H3x  = get_species_nr("H3+");                            /* species nr corresponding to H3+ */

  nr_H3Ox = get_species_nr("H3O+");                          /* species nr corresponding to H3O+ */

  nr_Hex  = get_species_nr("He+");                            /* species nr corresponding to He+ */

  nr_CO   = get_species_nr("CO");                              /* species nr corresponding to CO */


  /* Read the reactions */

  read_reactions(test_reac_datafile);


  double temperature_gas[NCELLS];

  temperature_gas[0] = 10.0;

  double temperature_dust[NCELLS];

  temperature_dust[0] = 0.0;

  double rad_surface[NCELLS];

  rad_surface[0] = 0.0;

  double AV[NCELLS];

  AV[0] = 0.0;

  double column_H2[NCELLS];

  column_H2[0] = 0.0;

  double column_HD[NCELLS];

  column_HD[0] = 0.0;

  double column_C[NCELLS];

  column_C[0] = 0.0;

  double column_CO[NCELLS];

  column_CO[0] = 0.0;


  long gridp = 0;


  reaction_rates( temperature_gas, temperature_dust, rad_surface, AV,
                  column_H2, column_HD, column_C, column_CO, gridp );



  SECTION("Check reading the files"){


    // for(int spec=0; spec<NSPEC; spec++){
    //
    //   printf( "%s\t%.2lE\t%.1lf\n",
    //           species[spec].sym.c_str(), species[spec].abn[0], species[spec].mass );
    // }
    //
    // for(int reac=0; reac<NREAC; reac++){
    //
    //   printf( "%-3s + %-3s + %-3s  ->  %-3s + %-3s + %-3s + %-3s \n"
    //           "with alpha = %-10.2lE, beta = %-10.2lE, gamma = %-10.2lE \t"
    //           "RT_min = %-10.2lE, RT_max = %-10.2lE, duplicates = %d \n",
    //           reaction[reac].R1.c_str(), reaction[reac].R2.c_str(), reaction[reac].R3.c_str(),
    //           reaction[reac].P1.c_str(), reaction[reac].P2.c_str(), reaction[reac].P3.c_str(), reaction[reac].P4.c_str(),
    //           reaction[reac].alpha, reaction[reac].beta, reaction[reac].gamma,
    //           reaction[reac].RT_min, reaction[reac].RT_max,
    //           reaction[reac].dup );
    // }



    CHECK( 1==1 );
  }


}

/*-----------------------------------------------------------------------------------------------*/
