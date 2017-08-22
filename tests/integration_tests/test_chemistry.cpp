/* Frederik De Ceuster - University College London                                               */
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

#include "../../src/declarations.hpp"
#include "../../src/definitions.hpp"
#include "../../src/data_tools.cpp"
#include "../../src/read_chemdata.cpp"
#include "../../src/species_tools.cpp"
#include "../../reaction_rates.hpp"
//  #include "sundials/rate_equation_solver.hpp"



TEST_CASE("Test chemistry"){


  SECTION("Check reading the files"){


  /* Since the executables are now in the directory /tests, we have to change the paths */

  spec_datafile  = "../" + spec_datafile;

  reac_datafile  = "../" + reac_datafile;


  /* Read the species (and their initial abundances) */

  read_species(spec_datafile);


  e_nr    = get_species_nr("e-");                       /* species nr corresponding to electrons */
  H2_nr   = get_species_nr("H2");                              /* species nr corresponding to H2 */
  HD_nr   = get_species_nr("HD");                              /* species nr corresponding to HD */
  C_nr    = get_species_nr("C");                                /* species nr corresponding to C */
  H_nr    = get_species_nr("H");                                /* species nr corresponding to H */
  H2x_nr  = get_species_nr("H2+");                            /* species nr corresponding to H2+ */
  HCOx_nr = get_species_nr("HCO+");                          /* species nr corresponding to HCO+ */
  H3x_nr  = get_species_nr("H3+");                            /* species nr corresponding to H3+ */
  H3Ox_nr = get_species_nr("H3O+");                          /* species nr corresponding to H3O+ */
  Hex_nr  = get_species_nr("He+");                            /* species nr corresponding to He+ */
  CO_nr   = get_species_nr("CO");                              /* species nr corresponding to CO */


  /* Read the reactions */

  read_reactions(reac_datafile);


  for(int spec=0; spec<NSPEC; spec++){

    printf( "%s\t%.2lE\t%.1lf\n", species[spec].sym.c_str(), species[spec].abn, species[spec].mass );
  }

  for(int reac=0; reac<NREAC; reac++){

    printf( "%-3s + %-3s + %-3s  ->  %-3s + %-3s + %-3s + %-3s \n"
            "with alpha = %-10.2lE, beta = %-10.2lE, gamma = %-10.2lE \t"
            "RT_min = %-10.2lE, RT_max = %-10.2lE, duplicates = %d \n",
            reaction[reac].R1.c_str(), reaction[reac].R2.c_str(), reaction[reac].R3.c_str(),
            reaction[reac].P1.c_str(), reaction[reac].P2.c_str(), reaction[reac].P3.c_str(), reaction[reac].P4.c_str(),
            reaction[reac].alpha, reaction[reac].beta, reaction[reac].gamma,
            reaction[reac].RT_min, reaction[reac].RT_max,
            reaction[reac].dup );
  }



    CHECK( 1==1 );
  }


}

/*-----------------------------------------------------------------------------------------------*/
