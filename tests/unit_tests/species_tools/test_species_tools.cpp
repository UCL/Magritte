/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_species_tools: tests the tools to handle SPECIES objects as defined in species_tools     */
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
#include "../../../src/species_tools.hpp"



TEST_CASE("Test species_tools"){


  species[0].sym = "dummy";
  species[0].abn[0] = 0.1;
  species[1].sym = "H";
  species[1].abn[0] = 0.1;
  species[2].sym = "H2";
  species[2].abn[0] = 0.1;
  species[3].sym = "He+";
  species[3].abn[0] = 0.4;
  species[4].sym = "Li";
  species[4].abn[0] = 0.1;
  species[5].sym = "e-";
  species[5].abn[0] = 0.0;


  CHECK( get_canonical_name("e") == "e-" );

  CHECK( get_species_nr("e") == 5 );

  CHECK( get_species_nr("Li") == 4 );

  CHECK( get_species_nr("oH2") == 2 );

  CHECK( get_species_nr("pH2") == 2 );

  CHECK( get_species_nr("o-H2") == 2 );

  CHECK( get_species_nr("p-H2") == 2 );

  char list[5];


  list[0] = check_ortho_para("B");

  // cout << get_species_nr("e") << "\n";

  cout << "charge is " << get_charge("H2O+") << "\n";
  cout << "charge is " << get_charge("H2O-") << "\n";
  cout << "charge is " << get_charge("H2O+++-+") << "\n";

  cout << "electron abundance is " << get_electron_abundance(0) << "\n";

  cout << NSPEC << "\n";


}

/*-----------------------------------------------------------------------------------------------*/
